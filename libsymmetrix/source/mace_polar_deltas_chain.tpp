// The chain's own staging helpers.  The learned blocks and the chain state
// live in the default execution space (the family's evaluator is templated on
// the model's precision, not on a tier), while the long-range operator group
// names its kernel memory space explicitly. Every host-facing boundary stages
// through a mirror rather than assuming the host is the kernel space, so the
// layer compiles and runs on every backend the framework builds.
namespace symmetrix::mace_polar {

namespace {

template<typename Precision>
std::vector<Precision> tensor_values_chain(const nlohmann::json& value)
{
    return value.at("values").get<std::vector<Precision>>();
}

// The fp64 values of a default-space view.
template<typename Precision>
std::vector<double> chain_values_from_view(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> view)
{
    const auto mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    std::vector<double> values(static_cast<std::size_t>(view.extent(0))
        * view.extent(1));
    for (std::size_t row = 0; row < mirror.extent(0); ++row)
        for (std::size_t column = 0; column < mirror.extent(1); ++column)
            values[row * mirror.extent(1) + column] =
                static_cast<double>(mirror(row, column));
    return values;
}

// target = values, narrowed to the target's precision.  The chain stages
// straight into the destination's own mirror, so a host->device boundary is
// one copy into the buffer the consumer reads rather than a temporary view
// plus a second device-to-device copy of it.  On a host build the mirror
// aliases the target, so the write lands where it will be read and the copy
// is skipped.
template<typename Precision>
void chain_store_values(
    Kokkos::View<Precision**, Kokkos::LayoutRight> target,
    const std::vector<double>& values, const char* label)
{
    const std::size_t rows = target.extent(0);
    const std::size_t columns = target.extent(1);
    if (values.size() != rows * columns)
        throw std::invalid_argument(std::string("the polar chain's ") + label
            + " buffer has an inconsistent size.");
    auto mirror = Kokkos::create_mirror_view(target);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * columns;
        for (std::size_t column = 0; column < columns; ++column)
            mirror(row, column) =
                static_cast<Precision>(values[base + column]);
    }
    if (mirror.data() == target.data())
        return;
    Kokkos::deep_copy(target, mirror);
}

// The long-range tables name their memory space explicitly, which is a
// distinct View spelling from the evaluator-default views on accelerator
// builds even when both resolve to device memory. Keep their public
// double-vector conversion equally explicit.
template<typename Precision>
void chain_store_long_range_values(
    execution::long_range::LongRangeKernelMatrix<Precision> target,
    const std::vector<double>& values, const char* label)
{
    const std::size_t rows = target.extent(0);
    const std::size_t columns = target.extent(1);
    if (values.size() != rows * columns)
        throw std::invalid_argument(std::string("the polar chain's ") + label
            + " buffer has an inconsistent size.");
    auto mirror = Kokkos::create_mirror_view(target);
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t base = row * columns;
        for (std::size_t column = 0; column < columns; ++column)
            mirror(row, column) =
                static_cast<Precision>(values[base + column]);
    }
    if (mirror.data() != target.data())
        Kokkos::deep_copy(target, mirror);
}

template<typename Precision>
std::vector<double> chain_long_range_values(
    execution::long_range::LongRangeKernelMatrix<Precision> view)
{
    const auto mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    std::vector<double> values(static_cast<std::size_t>(view.extent(0))
        * view.extent(1));
    for (std::size_t row = 0; row < mirror.extent(0); ++row)
        for (std::size_t column = 0; column < mirror.extent(1); ++column)
            values[row * mirror.extent(1) + column] =
                static_cast<double>(mirror(row, column));
    return values;
}

} // namespace

// ---------------------------------------------------------------------------
// The field update
// ---------------------------------------------------------------------------

template<typename Precision>
PolarFieldUpdateKokkosT<Precision>::PolarFieldUpdateKokkosT(
    const nlohmann::json& definition)
    :embedding_(definition.at("potential_embedding")),
     source_embedding_(polar_specialized_linear(
         definition.at("source_embedding"),"source_embedding")),
     dot_products_(definition.at("dot_products")),
     nonlinearity_(definition.at("nonlinearity")),
     tp_out_(definition.at("tp_out")),
     readout_(definition.at("readout"))
{
    if (embedding_.output_dimension() != dot_products_.input_2_dimension()
        || dot_products_.input_1_dimension() != tp_out_.input_1_dimension()
        || dot_products_.output_dimension()
            != source_embedding_.output_dimension()
        || dot_products_.output_dimension() != tp_out_.input_2_dimension()
        || tp_out_.output_dimension() != dot_products_.input_1_dimension()
        || readout_.input_dimension() != tp_out_.output_dimension())
        throw std::invalid_argument(
            "the polar field update's block dimensions are inconsistent.");
    if (nonlinearity_.input_size() != 2 * dot_products_.output_dimension()
        || nonlinearity_.output_size() != dot_products_.output_dimension())
        throw std::invalid_argument(
            "the polar field update's nonlinearity must map the concatenated "
            "invariant pair back to the invariant width.");
    // The readout emits every charge channel followed by the Fukui source
    // pair, so its output width pins the block's own charge width.
    if (readout_.output_dimension() != embedding_.charge_input_dimension() + 2)
        throw std::invalid_argument(
            "the polar field update's readout must emit every charge channel "
            "followed by the Fukui source pair.");
}

template<typename Precision>
void PolarFieldUpdateKokkosT<Precision>::prepare(const int nodes)
{
    if (static_cast<int>(mixed_.extent(0)) == nodes && mixed_.extent(1) != 0)
        return;
    const int node_width = node_feature_dimension();
    const int invariants = dot_products_.output_dimension();
    const int source_width = source_embedding_.output_dimension();
    const int output = readout_.output_dimension();
    mixed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update mixed", nodes, node_width);
    mixed_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update mixed adjoint", nodes, node_width);
    invariants_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update invariants", nodes, invariants);
    invariants_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update invariants adjoint", nodes, invariants);
    embedded_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update source embedding", nodes, source_width);
    embedded_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update source embedding seed", nodes, source_width);
    concatenated_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update concatenated", nodes, invariants + source_width);
    mlp_input_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update mlp input", nodes, invariants + source_width);
    mlp_input_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update mlp input adjoint", nodes, invariants + source_width);
    mlp_output_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update mlp output", nodes, invariants);
    mlp_output_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update mlp output adjoint", nodes, invariants);
    new_features_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update features", nodes, node_width);
    new_features_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update features adjoint", nodes, node_width);
    readout_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar update readout adjoint", nodes, output);
    dot_products_.prepare(nodes);
    tp_out_.prepare(nodes);
    readout_.prepare(nodes);
    embedding_.prepare(nodes);
}

template<typename Precision>
void PolarFieldUpdateKokkosT<Precision>::evaluate(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> node_features,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> potential_features,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> node_attrs,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> local_charges,
    Kokkos::View<Precision**, Kokkos::LayoutRight> output)
{
    require_rows<Precision>(output, output_dimension(), "field update output");
    embedding_.evaluate(
        potential_features, node_features, local_charges, mixed_);
    dot_products_.evaluate(node_features, mixed_, invariants_);
    source_embedding_.evaluate(node_attrs, embedded_);
    // The oracle concatenates the invariants with the embedded source before
    // the MLP; the MLP kernel reads one contiguous batch, so the pair is
    // staged into one buffer.
    auto concatenated = concatenated_;
    auto invariants = invariants_;
    auto embedded = embedded_;
    const int invariant_width = dot_products_.output_dimension();
    Kokkos::parallel_for(
        "polar update concatenate", concatenated.size(),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t row = flat / concatenated.extent(1);
            const std::size_t column = flat % concatenated.extent(1);
            concatenated(row, column) =
                column < static_cast<std::size_t>(invariant_width)
                ? invariants(row, column)
                : embedded(row, column - invariant_width);
        });
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, mlp_input_, concatenated_);
    nonlinearity_.evaluate(mlp_input_, mlp_output_);
    tp_out_.evaluate(node_features, mlp_output_, new_features_);
    readout_.evaluate(new_features_, output);
}

template<typename Precision>
void PolarFieldUpdateKokkosT<Precision>::reverse(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> node_features,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> potential_features,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> node_attrs,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> local_charges,
    Kokkos::View<Precision**, Kokkos::LayoutRight> node_features_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> potential_features_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> node_attrs_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> local_charges_adjoint,
    const bool initialize_adjoints)
{
    (void)potential_features;
    (void)node_attrs;
    (void)local_charges;
    if (initialize_adjoints) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            node_features_adjoint, Precision(0));
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            potential_features_adjoint, Precision(0));
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            node_attrs_adjoint, Precision(0));
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            local_charges_adjoint, Precision(0));
    }
    // The block's own scratch is cleared here, so every stage below
    // accumulates and the caller's four views are the only accumulators it
    // has to reason about.
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        mixed_adjoint_, Precision(0));
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        mlp_output_adjoint_, Precision(0));
    // readout -> tp_out -> MLP -> source embedding -> the potential embedding,
    // the two node-feature seeds accumulating into one buffer.
    readout_.reverse(output_adjoint, new_features_, new_features_adjoint_, true);
    tp_out_.reverse(new_features_adjoint_, node_features, mlp_output_,
        node_features_adjoint, mlp_output_adjoint_, false);
    nonlinearity_.reverse(mlp_input_, mlp_output_adjoint_, mlp_input_adjoint_);
    const int invariant_width = dot_products_.output_dimension();
    auto input_adjoint = mlp_input_adjoint_;
    auto invariant_seed = invariants_adjoint_;
    auto embedded_seed = embedded_seed_;
    Kokkos::parallel_for(
        "polar update split adjoint", input_adjoint.size(),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t row = flat / input_adjoint.extent(1);
            const std::size_t column = flat % input_adjoint.extent(1);
            if (column < static_cast<std::size_t>(invariant_width))
                invariant_seed(row, column) = input_adjoint(row, column);
            else
                embedded_seed(row, column - invariant_width)
                    = input_adjoint(row, column);
        });
    dot_products_.reverse(invariant_seed, node_features, mixed_,
        node_features_adjoint, mixed_adjoint_, false);
    embedding_.reverse(mixed_adjoint_, potential_features_adjoint,
        node_features_adjoint, local_charges_adjoint, false);
    source_embedding_.reverse(embedded_seed, node_attrs_adjoint, false);
}

// ---------------------------------------------------------------------------
// The local-electron readout
// ---------------------------------------------------------------------------

template<typename Precision>
PolarLocalElectronReadoutKokkosT<Precision>::PolarLocalElectronReadoutKokkosT(
    const nlohmann::json& definition)
    :linear_up_q_(polar_specialized_linear(
         definition.at("linear_up_q"),"local_charge_embedding")),
     linear_up_v_(polar_specialized_linear(
         definition.at("linear_up_v"),"local_potential_embedding")),
     dot_products_q_(definition.at("dot_products_q")),
     dot_products_v_(definition.at("dot_products_v")),
     mlp_(definition.at("mlp"))
{
    if (linear_up_q_.output_dimension() != dot_products_q_.input_2_dimension()
        || linear_up_v_.output_dimension() != dot_products_v_.input_2_dimension()
        || dot_products_q_.input_1_dimension()
            != dot_products_v_.input_1_dimension()
        || dot_products_q_.output_dimension()
            != dot_products_v_.output_dimension())
        throw std::invalid_argument(
            "the polar local-electron readout's block dimensions are "
            "inconsistent.");
    if (mlp_.input_size() != 2 * dot_products_q_.output_dimension()
        || mlp_.output_size() != 1)
        throw std::invalid_argument(
            "the polar local-electron readout's MLP must reduce the "
            "concatenated invariants to one energy per node.");
}

template<typename Precision>
void PolarLocalElectronReadoutKokkosT<Precision>::prepare(const int nodes)
{
    if (static_cast<int>(charges_.extent(0)) == nodes && charges_.extent(1) != 0)
        return;
    const int charges = linear_up_q_.input_dimension();
    const int field = linear_up_v_.input_dimension();
    const int node_width = linear_up_q_.output_dimension();
    const int invariants = dot_products_q_.output_dimension();
    charges_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local charges", nodes, charges);
    charges_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local charge seed", nodes, charges);
    charge_embedding_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local charge embedding", nodes, node_width);
    charge_embedding_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local charge embedding adjoint", nodes, node_width);
    field_embedding_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local field embedding", nodes, node_width);
    field_embedding_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local field embedding adjoint", nodes, node_width);
    invariant_q_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local invariant q", nodes, invariants);
    invariant_q_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local invariant q adjoint", nodes, invariants);
    invariant_v_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local invariant v", nodes, invariants);
    invariant_v_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local invariant v adjoint", nodes, invariants);
    invariants_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local invariants", nodes, 2 * invariants);
    invariants_adjoint_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local invariants adjoint", nodes, 2 * invariants);
    energy_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar local energy", nodes, 1);
    dot_products_q_.prepare(nodes);
    dot_products_v_.prepare(nodes);
}

template<typename Precision>
void PolarLocalElectronReadoutKokkosT<Precision>::evaluate(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> node_features,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> field_features,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> charges_0,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> charges_induced,
    Kokkos::View<Precision**, Kokkos::LayoutRight> output)
{
    require_rows<Precision>(output, 1, "local-electron energy");
    auto charges = charges_;
    auto induced = charges_induced;
    auto initial = charges_0;
    Kokkos::parallel_for(
        "polar local charge sum", charges.size(),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t row = flat / charges.extent(1);
            const std::size_t column = flat % charges.extent(1);
            charges(row, column) = induced(row, column) + initial(row, column);
        });
    linear_up_q_.evaluate(charges_, charge_embedding_);
    linear_up_v_.evaluate(field_features, field_embedding_);
    dot_products_q_.evaluate(node_features, charge_embedding_, invariant_q_);
    dot_products_v_.evaluate(node_features, field_embedding_, invariant_v_);
    auto concatenated = invariants_;
    auto invariant_q = invariant_q_;
    auto invariant_v = invariant_v_;
    const int invariant_width = dot_products_q_.output_dimension();
    Kokkos::parallel_for(
        "polar local concatenate", concatenated.size(),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t row = flat / concatenated.extent(1);
            const std::size_t column = flat % concatenated.extent(1);
            concatenated(row, column) =
                column < static_cast<std::size_t>(invariant_width)
                ? invariant_q(row, column)
                : invariant_v(row, column - invariant_width);
        });
    mlp_.evaluate(invariants_, energy_);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, output, energy_);
}

template<typename Precision>
void PolarLocalElectronReadoutKokkosT<Precision>::reverse(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> node_features,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> field_features,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> charges_0,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> charges_induced,
    Kokkos::View<Precision**, Kokkos::LayoutRight> node_features_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> field_features_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> charges_0_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> charges_induced_adjoint,
    const bool initialize_adjoints)
{
    (void)field_features;
    (void)charges_0;
    (void)charges_induced;
    if (initialize_adjoints) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            node_features_adjoint, Precision(0));
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            field_features_adjoint, Precision(0));
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            charges_0_adjoint, Precision(0));
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            charges_induced_adjoint, Precision(0));
    }
    mlp_.reverse(invariants_, output_adjoint, invariants_adjoint_);
    const int invariant_width = dot_products_q_.output_dimension();
    auto invariants_seed = invariants_adjoint_;
    auto q_seed = invariant_q_adjoint_;
    auto v_seed = invariant_v_adjoint_;
    Kokkos::parallel_for(
        "polar local split adjoint", invariants_seed.size(),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t row = flat / invariants_seed.extent(1);
            const std::size_t column = flat % invariants_seed.extent(1);
            if (column < static_cast<std::size_t>(invariant_width))
                q_seed(row, column) = invariants_seed(row, column);
            else
                v_seed(row, column - invariant_width)
                    = invariants_seed(row, column);
        });
    // Both contractions address the same node-feature input, so the first
    // initializes the seed and the second accumulates into it.  The two
    // embedding adjoints are this block's own accumulators, so they are
    // cleared here rather than carried over from an earlier reverse call (the
    // uvu's reverse adds into its input_2 adjoint).
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        charge_embedding_adjoint_, Precision(0));
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        field_embedding_adjoint_, Precision(0));
    dot_products_q_.reverse(q_seed, node_features, charge_embedding_,
        node_features_adjoint, charge_embedding_adjoint_, false);
    dot_products_v_.reverse(v_seed, node_features, field_embedding_,
        node_features_adjoint, field_embedding_adjoint_, false);
    linear_up_q_.reverse(charge_embedding_adjoint_, charges_seed_, true);
    linear_up_v_.reverse(field_embedding_adjoint_, field_features_adjoint, false);
    auto charge_seed = charges_seed_;
    auto induced_seed = charges_induced_adjoint;
    auto initial_seed = charges_0_adjoint;
    Kokkos::parallel_for(
        "polar local charge seed", charge_seed.size(),
        KOKKOS_LAMBDA(const std::size_t flat) {
            const std::size_t row = flat / charge_seed.extent(1);
            const std::size_t column = flat % charge_seed.extent(1);
            induced_seed(row, column) += charge_seed(row, column);
            initial_seed(row, column) += charge_seed(row, column);
        });
}

// ---------------------------------------------------------------------------
// The unrolled chain
// ---------------------------------------------------------------------------

template<typename Precision>
PolarDeltaChainKokkosT<Precision>::PolarDeltaChainKokkosT(
    const nlohmann::json& data, const Contract& contract)
    :contract_(contract),
     ops_(contract.long_range.kspace_cutoff,
         contract.long_range.density_max_l,
         contract.long_range.density_smearing_width,
         contract.long_range.feature_max_l,
         contract.long_range.feature_smearing_widths)
{
    const std::string reason =
        execution::long_range::validate_long_range_contract(contract.long_range);
    if (!reason.empty())
        throw std::invalid_argument(
            "the polar delta chain requires valid long-range contract data: "
            + reason);
    if (contract.interaction_count < 1 || contract.node_feature_dimension < 1)
        throw std::invalid_argument(
            "the polar delta chain requires the backbone's interaction and "
            "feature widths.");
    charges_dimension_ = static_cast<int>(contract.long_range.charges_dimension);
    density_width_ = (contract.long_range.density_max_l + 1)
        * (contract.long_range.density_max_l + 1);
    field_dimension_ = static_cast<int>(contract.long_range.field_dimension);
    potential_dimension_ = static_cast<int>(contract.long_range.potential_dimension);
    if (charges_dimension_ != 2 * density_width_)
        throw std::invalid_argument(
            "the polar delta chain requires two spin channels of density.");
    if (potential_dimension_ != 2 * field_dimension_)
        throw std::invalid_argument(
            "the polar delta chain requires two spin channels of potential "
            "features.");
    if (!data.contains("polar"))
        throw std::invalid_argument(
            "the polar delta chain requires the model's polar block.");
    const nlohmann::json& polar = data.at("polar");
    // The two chain ends are model blocks the parse requires, so a file that
    // lost one is refused with its own key named rather than by whatever
    // nlohmann's lookup reports.
    for (const char* key : {"spin_source_heads", "feature_mixer",
            "field_update_blocks", "fukui_source_map", "local_electron_energy"})
        if (!polar.contains(key) || polar.at(key).is_null())
            throw std::invalid_argument(
                std::string("the polar delta chain requires the model's "
                    "'polar.") + key + "' block.");
    for (const auto& head : polar.at("spin_source_heads")) {
        E3LinearKokkosT<Precision> linear(
            polar_specialized_linear(head.at("linear"),"spin_source"));
        if (linear.output_dimension() != charges_dimension_
            || linear.input_dimension() != contract.node_feature_dimension)
            throw std::invalid_argument(
                "a polar spin-source head does not match the backbone's "
                "feature width and the contract's charge width.");
        spin_source_heads_.push_back(std::move(linear));
        spin_source_zero_charges_.push_back(head.at("zero_charges").get<bool>());
    }
    if (static_cast<int>(spin_source_heads_.size()) != contract.interaction_count)
        throw std::invalid_argument(
            "the polar spin-source heads must match the interaction count.");
    for (const auto& linear : polar.at("feature_mixer").at("linears")) {
        E3LinearKokkosT<Precision> mixer(
            polar_specialized_linear(linear,"feature_mixer"));
        if (mixer.input_dimension() != contract.node_feature_dimension
            || mixer.output_dimension() != contract.node_feature_dimension)
            throw std::invalid_argument(
                "a polar feature-mixer linear does not match the backbone's "
                "feature width.");
        feature_mixer_.push_back(std::move(mixer));
    }
    if (static_cast<int>(feature_mixer_.size()) != contract.interaction_count)
        throw std::invalid_argument(
            "the polar feature mixer must match the interaction count.");
    fukui_source_map_ =
        PolarNonLinearBiasReadoutKokkosT<Precision>(polar.at("fukui_source_map"));
    if (fukui_source_map_.input_dimension() != contract.node_feature_dimension
        || fukui_source_map_.output_dimension() != 2)
        throw std::invalid_argument(
            "the polar Fukui source map must consume the backbone's features "
            "and emit the source pair.");
    // One update block per recursion step: the shipped checkpoints carry
    // their own weights per step, so the chain drives them by step index.
    const auto& update_blocks = polar.at("field_update_blocks");
    if (static_cast<int>(update_blocks.size()) != contract.long_range.recursion_step_count)
        throw std::invalid_argument(
            "the polar field update count must match the recursion depth.");
    for (const auto& block : update_blocks) {
        PolarFieldUpdateKokkosT<Precision> update(block);
        if (update.node_feature_dimension() != contract.node_feature_dimension
            || update.output_dimension() != charges_dimension_ + 2)
            throw std::invalid_argument(
                "a polar field update does not match the contract's widths.");
        field_updates_.push_back(std::move(update));
    }
    local_electron_readout_ = PolarLocalElectronReadoutKokkosT<Precision>(
        polar.at("local_electron_energy"));
    if (local_electron_readout_.node_feature_dimension()
            != contract.node_feature_dimension
        // The readout's field input is the two spin channels' potential
        // features, and its charge input the two channels' multipoles.
        || local_electron_readout_.field_dimension() != potential_dimension_
        || local_electron_readout_.charge_dimension() != charges_dimension_)
        throw std::invalid_argument(
            "the polar local-electron readout does not match the contract's "
            "widths.");
    const std::vector<double> norms =
        polar.at("field_feature_norms").at("values").get<std::vector<double>>();
    if (static_cast<int>(norms.size()) != field_dimension_)
        throw std::invalid_argument(
            "the polar field-feature norms must carry one entry per feature.");
    norms_ = execution::long_range::LongRangeVector<double>(
        "polar field feature norms", norms.size());
    for (std::size_t index = 0; index < norms.size(); ++index)
        norms_(index) = norms[index];
    field_norms_device_ = Kokkos::View<Precision*, Kokkos::LayoutRight>(
        "polar field feature norms device", norms.size());
    auto norms_mirror = Kokkos::create_mirror_view(field_norms_device_);
    for (std::size_t index = 0; index < norms.size(); ++index)
        norms_mirror(index) = static_cast<Precision>(norms[index]);
    Kokkos::deep_copy(field_norms_device_, norms_mirror);
    si_feature_constants_device_ =
        Kokkos::View<Precision*, Kokkos::LayoutRight>(
            "polar feature self constants", ops_.si_feature_constants_.extent(0));
    si_feature_select_device_ =
        Kokkos::View<std::size_t*, Kokkos::LayoutRight>(
            "polar feature self selection", ops_.si_feature_select_.size());
    auto constants_mirror =
        Kokkos::create_mirror_view(si_feature_constants_device_);
    auto selection_mirror = Kokkos::create_mirror_view(si_feature_select_device_);
    for (std::size_t index = 0; index < ops_.si_feature_select_.size(); ++index) {
        constants_mirror(index) = static_cast<Precision>(
            ops_.si_feature_constants_(index));
        selection_mirror(index) = ops_.si_feature_select_[index];
    }
    Kokkos::deep_copy(si_feature_constants_device_, constants_mirror);
    Kokkos::deep_copy(si_feature_select_device_, selection_mirror);
    ready_ = true;
}

template<typename Precision>
std::vector<const E3LinearKokkosT<Precision>*>
PolarDeltaChainKokkosT<Precision>::e3linear_modules() const
{
    std::vector<const E3LinearKokkosT<Precision>*> linears;
    for(const auto& linear:spin_source_heads_) linears.push_back(&linear);
    for(const auto& linear:feature_mixer_) linears.push_back(&linear);
    {
        const auto nested=fukui_source_map_.e3linears();
        linears.insert(linears.end(),nested.begin(),nested.end());
    }
    for(const auto& update:field_updates_) {
        const auto nested=update.e3linears();
        linears.insert(linears.end(),nested.begin(),nested.end());
    }
    {
        const auto nested=local_electron_readout_.e3linears();
        linears.insert(linears.end(),nested.begin(),nested.end());
    }
    return linears;
}

template<typename Precision>
typename PolarDeltaChainKokkosT<Precision>::E3LinearDiagnostics
PolarDeltaChainKokkosT<Precision>::e3linear_diagnostics() const
{
    E3LinearDiagnostics result;
    const auto linears=e3linear_modules();
    result.linear_count=linears.size();
    for(const auto* linear:linears) {
        if(linear->specialization_family()=="polar_e3linear")
            ++result.selected_count;
        result.forward_count+=linear->specialized_forward_count();
        result.reverse_count+=linear->specialized_reverse_count();
        result.fallback_count+=linear->specialization_fallback_count();
        const std::string role=linear->specialization_role();
        const auto existing=std::find_if(result.roles.begin(),result.roles.end(),
            [&](const auto& item) { return item.first==role; });
        if(existing==result.roles.end()) result.roles.emplace_back(role,1);
        else ++existing->second;
    }
    return result;
}

template<typename Precision>
typename PolarDeltaChainKokkosT<Precision>::PotentialAdjointExtents
PolarDeltaChainKokkosT<Precision>::validate_potential_adjoint_extents(
    const std::size_t nodes, const std::size_t graphs)
{
    constexpr auto model_extent_max =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (nodes > model_extent_max)
        throw std::invalid_argument(
            "the polar chain node count exceeds INT_MAX before preparation.");
    if (graphs > model_extent_max)
        throw std::invalid_argument(
            "the polar chain graph count exceeds INT_MAX before preparation.");

    const auto checked_product = [] (const std::uint64_t extent,
                                     const std::uint64_t width,
                                     const char* what) {
        if (extent > std::numeric_limits<std::uint64_t>::max() / width
            || extent > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()) / width)
            throw std::length_error(std::string(what)
                + " staging extent overflows checked 64-bit allocation size.");
        return extent * width;
    };
    const auto node_extent = static_cast<std::uint64_t>(nodes);
    const auto graph_extent = static_cast<std::uint64_t>(graphs);
    return PotentialAdjointExtents{
        node_extent,
        graph_extent,
        checked_product(node_extent, 4, "external-node"),
        checked_product(graph_extent, 2, "external-graph"),
        checked_product(node_extent, 6, "correction-node"),
        checked_product(graph_extent, 6, "correction-graph")};
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::prepare(
    const int nodes, const int graphs)
{
    if (nodes < 0 || graphs < 0)
        throw std::invalid_argument(
            "the polar chain preparation extents must be non-negative.");
    // Validate every new staging product before the first workspace allocation.
    const auto potential_adjoint_extents = validate_potential_adjoint_extents(
        static_cast<std::size_t>(nodes), static_cast<std::size_t>(graphs));
    (void)potential_adjoint_extents;
    if (nodes == nodes_ && graphs == graphs_ && !density_steps_.empty())
        return;
    nodes_ = nodes;
    graphs_ = graphs;
    staged_node_types_.clear();
    graph_pbc_.clear();
    external_potential_.clear();
    total_charge_.clear();
    total_spin_.clear();
    const int layers = contract_.interaction_count;
    const int width = contract_.node_feature_dimension;
    const int steps = this->steps();
    node_features_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain node features", nodes, layers * width);
    node_attrs_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain node attrs", nodes, contract_.num_elements);
    node_types_ = Kokkos::View<int*, Kokkos::LayoutRight>(
        "polar chain node types", nodes);
    density_steps_.assign(static_cast<std::size_t>(steps + 1), {});
    for (int step = 0; step <= steps; ++step)
        density_steps_[static_cast<std::size_t>(step)] =
            Kokkos::View<Precision**, Kokkos::LayoutRight>(
                "polar chain density", nodes, charges_dimension_);
    fukui_raw_steps_.assign(static_cast<std::size_t>(steps + 1), {});
    fukui_steps_.assign(static_cast<std::size_t>(steps + 1), {});
    source_steps_.assign(static_cast<std::size_t>(steps + 1), {});
    for (int step = 0; step <= steps; ++step) {
        fukui_raw_steps_[static_cast<std::size_t>(step)] =
            Kokkos::View<Precision**, Kokkos::LayoutRight>(
                "polar chain raw fukui", nodes, 2);
        fukui_steps_[static_cast<std::size_t>(step)] =
            Kokkos::View<Precision**, Kokkos::LayoutRight>(
                "polar chain normalized fukui", nodes, 2);
        source_steps_[static_cast<std::size_t>(step)] =
            Kokkos::View<Precision**, Kokkos::LayoutRight>(
                "polar chain update sources", nodes, charges_dimension_);
    }
    potential_features_steps_.assign(static_cast<std::size_t>(steps), {});
    density_seed_steps_.assign(static_cast<std::size_t>(steps + 1), {});
    fukui_seed_steps_.assign(static_cast<std::size_t>(steps + 1), {});
    potential_seed_steps_.assign(static_cast<std::size_t>(steps), {});
    for (int step = 0; step <= steps; ++step) {
        density_seed_steps_[static_cast<std::size_t>(step)] =
            Kokkos::View<Precision**, Kokkos::LayoutRight>(
                "polar chain density seed", nodes, charges_dimension_);
        fukui_seed_steps_[static_cast<std::size_t>(step)] =
            Kokkos::View<Precision**, Kokkos::LayoutRight>(
                "polar chain fukui seed step", nodes, 2);
    }
    for (int step = 0; step < steps; ++step)
    {
        potential_features_steps_[static_cast<std::size_t>(step)] =
            Kokkos::View<Precision**, Kokkos::LayoutRight>(
                "polar chain potential features", nodes, potential_dimension_);
        potential_seed_steps_[static_cast<std::size_t>(step)] =
            Kokkos::View<Precision**, Kokkos::LayoutRight>(
                "polar chain potential seed step", nodes, potential_dimension_);
    }
    features_mixed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain mixed features", nodes, width);
    head_row_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain head row", nodes, charges_dimension_);
    mixer_row_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain mixer row", nodes, width);
    fukui_sources_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain fukui sources", nodes, 2);
    fukui_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain fukui seed", nodes, 2);
    fukui_input_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain fukui input", nodes, width);
    update_output_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain update output", nodes, charges_dimension_ + 2);
    local_energy_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain local energy", nodes, 1);
    node_feature_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain node seed", nodes, layers * width);
    heads_density_device_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain heads density", nodes, charges_dimension_);
    composed_density_device_ =
        Kokkos::View<Precision**, Kokkos::LayoutRight>(
            "polar composed charge density", nodes, density_width_);
    composed_graph_multipoles_device_ =
        Kokkos::View<double**, Kokkos::LayoutRight>(
            "polar composed graph multipoles", graphs, 5);
    // [total, local, reciprocal, correction, external, dipole xyz]
    composed_graph_outputs_device_ =
        Kokkos::View<double**, Kokkos::LayoutRight>(
            "polar composed graph outputs", graphs, 8);
    composed_graph_seeds_device_ =
        Kokkos::View<double**, Kokkos::LayoutRight>(
            "polar composed graph seeds", graphs, 5);
    composed_stress_device_ =
        Kokkos::View<double**, Kokkos::LayoutRight>(
            "polar composed stress", graphs, 9);
    realspace_position_seed_device_ =
        Kokkos::View<double**, Kokkos::LayoutRight>(
            "polar realspace position seed", nodes, 3);
    public_position_seed_device_ =
        Kokkos::View<double**, Kokkos::LayoutRight>(
            "polar public position seed", nodes, 3);
    realspace_barycenter_device_ =
        Kokkos::View<double**, Kokkos::LayoutRight>(
            "polar realspace barycenter", graphs, 3);
    reporting_volumes_device_ = Kokkos::View<double*, Kokkos::LayoutRight>(
        "polar reporting volumes", graphs);
    composed_energy_strain_device_ =
        Kokkos::View<double***, Kokkos::LayoutRight>(
            "polar composed energy strain", graphs, 3, 3);
    si_energy_constants_device_ =
        Kokkos::View<Precision*, Kokkos::LayoutRight>(
            "polar composed self constants",
            ops_.si_energy_constants_.extent(0));
    si_energy_select_device_ =
        Kokkos::View<std::size_t*, Kokkos::LayoutRight>(
            "polar composed self select", ops_.si_energy_select_.size());
    realspace_graph_energy_seed_device_ =
        Kokkos::View<Precision*, Kokkos::LayoutRight>(
            "polar realspace graph energy seed", graphs);
#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
    pme_forward_features_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar PME forward features", nodes, potential_dimension_);
    pme_feature_seeds_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar PME feature seeds", nodes, potential_dimension_);
    pme_source_adjoints_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar PME source adjoints", nodes, charges_dimension_);
    pme_energy_features_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar PME energy features", nodes, field_dimension_);
    pme_energy_feature_seeds_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar PME energy feature seeds", nodes, field_dimension_);
    pme_energy_source_adjoints_ =
        Kokkos::View<Precision**, Kokkos::LayoutRight>(
            "polar PME energy source adjoints", nodes, density_width_);
    pme_position_adjoints_ = Kokkos::View<double**, Kokkos::LayoutRight>(
        "polar PME position adjoints", nodes, 3);
    pme_channel_energies_ = Kokkos::View<double*, Kokkos::LayoutRight>(
        "polar PME channel energies", 2);
    pme_energy_seeds_ = Kokkos::View<double*, Kokkos::LayoutRight>(
        "polar PME energy seeds", 2);
    pme_energy_channel_energy_ = Kokkos::View<double*, Kokkos::LayoutRight>(
        "polar PME composed energy", 1);
    pme_energy_seed_ = Kokkos::View<double*, Kokkos::LayoutRight>(
        "polar PME composed energy seed", 1);
    pme_cell_adjoints_ = Kokkos::View<double*, Kokkos::LayoutRight>(
        "polar PME cell adjoints", 9);
#endif
    {
        auto constants = Kokkos::create_mirror_view(si_energy_constants_device_);
        auto select = Kokkos::create_mirror_view(si_energy_select_device_);
        for (std::size_t index = 0;
            index < ops_.si_energy_constants_.extent(0); ++index) {
            constants(index) = static_cast<Precision>(
                ops_.si_energy_constants_(index));
            select(index) = ops_.si_energy_select_[index];
        }
        Kokkos::deep_copy(si_energy_constants_device_, constants);
        Kokkos::deep_copy(si_energy_select_device_, select);
    }
    graph_pair_sums_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain graph pair sums", graphs, 2);
    channel_density_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain channel density", nodes, density_width_);
    graph_multipoles_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain graph multipoles", graphs * 2, 5);
    barycenter_device_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain barycenter", graphs, 3);
    graph_node_counts_device_ = Kokkos::View<int*>(
        "polar chain graph node counts", graphs);
    Kokkos::deep_copy(graph_node_counts_device_, 0);
    external_potential_device_ =
        Kokkos::View<Precision**, Kokkos::LayoutRight>(
            "polar chain external potential", graphs, 4);
    graph_pbc_device_ = Kokkos::View<int**, Kokkos::LayoutRight>(
        "polar chain graph pbc", graphs, 3);
    total_charge_device_ = Kokkos::View<Precision*, Kokkos::LayoutRight>(
        "polar chain total charge", graphs);
    total_spin_device_ = Kokkos::View<Precision*, Kokkos::LayoutRight>(
        "polar chain total spin", graphs);
    graph_reverse_sums_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain reverse graph sums", graphs, 8);
    position_seed_device_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain position seed", nodes, 3);
    external_seed_device_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain external seed", graphs, 4);
    target_seed_device_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain charge spin seed", graphs, 2);
    correction_strain_device_ =
        Kokkos::View<Precision**, Kokkos::LayoutRight>(
            "polar chain correction strain", graphs, 9);
    channel_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain channel feature seed", nodes, field_dimension_);
    projected_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain projected feature seed", nodes, field_dimension_);
    multipole_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain multipole seed", nodes, density_width_);
    node_field_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain node field seed", nodes, 4);
    external_node_adjoint_ =
        Kokkos::View<Precision**, Kokkos::LayoutRight>(
            "polar external node adjoint", nodes, 4);
    external_graph_adjoint_ =
        Kokkos::View<Precision**, Kokkos::LayoutRight>(
            "polar external graph adjoint", graphs, 2);
    correction_node_adjoint_ =
        Kokkos::View<Precision**, Kokkos::LayoutRight>(
            "polar correction node adjoint", nodes, 6);
    correction_graph_adjoint_ =
        Kokkos::View<Precision**, Kokkos::LayoutRight>(
            "polar correction graph adjoint", graphs, 6);
    for (auto& update : field_updates_)
        update.prepare(nodes);
    local_electron_readout_.prepare(nodes);
    fukui_source_map_.prepare(nodes);
    density_history_.assign(static_cast<std::size_t>(steps + 1), {});
    fukui_history_.assign(static_cast<std::size_t>(steps + 1), {});
    potential_features_history_.assign(static_cast<std::size_t>(steps), {});
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::validate_inputs(
    const Inputs& inputs, const int node_count, const int graph_count) const
{
    if (!ready_)
        throw std::invalid_argument("the polar delta chain is not parsed.");
    const std::size_t nodes = static_cast<std::size_t>(node_count);
    const std::size_t graphs = static_cast<std::size_t>(graph_count);
    const std::size_t width = static_cast<std::size_t>(
        contract_.node_feature_dimension * contract_.interaction_count);
    const bool host_features = !inputs.node_features.empty();
    const bool device_features = !inputs.node_feature_views.empty();
    if (host_features == device_features)
        throw std::invalid_argument(
            "the polar chain requires exactly one host or device node-feature "
            "input.");
    if (host_features && inputs.node_features.size() != nodes * width)
        throw std::invalid_argument(
            "the polar chain's node features must carry every layer's width.");
    if (device_features) {
        if (inputs.node_feature_views.size()
            != static_cast<std::size_t>(contract_.interaction_count))
            throw std::invalid_argument(
                "the polar chain's device node features must carry one view "
                "per interaction layer.");
        for (const auto& layer : inputs.node_feature_views)
            if (layer.extent(0) != nodes
                || layer.extent(1)
                    != static_cast<std::size_t>(
                        contract_.node_feature_dimension))
                throw std::invalid_argument(
                    "a polar chain device node-feature view has the wrong "
                    "node count or width.");
    }
    const bool host_attrs = !inputs.node_attrs.empty();
    const bool typed_attrs = !inputs.node_types.empty();
    if (host_attrs == typed_attrs)
        throw std::invalid_argument(
            "the polar chain requires exactly one dense node-attribute or "
            "compact node-type input.");
    if (host_attrs && inputs.node_attrs.size()
        != nodes * static_cast<std::size_t>(contract_.num_elements))
        throw std::invalid_argument(
            "the polar chain's node attributes must carry the model's element "
            "count.");
    if (typed_attrs) {
        if (inputs.node_types.size() != nodes)
            throw std::invalid_argument(
                "the polar chain's node types must carry one entry per node.");
        for (const int type : inputs.node_types)
            if (type < 0 || type >= contract_.num_elements)
                throw std::invalid_argument(
                    "the polar chain's node type is outside the model's "
                    "element range.");
    }
    if (inputs.positions.size() != nodes * 3)
        throw std::invalid_argument(
            "the polar chain's positions must be [nodes, 3].");
    if (inputs.cell.size() != graphs * 9)
        throw std::invalid_argument(
            "the polar chain's cell must be [graphs, 3, 3].");
    if (inputs.graph_pbc.size() != graphs * 3)
        throw std::invalid_argument(
            "the polar chain's graph pbc rows must be [graphs, 3].");
    if (inputs.node_graph_index.size() != nodes)
        throw std::invalid_argument(
            "the polar chain's node graph index must carry one entry per node.");
    for (const int graph : inputs.node_graph_index)
        if (graph < 0 || graph >= static_cast<int>(graphs))
            throw std::invalid_argument(
                "the polar chain's node graph index is outside its graph range.");
    if (inputs.total_charge.size() != graphs
        || inputs.total_spin.size() != graphs)
        throw std::invalid_argument(
            "the polar chain's charge and spin channels must carry one entry "
            "per graph.");
    if (graphs != 0 && inputs.external_potential.size() != graphs * 4)
        throw std::invalid_argument(
            "the polar chain's external potential must be [graphs, 4].");
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::build_geometry_cache(
    const Inputs& inputs)
{
    const execution::long_range::LongRangeConstMatrix<double> positions(
        inputs.positions.data(), nodes_, 3);
    const execution::long_range::LongRangeConstTensor3<double> cell(
        inputs.cell.data(), graphs_, 3, 3);
    const execution::long_range::LongRangeConstVector<int> node_graph_index(
        inputs.node_graph_index.data(), nodes_);
    if (solver_decision_.resolved
            == execution::long_range::LongRangeSolver::realspace
        || solver_decision_.resolved
            == execution::long_range::LongRangeSolver::pme) {
        ops_.realspace_geometry_cache_into(
            positions, cell, node_graph_index, cache_);
        grad_cos_workspace_ = {};
        grad_sin_workspace_ = {};
        grad_k_workspace_ = {};
        grad_volume_workspace_ = {};
        prepare_long_range_workspace();
        return;
    }
    ops_.geometry_cache_into(positions, cell, node_graph_index, cache_);
    ops_.prepare_workspace(cache_.n_k, cache_.n_nodes, cache_.n_graphs);
    prepare_long_range_workspace();
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::release_reciprocal_state()
{
    cache_ = {};
    ops_.prepare_workspace(0, 0, 0);
    grad_cos_workspace_ = {};
    grad_sin_workspace_ = {};
    grad_k_workspace_ = {};
    grad_volume_workspace_ = {};
    workspace_nodes_ = 0;
    workspace_graphs_ = 0;
    workspace_n_k_ = 0;
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::resolve_solver(
    const Inputs& inputs, const int nodes, const int graphs)
{
    using execution::long_range::LongRangeSolver;
    using execution::long_range::LongRangeSolverDecision;
    using execution::long_range::LongRangeSolverRequest;
    using execution::long_range::resolve_long_range_solver;

    automatic_nodes_ = 0;
    automatic_reciprocal_modes_ = 0;
    automatic_reciprocal_modes_are_lower_bound_ = false;

    if (graphs != 1) {
        solver_decision_ = LongRangeSolverDecision{};
        solver_decision_.requested = inputs.solver_request;
        solver_decision_.reason = "single_system_required";
        solver_decision_.error =
            "POLAR electrostatics supports exactly one input system; mixed "
            "or multi-system batches are unsupported.";
        release_reciprocal_state();
        throw std::invalid_argument(solver_decision_.error);
    }
    if (graphs < 1
        || inputs.graph_pbc.size() != static_cast<std::size_t>(graphs) * 3) {
        solver_decision_ = LongRangeSolverDecision{};
        solver_decision_.requested = inputs.solver_request;
        solver_decision_.reason = "invalid_pbc_shape";
        solver_decision_.error =
            "POLAR electrostatics routing requires one three-axis PBC row "
            "per input graph.";
        release_reciprocal_state();
        throw std::invalid_argument(solver_decision_.error);
    }
    const std::array<int, 3> pbc{
        inputs.graph_pbc[0], inputs.graph_pbc[1], inputs.graph_pbc[2]};
    execution::long_range::LongRangeSolverCapabilities capabilities;
#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
    capabilities.pme_fft_provider_available = particle_mesh_workspace_ != nullptr;
#if defined(KOKKOS_ENABLE_CUDA)
    capabilities.pme_backend_qualified = particle_mesh_workspace_ != nullptr;
#endif
#endif
    if (inputs.solver_request == LongRangeSolverRequest::automatic
        && pbc == std::array<int, 3>{1, 1, 1}
        && capabilities.pme_fft_provider_available
        && capabilities.pme_backend_qualified
        && inputs.cell.size() == 9) {
        const execution::long_range::LongRangeConstTensor3<double> cell(
            inputs.cell.data(), 1, 3, 3);
        capabilities.automatic_nodes = static_cast<std::size_t>(nodes);
        const std::size_t required_modes =
            execution::long_range::automatic_pme_required_modes(
                capabilities.automatic_nodes);
        capabilities.automatic_reciprocal_modes =
            ops_.reciprocal_mode_count_up_to(cell, required_modes);
        automatic_nodes_ = capabilities.automatic_nodes;
        automatic_reciprocal_modes_ =
            capabilities.automatic_reciprocal_modes;
        automatic_reciprocal_modes_are_lower_bound_ =
            automatic_reciprocal_modes_ == required_modes;
    }
    solver_decision_ = resolve_long_range_solver(
        inputs.solver_request, pbc, contract_.long_range.density_max_l,
        contract_.long_range.feature_max_l, capabilities);
    if (!solver_decision_.accepted) {
        release_reciprocal_state();
        throw std::invalid_argument(solver_decision_.error);
    }
}

template<typename Precision>
std::array<std::int64_t, 3>
PolarDeltaChainKokkosT<Precision>::automatic_pme_mesh(const Inputs& inputs) const
{
    constexpr double target_spacing = 0.45;
    constexpr std::int64_t quantum = 16;
    constexpr std::int64_t minimum_extent = 64;
    std::array<std::int64_t, 3> mesh{};
    for (int axis = 0; axis < 3; ++axis) {
        const std::size_t offset = static_cast<std::size_t>(axis) * 3;
        const double norm = std::sqrt(
            inputs.cell[offset] * inputs.cell[offset]
            + inputs.cell[offset + 1] * inputs.cell[offset + 1]
            + inputs.cell[offset + 2] * inputs.cell[offset + 2]);
        const long double requested = std::round(
            static_cast<long double>(norm)
            / static_cast<long double>(target_spacing));
        if (!std::isfinite(requested)
            || requested > static_cast<long double>(
                std::numeric_limits<std::int64_t>::max() - quantum))
            throw std::length_error(
                "automatic POLAR PME mesh extent overflows int64.");
        const auto points = static_cast<std::int64_t>(requested);
        const auto rounded = ((points + quantum / 2) / quantum) * quantum;
        mesh[axis] = std::max(minimum_extent, rounded);
    }
    return mesh;
}

template<typename Precision>
execution::long_range::LongRangeAllocationUsage
PolarDeltaChainKokkosT<Precision>::reciprocal_cache_allocation_usage() const
{
    if (solver_decision_.resolved
            == execution::long_range::LongRangeSolver::realspace
        || solver_decision_.resolved
            == execution::long_range::LongRangeSolver::pme)
        return {};
    execution::long_range::LongRangeAllocationUsage usage;
    const auto add_view = [&usage] (const auto& view) {
        const std::size_t elements = static_cast<std::size_t>(view.span());
        usage.elements += elements;
        usage.bytes += elements
            * sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    const auto add_vector = [&usage] (const auto& values) {
        const std::size_t elements = values.capacity();
        usage.elements += elements;
        usage.bytes += elements
            * sizeof(typename std::decay_t<decltype(values)>::value_type);
    };
    add_view(cache_.k_vectors);
    add_view(cache_.k_norm2);
    add_view(cache_.k_graph_index);
    add_view(cache_.node_graph_index);
    add_view(cache_.k0_mask);
    add_view(cache_.volume_per_k);
    add_view(cache_.k_factor_coulomb);
    add_view(cache_.k_factor_proj);
    add_view(cache_.cos_table);
    add_view(cache_.sin_table);
    add_view(cache_.feature_fourier_factors);
    add_view(cache_.density_fourier_factors);
    add_view(cache_.kernel_volumes);
    add_view(cache_.positions_workspace);
    add_view(cache_.phase_workspace);
    add_view(cache_.k_vectors_host);
    add_view(cache_.k_norm2_host);
    add_view(cache_.volume_per_k_host);
    add_view(cache_.volumes_host);
    add_view(cache_.k_graph_index_host);
    add_view(cache_.node_graph_index_host);
    add_vector(cache_.positions_host_values);
    add_vector(cache_.cell_host_values);
    add_vector(cache_.node_graph_host_values);
    return usage;
}

template<typename Precision>
execution::long_range::LongRangeAllocationUsage
PolarDeltaChainKokkosT<Precision>::particle_mesh_adapter_allocation_usage() const
{
    execution::long_range::LongRangeAllocationUsage usage;
#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
    const auto add_view = [&usage] (const auto& view) {
        const std::size_t elements = static_cast<std::size_t>(view.span());
        using Value = typename std::decay_t<decltype(view)>::value_type;
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(Value))
            throw std::overflow_error(
                "POLAR PME adapter view byte count overflows size_t.");
        const std::size_t bytes = elements * sizeof(Value);
        if (usage.elements
                > std::numeric_limits<std::size_t>::max() - elements
            || usage.bytes > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error(
                "POLAR PME adapter allocation usage overflows size_t.");
        usage.elements += elements;
        usage.bytes += bytes;
    };
    add_view(pme_forward_features_);
    add_view(pme_feature_seeds_);
    add_view(pme_source_adjoints_);
    add_view(pme_energy_features_);
    add_view(pme_energy_feature_seeds_);
    add_view(pme_energy_source_adjoints_);
    add_view(pme_position_adjoints_);
    add_view(pme_channel_energies_);
    add_view(pme_energy_seeds_);
    add_view(pme_cell_adjoints_);
    add_view(pme_energy_channel_energy_);
    add_view(pme_energy_seed_);
#endif
    return usage;
}

template<typename Precision>
execution::long_range::LongRangeAllocationUsage
PolarDeltaChainKokkosT<Precision>::reciprocal_adjoint_allocation_usage() const
{
    execution::long_range::LongRangeAllocationUsage usage;
    const auto add = [&usage] (const auto& view) {
        const std::size_t elements = static_cast<std::size_t>(view.span());
        usage.elements += elements;
        usage.bytes += elements
            * sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    add(grad_cos_workspace_);
    add(grad_sin_workspace_);
    add(grad_k_workspace_);
    add(grad_volume_workspace_);
    return usage;
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::prepare_long_range_workspace()
{
    const std::size_t nodes = cache_.n_nodes;
    const std::size_t graphs = cache_.n_graphs;
    const std::size_t n_k = cache_.n_k;
    if (workspace_nodes_ == nodes && workspace_graphs_ == graphs
        && workspace_n_k_ == n_k && energy_seed_.data() != nullptr) {
        ++workspace_reuse_count_;
        return;
    }
    workspace_nodes_ = nodes;
    workspace_graphs_ = graphs;
    workspace_n_k_ = n_k;
    ++workspace_replacement_count_;
    const int width = contract_.node_feature_dimension;
    attrs_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain attrs seed", nodes, contract_.num_elements);
    mixed_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain mixed seed", nodes, width);
    row_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain row seed", nodes, charges_dimension_);
    potential_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain potential seed", nodes, potential_dimension_);
    charge_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain charge seed", nodes, charges_dimension_);
    pre_loop_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain pre-loop seed", nodes, charges_dimension_);
    update_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain update seed", nodes, charges_dimension_ + 2);
    local_feature_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain local feature seed", nodes, potential_dimension_);
    energy_seed_ = Kokkos::View<Precision**, Kokkos::LayoutRight>(
        "polar chain energy seed", nodes, 1);
    grad_cos_workspace_ =
        execution::long_range::LongRangeKernelMatrix<Precision>(
            "polar chain grad cos accumulator", n_k, nodes);
    grad_sin_workspace_ =
        execution::long_range::LongRangeKernelMatrix<Precision>(
            "polar chain grad sin accumulator", n_k, nodes);
    grad_k_workspace_ =
        execution::long_range::LongRangeKernelMatrix<Precision>(
            "polar chain grad k accumulator", n_k, 3);
    grad_volume_workspace_ =
        execution::long_range::LongRangeKernelVector<Precision>(
            "polar chain grad volume accumulator", n_k);
}

template<typename Precision>
std::vector<double> PolarDeltaChainKokkosT<Precision>::to_double(
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& values) const
{
    return chain_values_from_view<Precision>(values);
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::normalize_fukui(
    std::vector<double>& sources, const Inputs& inputs) const
{
    // The oracle's own normalization: the per-graph fp64 sum, a zero guard
    // that holds the norm at one, and a broadcast divide.
    std::vector<double> norms(static_cast<std::size_t>(graphs_) * 2, 0.0);
    for (int node = 0; node < nodes_; ++node) {
        const std::size_t graph =
            static_cast<std::size_t>(inputs.node_graph_index[node]);
        for (int channel = 0; channel < 2; ++channel)
            norms[2 * graph + channel] +=
                sources[static_cast<std::size_t>(node) * 2 + channel];
    }
    for (std::size_t index = 0; index < norms.size(); ++index)
        if (norms[index] == 0.0)
            norms[index] = 1.0;
    for (int node = 0; node < nodes_; ++node) {
        const std::size_t graph =
            static_cast<std::size_t>(inputs.node_graph_index[node]);
        for (int channel = 0; channel < 2; ++channel)
            sources[static_cast<std::size_t>(node) * 2 + channel] /=
                norms[2 * graph + channel];
    }
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::normalize_fukui_device(
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& raw,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& normalized)
{
    if (graphs_ > std::numeric_limits<int>::max() / 2)
        throw std::length_error(
            "polar Fukui-normalization team count exceeds INT_MAX.");
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        graph_pair_sums_, Precision(0));
    auto sums = graph_pair_sums_;
    auto node_graph = cache_.node_graph_index;
    const int nodes = nodes_;
    using FukuiSumPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using FukuiSumMember = typename FukuiSumPolicy::member_type;
    Kokkos::parallel_for("polar chain fukui graph sums",
        FukuiSumPolicy(graphs_ * 2, Kokkos::AUTO),
        KOKKOS_LAMBDA(const FukuiSumMember& team) {
            const int index = team.league_rank();
            const int graph = index / 2;
            const int channel = index % 2;
            Precision value = Precision(0);
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, nodes),
                [=](const int node, Precision& sum) {
                    if (node_graph(node) == graph)
                        sum += raw(node, channel);
                }, value);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                sums(graph, channel) = value;
            });
        });
    Kokkos::parallel_for("polar chain normalize fukui",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
        KOKKOS_LAMBDA(const int node) {
            const int graph = node_graph(node);
            for (int channel = 0; channel < 2; ++channel) {
                const Precision norm = sums(graph, channel) == Precision(0)
                    ? Precision(1) : sums(graph, channel);
                normalized(node, channel) = raw(node, channel) / norm;
            }
        });
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::equilibrate_device(
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& sources,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& state)
{
    if (graphs_ > std::numeric_limits<int>::max() / 2)
        throw std::length_error(
            "polar equilibration team count exceeds INT_MAX.");
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        graph_pair_sums_, Precision(0));
    auto sums = graph_pair_sums_;
    auto node_graph = cache_.node_graph_index;
    const int density_width = density_width_;
    const int nodes = nodes_;
    using DensitySumPolicy =
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using DensitySumMember = typename DensitySumPolicy::member_type;
    Kokkos::parallel_for("polar chain l0 graph sums",
        DensitySumPolicy(graphs_ * 2, Kokkos::AUTO),
        KOKKOS_LAMBDA(const DensitySumMember& team) {
            const int index = team.league_rank();
            const int graph = index / 2;
            const int channel = index % 2;
            const int column = channel == 0 ? 0 : density_width;
            Precision value = Precision(0);
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, nodes),
                [=](const int node, Precision& sum) {
                    if (node_graph(node) == graph)
                        sum += state(node, column);
                }, value);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                sums(graph, channel) = value;
            });
        });
    auto total_charge = total_charge_device_;
    auto total_spin = total_spin_device_;
    Kokkos::parallel_for("polar chain equilibrate density",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
        KOKKOS_LAMBDA(const int node) {
            const int graph = node_graph(node);
            const Precision charge = total_charge(graph);
            const Precision spin = total_spin(graph);
            state(node, 0) += sources(node, 0)
                * ((charge + (spin - Precision(1))) / Precision(2)
                    - sums(graph, 0));
            state(node, density_width) += sources(node, 1)
                * ((charge - (spin - Precision(1))) / Precision(2)
                    - sums(graph, 1));
        });
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::potential_features_device(
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& density,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& output)
{
#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
    if (solver_decision_.resolved
        == execution::long_range::LongRangeSolver::pme) {
        if (!particle_mesh_workspace_)
            throw std::logic_error("the admitted POLAR PME route has no workspace.");
        std::array<double, 9> cell{};
        std::copy_n(cell_.data(), cell.size(), cell.begin());
        const std::array<int, 3> pbc{graph_pbc_[0], graph_pbc_[1],
            graph_pbc_[2]};
        try {
        particle_mesh_workspace_->forward_channels_device(
            typename PolarParticleMeshWorkspaceT<Precision>::
                const_device_double_matrix(cache_.realspace_positions),
            typename PolarParticleMeshWorkspaceT<Precision>::const_device_matrix(
                density),
            output, pme_channel_energies_, cell, pbc,
            contract_.long_range.density_smearing_width,
            std::span<const double>(
                contract_.long_range.feature_smearing_widths.data(),
                contract_.long_range.feature_smearing_widths.size()),
            current_inputs_pme_mesh_, current_inputs_pme_spline_order_,
            !contract_.subtract_feature_self_terms,
            contract_.long_range.include_electrostatic_self_interaction, 2);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("POLAR PME potential-feature forward failed: ")
                + error.what());
        }

        auto node_graph = cache_.node_graph_index;
        auto positions = cache_.realspace_positions;
        auto matrix = ops_.external_field_kernel_;
        auto norms = field_norms_device_;
        auto external = external_potential_device_;
        auto barycenter = realspace_barycenter_device_;
        const int field_dimension = field_dimension_;
        Kokkos::parallel_for("polar chain compose PME potential features",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
            KOKKOS_LAMBDA(const int node) {
                const int graph = node_graph(node);
                double external_fields[4] = {
                    static_cast<double>(external(graph, 0)),
                    static_cast<double>(external(graph, 1)),
                    static_cast<double>(external(graph, 2)),
                    static_cast<double>(external(graph, 3))};
                for (int component = 0; component < 3; ++component)
                    external_fields[0] +=
                        (positions(node, component)
                            - barycenter(graph, component))
                        * static_cast<double>(external(graph, 1 + component));
                const double external_permuted[4] = {external_fields[0],
                    external_fields[3], external_fields[1], external_fields[2]};
                for (int channel = 0; channel < 2; ++channel)
                    for (int feature = 0; feature < field_dimension; ++feature) {
                        double external_value = 0.0;
                        for (int component = 0; component < 4; ++component)
                            external_value += static_cast<double>(
                                matrix(feature, component))
                                * external_permuted[component];
                        const int lane = channel * field_dimension + feature;
                        output(node, lane) = static_cast<Precision>(
                            (static_cast<double>(output(node, lane))
                                + 0.5 * external_value)
                            / static_cast<double>(norms(feature)));
                    }
            });
        return;
    }
#endif
    if (solver_decision_.resolved
        == execution::long_range::LongRangeSolver::realspace) {
        auto channel_density = channel_density_;
        auto node_graph = cache_.node_graph_index;
        auto positions = cache_.realspace_positions;
        auto matrix = ops_.external_field_kernel_;
        auto norms = field_norms_device_;
        auto external = external_potential_device_;
        auto barycenter = realspace_barycenter_device_;
        const int nodes = nodes_;
        const int density_width = density_width_;
        const int field_dimension = field_dimension_;
        const bool include_self = !contract_.subtract_feature_self_terms;
        for (int channel = 0; channel < 2; ++channel) {
            const int offset = channel * density_width;
            Kokkos::parallel_for("polar chain realspace density channel",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
                KOKKOS_LAMBDA(const int node) {
                    for (int column = 0; column < density_width; ++column)
                        channel_density(node, column) =
                            density(node, offset + column);
                });
            const auto projected = ops_.realspace_features_kernel(
                cache_, channel_density, include_self,
                /*use_workspace=*/true);
            Kokkos::parallel_for("polar chain compose realspace potential features",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
                KOKKOS_LAMBDA(const int node) {
                    const int graph = node_graph(node);
                    double external_fields[4] = {
                        static_cast<double>(external(graph, 0)),
                        static_cast<double>(external(graph, 1)),
                        static_cast<double>(external(graph, 2)),
                        static_cast<double>(external(graph, 3))};
                    for (int component = 0; component < 3; ++component)
                        external_fields[0] +=
                            (positions(node, component)
                                - static_cast<double>(
                                    barycenter(graph, component)))
                            * static_cast<double>(
                                external(graph, 1 + component));
                    const double external_permuted[4] = {
                        external_fields[0], external_fields[3],
                        external_fields[1], external_fields[2]};
                    for (int feature = 0; feature < field_dimension; ++feature) {
                        double external_value = 0.0;
                        for (int component = 0; component < 4; ++component)
                            external_value += static_cast<double>(
                                matrix(feature, component))
                                * external_permuted[component];
                        output(node, channel * field_dimension + feature) =
                            static_cast<Precision>(
                                (static_cast<double>(projected(node, feature))
                                    + 0.5 * external_value)
                                / static_cast<double>(norms(feature)));
                    }
                });
        }
        return;
    }
    auto aggregates = graph_multipoles_;
    auto positions = cache_.positions_workspace;
    auto node_graph = cache_.node_graph_index;
    const int nodes = nodes_;
    const int graphs = graphs_;
    const int density_width = density_width_;
    constexpr int spin_channel_count = 2;
    constexpr int multipole_quantity_count = 5;
    constexpr int teams_per_graph =
        spin_channel_count * multipole_quantity_count;
    if (graphs > std::numeric_limits<int>::max() / teams_per_graph)
        throw std::length_error(
            "polar correction-multipole team count exceeds INT_MAX.");
    const int multipole_team_count = graphs * teams_per_graph;
    using MultipolePolicy =
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using MultipoleMember = typename MultipolePolicy::member_type;
    Kokkos::parallel_for("polar chain correction multipoles",
        MultipolePolicy(multipole_team_count, Kokkos::AUTO),
        KOKKOS_LAMBDA(const MultipoleMember& team) {
            const int league = team.league_rank();
            const int graph = league / teams_per_graph;
            const int channel =
                (league / multipole_quantity_count) % spin_channel_count;
            const int quantity = league % multipole_quantity_count;
            const int offset = channel * density_width;
            Precision value = Precision(0);
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, nodes),
                [=](const int node, Precision& sum) {
                    if (node_graph(node) != graph)
                        return;
                    const Precision q = density(node, offset);
                    if (quantity == 0) {
                        sum += q;
                    } else if (quantity <= 3) {
                        const int component = quantity - 1;
                        const int column = component == 0 ? 3 : component;
                        const Precision local = density_width > column
                            ? density(node, offset + column) : Precision(0);
                        sum += q * positions(node, component) + local;
                    } else {
                        const Precision local[3] = {
                            density_width > 3
                                ? density(node, offset + 3) : Precision(0),
                            density_width > 1
                                ? density(node, offset + 1) : Precision(0),
                            density_width > 2
                                ? density(node, offset + 2) : Precision(0)};
                        Precision r2 = Precision(0);
                        Precision p_dot_r = Precision(0);
                        for (int component = 0; component < 3; ++component) {
                            const Precision r = positions(node, component);
                            r2 += r * r;
                            p_dot_r += r * local[component];
                        }
                        sum += q * r2 + Precision(2) * p_dot_r;
                    }
                }, value);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                aggregates(2 * graph + channel, quantity) = value;
            });
        });

    auto channel_density = channel_density_;
    auto pbc = graph_pbc_device_;
    auto volumes = cache_.kernel_volumes;
    auto matrix = ops_.external_field_kernel_;
    auto norms = field_norms_device_;
    auto self_constants = si_feature_constants_device_;
    auto self_select = si_feature_select_device_;
    auto external = external_potential_device_;
    auto barycenter = barycenter_device_;
    const int field_dimension = field_dimension_;
    const bool subtract_self = contract_.subtract_feature_self_terms;
    constexpr double pi = 3.14159265358979323846;
    constexpr double constant = (1.0 / 5.526349406e-3) / (4.0 * pi);
    constexpr double madelung = 2.837297;
    for (int channel = 0; channel < 2; ++channel) {
        const int offset = channel * density_width;
        Kokkos::parallel_for("polar chain density channel",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                for (int column = 0; column < density_width; ++column)
                    channel_density(node, column) =
                        density(node, offset + column);
            });
        auto reciprocal_density = ops_.assemble_fourier_kernel(
            cache_, channel_density, /*use_workspace=*/true);
        auto reciprocal_potential = ops_.apply_coulomb_kernel(
            cache_, reciprocal_density, /*use_workspace=*/true);
        auto projected = ops_.project_to_features_kernel(
            cache_, reciprocal_potential, /*use_workspace=*/true);
        Kokkos::parallel_for("polar chain compose potential features",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                const int graph = node_graph(node);
                const int aggregate_row = 2 * graph + channel;
                const Precision charge = aggregates(aggregate_row, 0);
                const Precision volume = volumes(graph);
                const bool molecule = pbc(graph, 0) == 0
                    && pbc(graph, 1) == 0 && pbc(graph, 2) == 0;
                const bool slab = pbc(graph, 0) != 0
                    && pbc(graph, 1) != 0 && pbc(graph, 2) == 0;
                Precision correction_fields[4] = {
                    Precision(0), Precision(0), Precision(0), Precision(0)};
                if (molecule) {
                    const Precision r2 = positions(node, 0) * positions(node, 0)
                        + positions(node, 1) * positions(node, 1)
                        + positions(node, 2) * positions(node, 2);
                    Precision dipole_dot_r = Precision(0);
                    for (int component = 0; component < 3; ++component)
                        dipole_dot_r += aggregates(aggregate_row, 1 + component)
                            * positions(node, component);
                    const Precision box_length = Kokkos::pow(volume, 0.333333);
                    correction_fields[0] = madelung * constant * charge / box_length
                        - constant * 2.0 * pi * charge * r2 / (3.0 * volume)
                        + constant * 4.0 * pi * dipole_dot_r / (3.0 * volume)
                        - constant * 2.0 * pi * aggregates(aggregate_row, 4)
                            / (3.0 * volume);
                    for (int component = 0; component < 3; ++component)
                        correction_fields[1 + component] = 4.0 * pi * constant
                            * (aggregates(aggregate_row, 1 + component)
                                - charge * positions(node, component))
                            / (3.0 * volume);
                } else if (slab) {
                    const Precision field_z = 4.0 * pi * constant
                        * aggregates(aggregate_row, 3) / volume;
                    correction_fields[0] = field_z * positions(node, 2);
                    correction_fields[3] = field_z;
                }
                Precision external_fields[4];
                for (int component = 0; component < 4; ++component)
                    external_fields[component] = external(graph, component);
                for (int component = 0; component < 3; ++component)
                    external_fields[0] +=
                        (positions(node, component) - barycenter(graph, component))
                        * external(graph, 1 + component);
                const Precision correction_permuted[4] = {
                    correction_fields[0], correction_fields[3],
                    correction_fields[1], correction_fields[2]};
                const Precision external_permuted[4] = {
                    external_fields[0], external_fields[3],
                    external_fields[1], external_fields[2]};
                for (int feature = 0; feature < field_dimension; ++feature) {
                    Precision correction = Precision(0);
                    Precision external_value = Precision(0);
                    for (int component = 0; component < 4; ++component) {
                        correction += matrix(feature, component)
                            * correction_permuted[component];
                        external_value += matrix(feature, component)
                            * external_permuted[component];
                    }
                    Precision value = projected(node, feature) + correction;
                    if (subtract_self)
                        value -= self_constants(feature)
                            * channel_density(node, self_select(feature));
                    output(node, channel * field_dimension + feature) =
                        value / norms(feature)
                        + Precision(0.5) * external_value / norms(feature);
                }
            });
    }
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::adjoint_normalize_fukui_device(
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& raw,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& seed)
{
    auto sums = graph_reverse_sums_;
    auto node_graph = cache_.node_graph_index;
    const int nodes = nodes_;
    constexpr int fukui_adjoint_quantities_per_graph = 4;
    if (graphs_ > std::numeric_limits<int>::max()
            / fukui_adjoint_quantities_per_graph)
        throw std::length_error(
            "polar Fukui-adjoint team count exceeds INT_MAX.");
    const int fukui_adjoint_team_count =
        graphs_ * fukui_adjoint_quantities_per_graph;
    using FukuiAdjointSumPolicy =
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using FukuiAdjointSumMember = typename FukuiAdjointSumPolicy::member_type;
    Kokkos::parallel_for("polar chain fukui normalization adjoint sums",
        FukuiAdjointSumPolicy(fukui_adjoint_team_count, Kokkos::AUTO),
        KOKKOS_LAMBDA(const FukuiAdjointSumMember& team) {
            const int league = team.league_rank();
            const int graph = league / fukui_adjoint_quantities_per_graph;
            const int local = league % fukui_adjoint_quantities_per_graph;
            const int channel = local / 2;
            const int quantity = local % 2;
            Precision value = Precision(0);
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, nodes),
                [=](const int node, Precision& sum) {
                    if (node_graph(node) != graph)
                        return;
                    if (quantity == 0)
                        sum += raw(node, channel);
                    else
                        sum += seed(node, channel) * raw(node, channel);
                }, value);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                sums(graph, 2 * channel + quantity) = value;
            });
        });
    Kokkos::parallel_for("polar chain fukui normalization adjoint",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
        KOKKOS_LAMBDA(const int node) {
            const int graph = node_graph(node);
            for (int channel = 0; channel < 2; ++channel) {
                const Precision sum = sums(graph, 2 * channel);
                const Precision norm = sum == Precision(0) ? Precision(1) : sum;
                const Precision norm_term = sum == Precision(0)
                    ? Precision(0)
                    : sums(graph, 2 * channel + 1) / (norm * norm);
                seed(node, channel) = seed(node, channel) / norm
                    - norm_term;
            }
        });
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::adjoint_equilibrate_device(
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& sources,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& unequilibrated,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& state_seed,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& source_seed,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& target_seed)
{
    auto sums = graph_reverse_sums_;
    auto node_graph = cache_.node_graph_index;
    auto total_charge = total_charge_device_;
    auto total_spin = total_spin_device_;
    const int nodes = nodes_;
    const int density_width = density_width_;
    constexpr int equilibration_adjoint_quantities_per_graph = 4;
    if (graphs_ > std::numeric_limits<int>::max()
            / equilibration_adjoint_quantities_per_graph)
        throw std::length_error(
            "polar equilibration-adjoint team count exceeds INT_MAX.");
    const int equilibration_adjoint_team_count =
        graphs_ * equilibration_adjoint_quantities_per_graph;
    using EquilibrationAdjointSumPolicy =
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using EquilibrationAdjointSumMember =
        typename EquilibrationAdjointSumPolicy::member_type;
    Kokkos::parallel_for("polar chain equilibration adjoint sums",
        EquilibrationAdjointSumPolicy(
            equilibration_adjoint_team_count, Kokkos::AUTO),
        KOKKOS_LAMBDA(const EquilibrationAdjointSumMember& team) {
            const int league = team.league_rank();
            const int graph =
                league / equilibration_adjoint_quantities_per_graph;
            const int local =
                league % equilibration_adjoint_quantities_per_graph;
            const int channel = local / 2;
            const int quantity = local % 2;
            const int column = channel == 0 ? 0 : density_width;
            Precision value = Precision(0);
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, nodes),
                [=](const int node, Precision& sum) {
                    if (node_graph(node) != graph)
                        return;
                    if (quantity == 0)
                        sum += unequilibrated(node, column);
                    else
                        sum += state_seed(node, column)
                            * sources(node, channel);
                }, value);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                sums(graph, (quantity == 0 ? 4 : 6) + channel) = value;
            });
        });
    Kokkos::parallel_for("polar chain equilibration target adjoint",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs_),
        KOKKOS_LAMBDA(const int graph) {
            target_seed(graph, 0) += Precision(0.5)
                * (sums(graph, 6) + sums(graph, 7));
            target_seed(graph, 1) += Precision(0.5)
                * (sums(graph, 6) - sums(graph, 7));
        });
    Kokkos::parallel_for("polar chain equilibration adjoint",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
        KOKKOS_LAMBDA(const int node) {
            const int graph = node_graph(node);
            const Precision alpha_target =
                (total_charge(graph) + total_spin(graph) - Precision(1))
                    / Precision(2)
                - sums(graph, 4);
            const Precision beta_target =
                (total_charge(graph) - total_spin(graph) + Precision(1))
                    / Precision(2)
                - sums(graph, 5);
            source_seed(node, 0) += state_seed(node, 0) * alpha_target;
            source_seed(node, 1) +=
                state_seed(node, density_width) * beta_target;
            state_seed(node, 0) -= sums(graph, 6);
            state_seed(node, density_width) -= sums(graph, 7);
        });
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::adjoint_potential_features_device(
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& density,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& feature_seed,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& density_seed,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& position_seed,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& external_seed,
    const Kokkos::View<Precision**, Kokkos::LayoutRight>& correction_strain,
    const execution::long_range::LongRangeKernelMatrix<Precision>& grad_cos,
    const execution::long_range::LongRangeKernelMatrix<Precision>& grad_sin,
    const execution::long_range::LongRangeKernelMatrix<Precision>& grad_k,
    const execution::long_range::LongRangeKernelVector<Precision>& grad_volume,
    const bool use_reference)
{
#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
    if (solver_decision_.resolved
        == execution::long_range::LongRangeSolver::pme) {
        auto pme_seeds = pme_feature_seeds_;
        auto norms = field_norms_device_;
        const int field_dimension = field_dimension_;
        Kokkos::parallel_for("polar chain PME potential adjoint prepare",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
            KOKKOS_LAMBDA(const int node) {
                for (int channel = 0; channel < 2; ++channel)
                    for (int feature = 0; feature < field_dimension; ++feature) {
                        const int lane = channel * field_dimension + feature;
                        pme_seeds(node, lane) =
                            feature_seed(node, lane) / norms(feature);
                    }
            });
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            pme_energy_seeds_, 0.0);
        std::array<double, 9> cell{};
        std::copy_n(cell_.data(), cell.size(), cell.begin());
        const std::array<int, 3> pbc{graph_pbc_[0], graph_pbc_[1],
            graph_pbc_[2]};
        particle_mesh_workspace_->reverse_channels_device(
            typename PolarParticleMeshWorkspaceT<Precision>::
                const_device_double_matrix(cache_.realspace_positions),
            typename PolarParticleMeshWorkspaceT<Precision>::const_device_matrix(
                density),
            typename PolarParticleMeshWorkspaceT<Precision>::const_device_matrix(
                pme_seeds),
            typename PolarParticleMeshWorkspaceT<Precision>::const_device_vector(
                pme_energy_seeds_),
            pme_forward_features_, pme_channel_energies_,
            pme_source_adjoints_, pme_position_adjoints_, pme_cell_adjoints_,
            cell, pbc, contract_.long_range.density_smearing_width,
            std::span<const double>(
                contract_.long_range.feature_smearing_widths.data(),
                contract_.long_range.feature_smearing_widths.size()),
            current_inputs_pme_mesh_, current_inputs_pme_spline_order_,
            !contract_.subtract_feature_self_terms,
            contract_.long_range.include_electrostatic_self_interaction, 2,
            true, false);

        auto pme_source = pme_source_adjoints_;
        auto pme_position = pme_position_adjoints_;
        const int charges_dimension = charges_dimension_;
        const int nodes = nodes_;
        Kokkos::parallel_for("polar chain PME potential adjoint accumulate",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
            KOKKOS_LAMBDA(const int node) {
                for (int column = 0; column < charges_dimension; ++column)
                    density_seed(node, column) += pme_source(node, column);
                for (int component = 0; component < 3; ++component)
                    position_seed(node, component) +=
                        static_cast<Precision>(pme_position(node, component));
            });
        auto positions = cache_.realspace_positions;
        using CellPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using CellMember = typename CellPolicy::member_type;
        auto pme_cell = pme_cell_adjoints_;
        Kokkos::parallel_for("polar chain PME fixed-Cartesian cell adjoint",
            CellPolicy(9, Kokkos::AUTO),
            KOKKOS_LAMBDA(const CellMember& team) {
                const int index = team.league_rank();
                const int row = index / 3;
                const int column = index % 3;
                double virial = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, nodes),
                    [=](const int node, double& sum) {
                        sum += positions(node, row)
                            * pme_position(node, column);
                    }, virial);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    correction_strain(0, index) += static_cast<Precision>(
                        pme_cell(index) - virial);
                });
            });

        auto node_graph = cache_.node_graph_index;
        auto matrix = ops_.external_field_kernel_;
        auto external = external_potential_device_;
        auto barycenter = realspace_barycenter_device_;
        auto external_node = external_node_adjoint_;
        auto external_graph = external_graph_adjoint_;
        auto graph_node_counts = graph_node_counts_device_;
        const int graphs = graphs_;
        for (int channel = 0; channel < 2; ++channel) {
            Kokkos::parallel_for("polar chain PME external adjoint nodes",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
                KOKKOS_LAMBDA(const int node) {
                    for (int component = 0; component < 4; ++component) {
                        Precision value = Precision(0);
                        for (int feature = 0; feature < field_dimension; ++feature)
                            value += Precision(0.5)
                                * matrix(feature, component)
                                * feature_seed(
                                    node, channel * field_dimension + feature)
                                / norms(feature);
                        external_node(node, component) = value;
                    }
                });
            using ExternalPolicy =
                Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
            using ExternalMember = typename ExternalPolicy::member_type;
            Kokkos::parallel_for("polar chain PME external adjoint reduction",
                ExternalPolicy(graphs * 4, Kokkos::AUTO),
                KOKKOS_LAMBDA(const ExternalMember& team) {
                    const int index = team.league_rank();
                    const int graph = index / 4;
                    const int quantity = index % 4;
                    Precision value = Precision(0);
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, nodes),
                        [=](const int node, Precision& sum) {
                            if (node_graph(node) != graph) return;
                            if (quantity == 0) {
                                sum += external_node(node, 0);
                            } else {
                                const int component = quantity - 1;
                                const int field_column[3] = {2, 3, 1};
                                sum += external_node(node, field_column[component])
                                    + external_node(node, 0)
                                        * static_cast<Precision>(
                                            positions(node, component)
                                            - barycenter(graph, component));
                            }
                        }, value);
                    Kokkos::single(Kokkos::PerTeam(team), [=]() {
                        if (quantity == 0)
                            external_graph(graph, 0) = value;
                        external_seed(graph, quantity) += value;
                    });
                });
            Kokkos::parallel_for("polar chain PME external position adjoint",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
                KOKKOS_LAMBDA(const int node) {
                    const int graph = node_graph(node);
                    const int count = graph_node_counts(graph);
                    const Precision mean = count == 0 ? Precision(0)
                        : external_graph(graph, 0) / Precision(count);
                    for (int component = 0; component < 3; ++component)
                        position_seed(node, component) +=
                            external(graph, 1 + component)
                            * (external_node(node, 0) - mean);
                });
        }
        static_cast<void>(grad_cos);
        static_cast<void>(grad_sin);
        static_cast<void>(grad_k);
        static_cast<void>(grad_volume);
        static_cast<void>(use_reference);
        return;
    }
#endif
    if (solver_decision_.resolved
        == execution::long_range::LongRangeSolver::realspace) {
        auto node_graph = cache_.node_graph_index;
        auto positions = cache_.realspace_positions;
        auto matrix = ops_.external_field_kernel_;
        auto norms = field_norms_device_;
        auto external = external_potential_device_;
        auto barycenter = realspace_barycenter_device_;
        auto realspace_position_seed = realspace_position_seed_device_;
        const int nodes = nodes_;
        const int graphs = graphs_;
        const int density_width = density_width_;
        const int field_dimension = field_dimension_;
        const bool include_self = !contract_.subtract_feature_self_terms;
        for (int channel = 0; channel < 2; ++channel) {
            const int offset = channel * density_width;
            auto channel_density = channel_density_;
            auto channel_seed = channel_seed_;
            auto projected_seed = projected_seed_;
            Kokkos::parallel_for("polar chain realspace potential adjoint prepare",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
                KOKKOS_LAMBDA(const int node) {
                    for (int column = 0; column < density_width; ++column)
                        channel_density(node, column) =
                            density(node, offset + column);
                    for (int feature = 0; feature < field_dimension; ++feature) {
                        const Precision raw = feature_seed(
                            node, channel * field_dimension + feature);
                        channel_seed(node, feature) = raw;
                        projected_seed(node, feature) = raw / norms(feature);
                    }
                });
            const auto realspace = ops_.adjoint_realspace_features_kernel(
                cache_, channel_density, projected_seed, include_self,
                /*use_workspace=*/true);
            auto multipole_seed = realspace.grad_multipoles;
            auto stage_position_seed = realspace.grad_positions;
            Kokkos::parallel_for(
                "polar chain realspace potential adjoint accumulate",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
                KOKKOS_LAMBDA(const int node) {
                    for (int column = 0; column < density_width; ++column)
                        density_seed(node, offset + column) +=
                            multipole_seed(node, column);
                    for (int component = 0; component < 3; ++component)
                        realspace_position_seed(node, component) +=
                            stage_position_seed(node, component);
                });

            auto external_node = external_node_adjoint_;
            auto external_graph = external_graph_adjoint_;
            auto graph_node_counts = graph_node_counts_device_;
            Kokkos::parallel_for(
                "polar chain realspace external adjoint node contributions",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
                KOKKOS_LAMBDA(const int node) {
                    for (int component = 0; component < 4; ++component) {
                        Precision value = Precision(0);
                        for (int feature = 0; feature < field_dimension;
                            ++feature)
                            value += Precision(0.5)
                                * matrix(feature, component)
                                * channel_seed(node, feature) / norms(feature);
                        external_node(node, component) = value;
                    }
                });
            constexpr int external_quantity_count = 4;
            if (graphs > std::numeric_limits<int>::max()
                    / external_quantity_count)
                throw std::length_error(
                    "polar external-adjoint team count exceeds INT_MAX.");
            using ExternalPolicy =
                Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
            using ExternalMember = typename ExternalPolicy::member_type;
            Kokkos::parallel_for(
                "polar chain realspace external adjoint graph reduction",
                ExternalPolicy(
                    graphs * external_quantity_count, Kokkos::AUTO),
                KOKKOS_LAMBDA(const ExternalMember& team) {
                    const int league = team.league_rank();
                    const int graph = league / external_quantity_count;
                    const int quantity = league % external_quantity_count;
                    Precision value = Precision(0);
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, nodes),
                        [=](const int node, Precision& sum) {
                            if (node_graph(node) != graph)
                                return;
                            if (quantity == 0) {
                                sum += external_node(node, 0);
                            } else {
                                const int component = quantity - 1;
                                const int field_column[3] = {2, 3, 1};
                                sum += external_node(
                                           node, field_column[component])
                                    + external_node(node, 0)
                                        * static_cast<Precision>(
                                            positions(node, component)
                                            - static_cast<double>(barycenter(
                                                graph, component)));
                            }
                        }, value);
                    Kokkos::single(Kokkos::PerTeam(team), [=]() {
                        if (quantity == 0) {
                            external_graph(graph, 0) = value;
                            external_seed(graph, 0) += value;
                        } else {
                            external_seed(graph, quantity) += value;
                        }
                    });
                });
            Kokkos::parallel_for(
                "polar chain realspace external adjoint node apply",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
                KOKKOS_LAMBDA(const int node) {
                    const int graph = node_graph(node);
                    const int count = graph_node_counts(graph);
                    const double mean = count == 0
                        ? 0.0
                        : static_cast<double>(external_graph(graph, 0))
                            / static_cast<double>(count);
                    for (int component = 0; component < 3; ++component)
                        realspace_position_seed(node, component) +=
                            static_cast<double>(
                                external(graph, 1 + component))
                            * (static_cast<double>(external_node(node, 0))
                                - mean);
                });
        }
        static_cast<void>(position_seed);
        static_cast<void>(correction_strain);
        static_cast<void>(grad_cos);
        static_cast<void>(grad_sin);
        static_cast<void>(grad_k);
        static_cast<void>(grad_volume);
        static_cast<void>(use_reference);
        return;
    }
    auto positions = cache_.positions_workspace;
    auto node_graph = cache_.node_graph_index;
    auto pbc = graph_pbc_device_;
    auto volumes = cache_.kernel_volumes;
    auto matrix = ops_.external_field_kernel_;
    auto norms = field_norms_device_;
    auto self_constants = si_feature_constants_device_;
    auto self_select = si_feature_select_device_;
    auto external = external_potential_device_;
    auto barycenter = barycenter_device_;
    auto aggregates = graph_multipoles_;
    const int nodes = nodes_;
    const int graphs = graphs_;
    const int density_width = density_width_;
    const int field_dimension = field_dimension_;
    const bool subtract_self = contract_.subtract_feature_self_terms;
    constexpr double pi = 3.14159265358979323846;
    constexpr double constant = (1.0 / 5.526349406e-3) / (4.0 * pi);
    constexpr double madelung = 2.837297;

    // The aggregate spelling is identical to the forward's and is replayed
    // from the retained density for the step being differentiated.
    if (use_reference) {
        Kokkos::parallel_for("polar chain correction adjoint multipoles",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs * 2),
            KOKKOS_LAMBDA(const int index) {
                const int graph = index / 2;
                const int channel = index % 2;
                const int offset = channel * density_width;
                Precision charge = Precision(0), quadrupole = Precision(0);
                Precision dipole[3] = {
                    Precision(0), Precision(0), Precision(0)};
                for (int node = 0; node < nodes; ++node) {
                    if (node_graph(node) != graph)
                        continue;
                    const Precision q = density(node, offset);
                    const Precision local[3] = {
                        density_width > 3
                            ? density(node, offset + 3) : Precision(0),
                        density_width > 1
                            ? density(node, offset + 1) : Precision(0),
                        density_width > 2
                            ? density(node, offset + 2) : Precision(0)};
                    Precision r2 = Precision(0), p_dot_r = Precision(0);
                    charge += q;
                    for (int component = 0; component < 3; ++component) {
                        dipole[component] += q * positions(node, component)
                            + local[component];
                        r2 += positions(node, component)
                            * positions(node, component);
                        p_dot_r += positions(node, component) * local[component];
                    }
                    quadrupole += q * r2 + Precision(2) * p_dot_r;
                }
                aggregates(index, 0) = charge;
                for (int component = 0; component < 3; ++component)
                    aggregates(index, 1 + component) = dipole[component];
                aggregates(index, 4) = quadrupole;
            });
    } else {
        constexpr int spin_channel_count = 2;
        constexpr int multipole_quantity_count = 5;
        constexpr int teams_per_graph =
            spin_channel_count * multipole_quantity_count;
        if (graphs > std::numeric_limits<int>::max() / teams_per_graph)
            throw std::length_error(
                "polar correction-adjoint multipole team count exceeds INT_MAX.");
        const int multipole_team_count = graphs * teams_per_graph;
        using MultipolePolicy =
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using MultipoleMember = typename MultipolePolicy::member_type;
        Kokkos::parallel_for("polar chain correction adjoint multipoles",
            MultipolePolicy(multipole_team_count, Kokkos::AUTO),
            KOKKOS_LAMBDA(const MultipoleMember& team) {
                const int league = team.league_rank();
                const int graph = league / teams_per_graph;
                const int channel =
                    (league / multipole_quantity_count) % spin_channel_count;
                const int quantity = league % multipole_quantity_count;
                const int offset = channel * density_width;
                Precision value = Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, nodes),
                    [=](const int node, Precision& sum) {
                        if (node_graph(node) != graph)
                            return;
                        const Precision q = density(node, offset);
                        if (quantity == 0) {
                            sum += q;
                        } else if (quantity <= 3) {
                            const int component = quantity - 1;
                            const int column = component == 0 ? 3 : component;
                            const Precision local = density_width > column
                                ? density(node, offset + column) : Precision(0);
                            sum += q * positions(node, component) + local;
                        } else {
                            const Precision local[3] = {
                                density_width > 3
                                    ? density(node, offset + 3) : Precision(0),
                                density_width > 1
                                    ? density(node, offset + 1) : Precision(0),
                                density_width > 2
                                    ? density(node, offset + 2) : Precision(0)};
                            Precision r2 = Precision(0), p_dot_r = Precision(0);
                            for (int component = 0; component < 3; ++component) {
                                const Precision r = positions(node, component);
                                r2 += r * r;
                                p_dot_r += r * local[component];
                            }
                            sum += q * r2 + Precision(2) * p_dot_r;
                        }
                    }, value);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    aggregates(2 * graph + channel, quantity) = value;
                });
            });
    }

    for (int channel = 0; channel < 2; ++channel) {
        const int offset = channel * density_width;
        auto channel_density = channel_density_;
        auto channel_seed = channel_seed_;
        auto projected_seed = projected_seed_;
        auto node_fields = node_field_seed_;
        Kokkos::parallel_for("polar chain potential adjoint prepare",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                Precision permuted[4] = {Precision(0), Precision(0),
                    Precision(0), Precision(0)};
                for (int column = 0; column < density_width; ++column)
                    channel_density(node, column) = density(node, offset + column);
                for (int feature = 0; feature < field_dimension; ++feature) {
                    const Precision raw =
                        feature_seed(node, channel * field_dimension + feature);
                    channel_seed(node, feature) = raw;
                    projected_seed(node, feature) = raw / norms(feature);
                    for (int component = 0; component < 4; ++component)
                        permuted[component] += matrix(feature, component)
                            * projected_seed(node, feature);
                }
                node_fields(node, 0) = permuted[0];
                node_fields(node, 1) = permuted[2];
                node_fields(node, 2) = permuted[3];
                node_fields(node, 3) = permuted[1];
            });
        auto reciprocal = ops_.adjoint_coulomb_features_kernel(cache_,
            channel_density, projected_seed, grad_cos, grad_sin, grad_k,
            grad_volume);
        auto multipole_seed = multipole_seed_;
        Kokkos::parallel_for("polar chain reciprocal and self adjoint",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                for (int column = 0; column < density_width; ++column)
                    multipole_seed(node, column) = reciprocal(node, column);
                if (subtract_self)
                    for (int feature = 0; feature < field_dimension; ++feature)
                        multipole_seed(node, self_select(feature)) -=
                            self_constants(feature) * projected_seed(node, feature);
            });

        const auto staging_extents = validate_potential_adjoint_extents(
            static_cast<std::size_t>(nodes), static_cast<std::size_t>(graphs));
        const std::uint64_t node_extent = staging_extents.nodes;
        const std::uint64_t graph_extent = staging_extents.graphs;
        using IndexPolicy = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::uint64_t>>;

        if (!use_reference) {
            auto external_node = external_node_adjoint_;
            auto external_graph = external_graph_adjoint_;
            auto graph_node_counts = graph_node_counts_device_;
            // The forward external term uses h_n = 0.5 M^T(f_n / norm),
            // E_ext[0] = sum_n h_n0, and
            // E_ext[d+1] = sum_n(h_n,p(d) + h_n0(r_nd-rbar_gd)).
            // Its transpose first forms each h_n independently. Dedicated
            // graph/quantity teams own the reductions, and the final node
            // workers uniquely own the position rows. No atomics are needed.
            Kokkos::parallel_for(
                "polar chain external adjoint node contributions",
                IndexPolicy(0, node_extent),
                KOKKOS_LAMBDA(const std::uint64_t flat) {
                    const int node = static_cast<int>(flat);
                    for (int component = 0; component < 4; ++component) {
                        Precision value = Precision(0);
                        for (int feature = 0; feature < field_dimension;
                             ++feature)
                            value += Precision(0.5)
                                * matrix(feature, component)
                                * channel_seed(node, feature) / norms(feature);
                        external_node(node, component) = value;
                    }
                });
            constexpr int external_reduction_quantity_count = 4;
            if (graphs > std::numeric_limits<int>::max()
                    / external_reduction_quantity_count)
                throw std::length_error(
                    "polar external-adjoint team count exceeds INT_MAX.");
            const int external_reduction_team_count =
                graphs * external_reduction_quantity_count;
            using ExternalReductionPolicy =
                Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
            using ExternalReductionMember =
                typename ExternalReductionPolicy::member_type;
            Kokkos::parallel_for("polar chain external adjoint graph reduction",
                ExternalReductionPolicy(
                    external_reduction_team_count, Kokkos::AUTO),
                KOKKOS_LAMBDA(const ExternalReductionMember& team) {
                    const int league = team.league_rank();
                    const int graph =
                        league / external_reduction_quantity_count;
                    const int quantity =
                        league % external_reduction_quantity_count;
                    Precision value = Precision(0);
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, nodes),
                        [=](const int node, Precision& sum) {
                            if (node_graph(node) != graph)
                                return;
                            if (quantity == 0) {
                                sum += external_node(node, 0);
                            } else {
                                const int d = quantity - 1;
                                const int field_column[3] = {2, 3, 1};
                                sum += external_node(node, field_column[d])
                                    + external_node(node, 0)
                                        * (positions(node, d)
                                            - barycenter(graph, d));
                            }
                        }, value);
                    Kokkos::single(Kokkos::PerTeam(team), [=]() {
                        if (quantity == 0) {
                            external_graph(graph, 0) = value;
                            external_seed(graph, 0) += value;
                        } else {
                            external_seed(graph, quantity) += value;
                        }
                    });
                });
            Kokkos::parallel_for("polar chain external adjoint node apply",
                IndexPolicy(0, node_extent),
                KOKKOS_LAMBDA(const std::uint64_t flat) {
                    const int node = static_cast<int>(flat);
                    const int graph = node_graph(node);
                    const int count = graph_node_counts(graph);
                    const Precision mean = count == 0
                        ? Precision(0)
                        : external_graph(graph, 0)
                            / static_cast<Precision>(count);
                    for (int d = 0; d < 3; ++d)
                        position_seed(node, d) += external(graph, 1 + d)
                            * (external_node(node, 0)
                                - mean);
                });
        } else {
            Kokkos::parallel_for("polar chain external field adjoint reference",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs),
                KOKKOS_LAMBDA(const int graph) {
                Precision gauge_sum = Precision(0);
                int count = 0;
                for (int node = 0; node < nodes; ++node)
                    if (node_graph(node) == graph) {
                        Precision h0 = Precision(0);
                        for (int feature = 0; feature < field_dimension; ++feature)
                            h0 += Precision(0.5) * matrix(feature, 0)
                                * channel_seed(node, feature) / norms(feature);
                        gauge_sum += h0;
                        ++count;
                    }
                const Precision mean = count == 0
                    ? Precision(0) : gauge_sum / Precision(count);
                for (int node = 0; node < nodes; ++node) {
                    if (node_graph(node) != graph)
                        continue;
                    Precision h[4] = {Precision(0), Precision(0),
                        Precision(0), Precision(0)};
                    for (int feature = 0; feature < field_dimension; ++feature)
                        for (int component = 0; component < 4; ++component)
                            h[component] += Precision(0.5)
                                * matrix(feature, component)
                                * channel_seed(node, feature) / norms(feature);
                    external_seed(graph, 0) += h[0];
                    const int field_column[3] = {2, 3, 1};
                    for (int d = 0; d < 3; ++d) {
                        position_seed(node, d) += external(graph, 1 + d)
                            * (h[0] - mean);
                        external_seed(graph, 1 + d) += h[field_column[d]]
                            + h[0] * (positions(node, d) - barycenter(graph, d));
                    }
                }
                });
        }

        if (!use_reference) {
        // The forward correction maps each graph/channel's five multipoles to
        // each owned node's four correction fields and then through the field
        // matrix. Its transpose therefore has (1) node-local contributions to
        // the five multipole seeds and volume seed, (2) one deterministic
        // node-order reduction per graph, and (3) node-local application of
        // those graph seeds.  No two workers write the same node.  The graph
        // reduction retains the reference's node order, so FP32 is repeatable
        // without atomics while the O(nodes) work launches across the device.
        potential_adjoint_diagnostics_.selected_count += 1;
        potential_adjoint_diagnostics_.selected_node_extent = node_extent;
        potential_adjoint_diagnostics_.selected_graph_extent = graph_extent;
        auto node_adjoint = correction_node_adjoint_;
        auto graph_adjoint = correction_graph_adjoint_;
        Kokkos::parallel_for("polar chain correction adjoint node contributions",
            IndexPolicy(0, node_extent), KOKKOS_LAMBDA(const std::uint64_t flat) {
                const int node = static_cast<int>(flat);
                const int graph = node_graph(node);
                const bool molecule = pbc(graph, 0) == 0
                    && pbc(graph, 1) == 0 && pbc(graph, 2) == 0;
                const bool slab = pbc(graph, 0) != 0
                    && pbc(graph, 1) != 0 && pbc(graph, 2) == 0;
                for (int column = 0; column < 6; ++column)
                    node_adjoint(node, column) = Precision(0);
                if (!molecule && !slab) return;
                const int aggregate_row = 2 * graph + channel;
                const Precision charge = aggregates(aggregate_row, 0);
                const Precision volume = volumes(graph);
                if (molecule) {
                    const Precision inverse_3v = Precision(1)
                        / (Precision(3) * volume);
                    const Precision inverse_3v_squared = inverse_3v / volume;
                    const Precision s0 = node_fields(node, 0);
                    Precision r2 = Precision(0), dipole_dot_r = Precision(0);
                    for (int d = 0; d < 3; ++d) {
                        r2 += positions(node, d) * positions(node, d);
                        dipole_dot_r += aggregates(aggregate_row, 1 + d)
                            * positions(node, d);
                    }
                    Precision seed_charge = s0 * (madelung * constant
                            / Kokkos::pow(volume, 0.333333)
                        - Precision(2) * constant * pi * r2 * inverse_3v);
                    node_adjoint(node, 1) = s0
                        * (-Precision(2) * constant * pi * inverse_3v);
                    for (int d = 0; d < 3; ++d) {
                        seed_charge += node_fields(node, 1 + d)
                            * (-Precision(4) * constant * pi
                                * positions(node, d) * inverse_3v);
                        node_adjoint(node, 2 + d) =
                            s0 * Precision(4) * constant * pi
                                * positions(node, d) * inverse_3v
                            + node_fields(node, 1 + d) * Precision(4)
                                * constant * pi * inverse_3v;
                        position_seed(node, d) += s0
                                * (-Precision(4) * constant * pi * charge
                                        * positions(node, d) * inverse_3v
                                    + Precision(4) * constant * pi
                                        * aggregates(aggregate_row, 1 + d)
                                        * inverse_3v)
                            + node_fields(node, 1 + d)
                                * (-Precision(4) * constant * pi * charge
                                    * inverse_3v);
                    }
                    node_adjoint(node, 0) = seed_charge;
                    const Precision length_derivative = -Precision(0.333333)
                        * Kokkos::pow(volume, -Precision(1.333333));
                    Precision volume_seed = s0
                        * (madelung * constant * charge * length_derivative
                            + Precision(2) * constant * pi * charge * r2
                                * inverse_3v_squared
                            - Precision(4) * constant * pi * dipole_dot_r
                                * inverse_3v_squared
                            + Precision(2) * constant * pi
                                * aggregates(aggregate_row, 4)
                                * inverse_3v_squared);
                    for (int d = 0; d < 3; ++d)
                        volume_seed += node_fields(node, 1 + d)
                            * (-Precision(4) * constant * pi
                                * (aggregates(aggregate_row, 1 + d)
                                    - charge * positions(node, d))
                                * inverse_3v_squared);
                    node_adjoint(node, 5) = volume_seed;
                } else {
                    const Precision field_scale = Precision(4) * constant * pi
                        / volume;
                    const Precision field_z = field_scale
                        * aggregates(aggregate_row, 3);
                    node_adjoint(node, 0) = node_fields(node, 0)
                        * positions(node, 2) + node_fields(node, 3);
                    position_seed(node, 2) += node_fields(node, 0) * field_z;
                }
            });
        constexpr int correction_adjoint_columns = 6;
        if (graphs > std::numeric_limits<int>::max()
                / correction_adjoint_columns)
            throw std::length_error(
                "polar correction-adjoint team count exceeds INT_MAX.");
        const int correction_team_count =
            graphs * correction_adjoint_columns;
        using CorrectionReductionPolicy =
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using CorrectionReductionMember =
            typename CorrectionReductionPolicy::member_type;
        Kokkos::parallel_for("polar chain correction adjoint graph reduction",
            CorrectionReductionPolicy(correction_team_count, Kokkos::AUTO),
            KOKKOS_LAMBDA(const CorrectionReductionMember& team) {
                const int league = team.league_rank();
                const int graph = league / correction_adjoint_columns;
                const int column = league % correction_adjoint_columns;
                Precision value = Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, nodes),
                    [=](const int node, Precision& sum) {
                        if (node_graph(node) == graph)
                            sum += node_adjoint(node, column);
                    }, value);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    graph_adjoint(graph, column) = value;
                });
            });
        Kokkos::parallel_for("polar chain correction adjoint node apply",
            IndexPolicy(0, node_extent), KOKKOS_LAMBDA(const std::uint64_t flat) {
                const int node = static_cast<int>(flat);
                const int graph = node_graph(node);
                const bool molecule = pbc(graph, 0) == 0
                    && pbc(graph, 1) == 0 && pbc(graph, 2) == 0;
                const bool slab = pbc(graph, 0) != 0
                    && pbc(graph, 1) != 0 && pbc(graph, 2) == 0;
                if (!molecule && !slab) return;
                if (molecule) {
                    const Precision seed_charge = graph_adjoint(graph, 0);
                    const Precision seed_quadrupole = graph_adjoint(graph, 1);
                    const Precision q = density(node, offset);
                    const Precision local[3] = {
                        density_width > 3 ? density(node, offset + 3) : Precision(0),
                        density_width > 1 ? density(node, offset + 1) : Precision(0),
                        density_width > 2 ? density(node, offset + 2) : Precision(0)};
                    Precision r2 = Precision(0), displacement_seed = Precision(0);
                    for (int d = 0; d < 3; ++d) {
                        r2 += positions(node, d) * positions(node, d);
                        displacement_seed += graph_adjoint(graph, 2 + d)
                            * positions(node, d);
                    }
                    multipole_seed(node, 0) += seed_charge
                        + displacement_seed + seed_quadrupole * r2;
                    const int dipole_column[3] = {3, 1, 2};
                    for (int d = 0; d < 3; ++d) {
                        const Precision seed_dipole = graph_adjoint(graph, 2 + d);
                        position_seed(node, d) += seed_dipole * q
                            + seed_quadrupole * (Precision(2) * q
                                * positions(node, d) + Precision(2) * local[d]);
                        if (density_width > 3)
                            multipole_seed(node, dipole_column[d]) += seed_dipole
                                + seed_quadrupole * Precision(2)
                                    * positions(node, d);
                    }
                } else {
                    const Precision volume = volumes(graph);
                    const Precision field_scale = Precision(4) * constant * pi
                        / volume;
                    const Precision seed_dipole = graph_adjoint(graph, 0)
                        * field_scale;
                    const Precision q = density(node, offset);
                    multipole_seed(node, 0) += seed_dipole * positions(node, 2);
                    if (density_width > 3)
                        multipole_seed(node, 2) += seed_dipole;
                    position_seed(node, 2) += seed_dipole * q;
                }
            });
        Kokkos::parallel_for("polar chain correction adjoint strain",
            IndexPolicy(0, graph_extent), KOKKOS_LAMBDA(const std::uint64_t flat) {
                const int graph = static_cast<int>(flat);
                const bool molecule = pbc(graph, 0) == 0
                    && pbc(graph, 1) == 0 && pbc(graph, 2) == 0;
                const bool slab = pbc(graph, 0) != 0
                    && pbc(graph, 1) != 0 && pbc(graph, 2) == 0;
                if (!molecule && !slab) return;
                Precision volume_seed = graph_adjoint(graph, 5);
                if (slab) {
                    const int aggregate_row = 2 * graph + channel;
                    const Precision volume = volumes(graph);
                    const Precision field_z = Precision(4) * constant * pi
                        * aggregates(aggregate_row, 3) / volume;
                    volume_seed = -field_z * graph_adjoint(graph, 0) / volume;
                }
                const Precision diagonal = volume_seed * volumes(graph);
                correction_strain(graph, 0) += diagonal;
                correction_strain(graph, 4) += diagonal;
                correction_strain(graph, 8) += diagonal;
            });
        } else {
            ++potential_adjoint_diagnostics_.reference_count;
            Kokkos::parallel_for("polar chain correction feature adjoint reference",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs),
                KOKKOS_LAMBDA(const int graph) {
                    const bool molecule = pbc(graph, 0) == 0
                        && pbc(graph, 1) == 0 && pbc(graph, 2) == 0;
                    const bool slab = pbc(graph, 0) != 0
                        && pbc(graph, 1) != 0 && pbc(graph, 2) == 0;
                    if (!molecule && !slab) return;
                    const int aggregate_row = 2 * graph + channel;
                    const Precision charge = aggregates(aggregate_row, 0);
                    const Precision volume = volumes(graph);
                    Precision volume_seed = Precision(0);
                    if (molecule) {
                        const Precision inverse_3v = Precision(1)
                            / (Precision(3) * volume);
                        const Precision inverse_3v_squared = inverse_3v / volume;
                        Precision seed_charge = Precision(0);
                        Precision seed_quadrupole = Precision(0);
                        Precision seed_dipole[3] = {
                            Precision(0), Precision(0), Precision(0)};
                        for (int node = 0; node < nodes; ++node) {
                            if (node_graph(node) != graph) continue;
                            const Precision s0 = node_fields(node, 0);
                            Precision r2 = Precision(0);
                            Precision dipole_dot_r = Precision(0);
                            for (int d = 0; d < 3; ++d) {
                                r2 += positions(node, d) * positions(node, d);
                                dipole_dot_r += aggregates(aggregate_row, 1 + d)
                                    * positions(node, d);
                            }
                            seed_charge += s0 * (madelung * constant
                                    / Kokkos::pow(volume, 0.333333)
                                - Precision(2) * constant * pi * r2 * inverse_3v);
                            seed_quadrupole += s0 * (-Precision(2) * constant
                                * pi * inverse_3v);
                            for (int d = 0; d < 3; ++d) {
                                seed_charge += node_fields(node, 1 + d)
                                    * (-Precision(4) * constant * pi
                                        * positions(node, d) * inverse_3v);
                                seed_dipole[d] += s0 * Precision(4) * constant
                                        * pi * positions(node, d) * inverse_3v
                                    + node_fields(node, 1 + d) * Precision(4)
                                        * constant * pi * inverse_3v;
                                position_seed(node, d) += s0
                                        * (-Precision(4) * constant * pi * charge
                                                * positions(node, d) * inverse_3v
                                            + Precision(4) * constant * pi
                                                * aggregates(aggregate_row, 1 + d)
                                                * inverse_3v)
                                    + node_fields(node, 1 + d)
                                        * (-Precision(4) * constant * pi * charge
                                            * inverse_3v);
                            }
                            const Precision length_derivative =
                                -Precision(0.333333)
                                * Kokkos::pow(volume, -Precision(1.333333));
                            volume_seed += s0
                                * (madelung * constant * charge
                                        * length_derivative
                                    + Precision(2) * constant * pi * charge * r2
                                        * inverse_3v_squared
                                    - Precision(4) * constant * pi * dipole_dot_r
                                        * inverse_3v_squared
                                    + Precision(2) * constant * pi
                                        * aggregates(aggregate_row, 4)
                                        * inverse_3v_squared);
                            for (int d = 0; d < 3; ++d)
                                volume_seed += node_fields(node, 1 + d)
                                    * (-Precision(4) * constant * pi
                                        * (aggregates(aggregate_row, 1 + d)
                                            - charge * positions(node, d))
                                        * inverse_3v_squared);
                        }
                        for (int node = 0; node < nodes; ++node) {
                            if (node_graph(node) != graph) continue;
                            const Precision q = density(node, offset);
                            const Precision local[3] = {
                                density_width > 3
                                    ? density(node, offset + 3) : Precision(0),
                                density_width > 1
                                    ? density(node, offset + 1) : Precision(0),
                                density_width > 2
                                    ? density(node, offset + 2) : Precision(0)};
                            Precision r2 = Precision(0);
                            Precision displacement_seed = Precision(0);
                            for (int d = 0; d < 3; ++d) {
                                r2 += positions(node, d) * positions(node, d);
                                displacement_seed += seed_dipole[d]
                                    * positions(node, d);
                            }
                            multipole_seed(node, 0) += seed_charge
                                + displacement_seed + seed_quadrupole * r2;
                            const int dipole_column[3] = {3, 1, 2};
                            for (int d = 0; d < 3; ++d) {
                                position_seed(node, d) += seed_dipole[d] * q
                                    + seed_quadrupole * (Precision(2) * q
                                        * positions(node, d)
                                        + Precision(2) * local[d]);
                                if (density_width > 3)
                                    multipole_seed(node, dipole_column[d]) +=
                                        seed_dipole[d] + seed_quadrupole
                                            * Precision(2) * positions(node, d);
                            }
                        }
                    } else {
                        const Precision field_scale = Precision(4) * constant
                            * pi / volume;
                        const Precision field_z = field_scale
                            * aggregates(aggregate_row, 3);
                        Precision seed_field = Precision(0);
                        for (int node = 0; node < nodes; ++node)
                            if (node_graph(node) == graph) {
                                seed_field += node_fields(node, 0)
                                        * positions(node, 2)
                                    + node_fields(node, 3);
                                position_seed(node, 2) +=
                                    node_fields(node, 0) * field_z;
                            }
                        const Precision seed_dipole = seed_field * field_scale;
                        for (int node = 0; node < nodes; ++node)
                            if (node_graph(node) == graph) {
                                const Precision q = density(node, offset);
                                multipole_seed(node, 0) += seed_dipole
                                    * positions(node, 2);
                                if (density_width > 3)
                                    multipole_seed(node, 2) += seed_dipole;
                                position_seed(node, 2) += seed_dipole * q;
                            }
                        volume_seed -= field_z * seed_field / volume;
                    }
                    const Precision diagonal = volume_seed * volume;
                    correction_strain(graph, 0) += diagonal;
                    correction_strain(graph, 4) += diagonal;
                    correction_strain(graph, 8) += diagonal;
                });
        }
        Kokkos::parallel_for("polar chain potential density adjoint accumulate",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                for (int column = 0; column < density_width; ++column)
                    density_seed(node, offset + column) +=
                        multipole_seed(node, column);
            });
    }
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::adjoint_normalize_fukui(
    const std::vector<double>& raw_sources,
    const std::vector<double>& normalized_seed,
    const std::vector<int>& node_graph_index,
    std::vector<double>& raw_seed) const
{
    // The forward's normalization, per graph and channel:
    //     y_i = x_i / norm,  norm = sum_j x_j  (held at one when it vanishes)
    // so the seed of a normalized source returns
    //     dx_i = dy_i/norm - sum_j (dy_j x_j) / norm^2,
    // and a graph whose norm the forward held at one carries no derivative on
    // that norm (the guard is the branch the forward selected).
    std::vector<double> norms(static_cast<std::size_t>(graphs_) * 2, 0.0);
    std::vector<double> numerators(norms.size(), 0.0);
    for (int node = 0; node < nodes_; ++node) {
        const std::size_t graph =
            static_cast<std::size_t>(node_graph_index[static_cast<std::size_t>(node)]);
        for (int channel = 0; channel < 2; ++channel)
            norms[2 * graph + channel] +=
                raw_sources[static_cast<std::size_t>(node) * 2 + channel];
    }
    for (int node = 0; node < nodes_; ++node) {
        const std::size_t graph =
            static_cast<std::size_t>(node_graph_index[static_cast<std::size_t>(node)]);
        for (int channel = 0; channel < 2; ++channel)
            numerators[2 * graph + channel] +=
                normalized_seed[static_cast<std::size_t>(node) * 2 + channel]
                * raw_sources[static_cast<std::size_t>(node) * 2 + channel];
    }
    for (std::size_t index = 0; index < norms.size(); ++index)
        if (norms[index] == 0.0)
            norms[index] = 1.0;
    for (int node = 0; node < nodes_; ++node) {
        const std::size_t graph =
            static_cast<std::size_t>(node_graph_index[static_cast<std::size_t>(node)]);
        for (int channel = 0; channel < 2; ++channel) {
            const std::size_t index = static_cast<std::size_t>(node) * 2 + channel;
            const std::size_t graph_index = 2 * graph + channel;
            raw_seed[index] +=
                normalized_seed[index] / norms[graph_index]
                - numerators[graph_index]
                    / (norms[graph_index] * norms[graph_index]);
        }
    }
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::correction_fields(
    const std::vector<double>& density,
    std::vector<double>& alpha_out, std::vector<double>& beta_out) const
{
    std::vector<double> alpha(static_cast<std::size_t>(nodes_) * density_width_);
    std::vector<double> beta(alpha.size());
    for (int node = 0; node < nodes_; ++node)
        for (int column = 0; column < density_width_; ++column) {
            const std::size_t row = static_cast<std::size_t>(node);
            alpha[row * density_width_ + column] =
                density[row * 2 * density_width_ + column];
            beta[row * density_width_ + column] =
                density[row * 2 * density_width_ + density_width_ + column];
        }
    const execution::long_range::LongRangeConstMatrix<double> alpha_view(
        alpha.data(), nodes_, density_width_);
    const execution::long_range::LongRangeConstMatrix<double> beta_view(
        beta.data(), nodes_, density_width_);
    const execution::long_range::LongRangeConstMatrix<double> positions(
        positions_.data(), nodes_, 3);
    const execution::long_range::LongRangeConstTensor3<double> cell(
        cell_.data(), graphs_, 3, 3);
    const execution::long_range::LongRangeConstVector<int> node_graph_index(
        node_graph_index_.data(), nodes_);
    const execution::long_range::LongRangeConstMatrix<int> graph_pbc(
        graph_pbc_.data(), graphs_, 3);
    auto result = ops_.analytic_correction_features(alpha_view, beta_view,
        positions, cell, node_graph_index, graph_pbc);
    alpha_out.assign(static_cast<std::size_t>(nodes_) * field_dimension_, 0.0);
    beta_out.assign(alpha_out.size(), 0.0);
    for (int node = 0; node < nodes_; ++node)
        for (int column = 0; column < field_dimension_; ++column) {
            const std::size_t row = static_cast<std::size_t>(node);
            alpha_out[row * field_dimension_ + column] =
                result.alpha(node, column);
            beta_out[row * field_dimension_ + column] =
                result.beta(node, column);
        }
}

template<typename Precision>
std::vector<double> PolarDeltaChainKokkosT<Precision>::feature_path(
    const std::vector<double>& channel_density,
    const std::vector<double>& corrections) const
{
    const execution::long_range::LongRangeConstMatrix<double> multipoles(
        channel_density.data(), nodes_, density_width_);
    auto projected = ops_.coulomb_features(
        cache_, multipoles, /*use_workspace=*/true);
    std::vector<double> features(
        static_cast<std::size_t>(nodes_) * field_dimension_);
    for (std::size_t node = 0; node < static_cast<std::size_t>(nodes_); ++node)
        for (int column = 0; column < field_dimension_; ++column)
            features[node * field_dimension_ + column] = projected(node, column);
    if (contract_.subtract_feature_self_terms) {
        const auto self_terms = ops_.feature_self_terms(multipoles);
        for (std::size_t node = 0; node < static_cast<std::size_t>(nodes_); ++node)
            for (int column = 0; column < field_dimension_; ++column)
                features[node * field_dimension_ + column] -=
                    self_terms(node, column);
    }
    for (std::size_t index = 0; index < features.size(); ++index)
        features[index] = (features[index] + corrections[index])
            / norms_(index % field_dimension_);
    return features;
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::equilibrate(
    const std::vector<double>& sources, const std::vector<double>& state,
    std::vector<double>& corrections_alpha,
    std::vector<double>& corrections_beta) const
{
    std::vector<double> l0_alpha(static_cast<std::size_t>(nodes_));
    std::vector<double> l0_beta(static_cast<std::size_t>(nodes_));
    for (int node = 0; node < nodes_; ++node) {
        const std::size_t row = static_cast<std::size_t>(node);
        l0_alpha[row] = state[row * 2 * density_width_];
        l0_beta[row] = state[row * 2 * density_width_ + density_width_];
    }
    const execution::long_range::LongRangeConstMatrix<double> sources_view(
        sources.data(), nodes_, 2);
    const execution::long_range::LongRangeConstVector<double> alpha_view(
        l0_alpha.data(), nodes_);
    const execution::long_range::LongRangeConstVector<double> beta_view(
        l0_beta.data(), nodes_);
    const execution::long_range::LongRangeConstVector<double> charge_view(
        total_charge_.data(), graphs_);
    const execution::long_range::LongRangeConstVector<double> spin_view(
        total_spin_.data(), graphs_);
    const execution::long_range::LongRangeConstVector<int> node_graph_view(
        node_graph_index_.data(), nodes_);
    auto result = ops_.fukui_normalize(sources_view, alpha_view, beta_view,
        charge_view, spin_view, node_graph_view);
    corrections_alpha.resize(static_cast<std::size_t>(nodes_));
    corrections_beta.resize(static_cast<std::size_t>(nodes_));
    for (int node = 0; node < nodes_; ++node) {
        corrections_alpha[static_cast<std::size_t>(node)] = result.alpha(node);
        corrections_beta[static_cast<std::size_t>(node)] = result.beta(node);
    }
}

template<typename Precision>
void PolarDeltaChainKokkosT<Precision>::record_input_h2d(
    const std::size_t elements,const std::size_t element_size,
    const char* what)
{
    if(element_size!=0
        &&elements>std::numeric_limits<std::size_t>::max()/element_size)
        throw std::overflow_error(
            std::string("polar chain ")+what+
            " input byte count overflows size_t.");
    const std::size_t bytes=elements*element_size;
    if(input_h2d_bytes_>std::numeric_limits<std::size_t>::max()-bytes)
        throw std::overflow_error(
            std::string("polar chain cumulative ")+what+
            " input byte count overflows size_t.");
    input_h2d_bytes_+=bytes;
}

template<typename Precision>
typename PolarDeltaChainKokkosT<Precision>::State
PolarDeltaChainKokkosT<Precision>::forward(
    const Inputs& inputs, const bool materialize_public_state)
{
    if (!ready_)
        throw std::invalid_argument("the polar delta chain is not parsed.");
    const auto potential_adjoint_extents = validate_potential_adjoint_extents(
        inputs.node_graph_index.size(), inputs.total_charge.size());
    const int nodes = static_cast<int>(potential_adjoint_extents.nodes);
    const int graphs = static_cast<int>(potential_adjoint_extents.graphs);
    validate_inputs(inputs, nodes, graphs);
    resolve_solver(inputs, nodes, graphs);
    const bool shape_changed = nodes_ != nodes || graphs_ != graphs;
    prepare(nodes, graphs);
    const bool node_graph_changed = shape_changed
        || node_graph_index_ != inputs.node_graph_index;
    const bool graph_channels_changed = graph_pbc_ != inputs.graph_pbc
        || external_potential_ != inputs.external_potential
        || total_charge_ != inputs.total_charge
        || total_spin_ != inputs.total_spin;
    positions_ = inputs.positions;
    cell_ = inputs.cell;
    node_graph_index_ = inputs.node_graph_index;
    graph_pbc_ = inputs.graph_pbc;
    external_potential_ = inputs.external_potential;
    total_charge_ = inputs.total_charge;
    total_spin_ = inputs.total_spin;
    current_inputs_pme_mesh_ =
        inputs.solver_request
                == execution::long_range::LongRangeSolverRequest::automatic
            && solver_decision_.resolved
                == execution::long_range::LongRangeSolver::pme
        ? automatic_pme_mesh(inputs)
        : inputs.pme_mesh;
    current_inputs_pme_spline_order_ = inputs.pme_spline_order;
    build_geometry_cache(inputs);
    if (solver_decision_.resolved
            == execution::long_range::LongRangeSolver::realspace
        || solver_decision_.resolved
            == execution::long_range::LongRangeSolver::pme) {
        auto volumes_host = Kokkos::create_mirror_view(reporting_volumes_device_);
        for (int graph = 0; graph < graphs; ++graph) {
            const double* c = inputs.cell.data()
                + static_cast<std::size_t>(graph) * 9;
            volumes_host(graph) = c[0] * (c[4] * c[8] - c[5] * c[7])
                - c[1] * (c[3] * c[8] - c[5] * c[6])
                + c[2] * (c[3] * c[7] - c[4] * c[6]);
        }
        Kokkos::deep_copy(reporting_volumes_device_, volumes_host);
        record_input_h2d(
            static_cast<std::size_t>(graphs),sizeof(double),
            "reporting volume");
    }
    if (node_graph_changed) {
        auto counts_host =
            Kokkos::create_mirror_view(graph_node_counts_device_);
        for (int graph = 0; graph < graphs; ++graph)
            counts_host(graph) = 0;
        for (const int graph : node_graph_index_)
            ++counts_host(graph);
        Kokkos::deep_copy(graph_node_counts_device_, counts_host);
        record_input_h2d(
            static_cast<std::size_t>(graphs),sizeof(int),"graph node count");
    }
    if (graph_channels_changed) {
        auto charge_host = Kokkos::create_mirror_view(total_charge_device_);
        auto spin_host = Kokkos::create_mirror_view(total_spin_device_);
        auto external_host =
            Kokkos::create_mirror_view(external_potential_device_);
        auto pbc_host = Kokkos::create_mirror_view(graph_pbc_device_);
        for (int graph = 0; graph < graphs; ++graph) {
            charge_host(graph) = static_cast<Precision>(total_charge_[graph]);
            spin_host(graph) = static_cast<Precision>(total_spin_[graph]);
            for (int component = 0; component < 4; ++component)
                external_host(graph, component) = static_cast<Precision>(
                    external_potential_[4 * graph + component]);
            for (int component = 0; component < 3; ++component)
                pbc_host(graph, component) = graph_pbc_[3 * graph + component];
        }
        Kokkos::deep_copy(total_charge_device_, charge_host);
        Kokkos::deep_copy(total_spin_device_, spin_host);
        Kokkos::deep_copy(external_potential_device_, external_host);
        Kokkos::deep_copy(graph_pbc_device_, pbc_host);
        record_input_h2d(
            static_cast<std::size_t>(graphs),sizeof(Precision),
            "total charge");
        record_input_h2d(
            static_cast<std::size_t>(graphs),sizeof(Precision),
            "total spin");
        record_input_h2d(
            static_cast<std::size_t>(graphs),4*sizeof(Precision),
            "external potential");
        record_input_h2d(
            static_cast<std::size_t>(graphs),3*sizeof(int),"graph pbc");
    }
    if (solver_decision_.resolved
            == execution::long_range::LongRangeSolver::realspace
        || solver_decision_.resolved
            == execution::long_range::LongRangeSolver::pme) {
        auto barycenter = realspace_barycenter_device_;
        auto positions_device = cache_.realspace_positions;
        auto node_graph_device = cache_.node_graph_index;
        auto graph_node_counts = graph_node_counts_device_;
        constexpr int barycenter_component_count = 3;
        if (graphs > std::numeric_limits<int>::max()
                / barycenter_component_count)
            throw std::length_error(
                "polar barycenter team count exceeds INT_MAX.");
        using BarycenterPolicy =
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using BarycenterMember = typename BarycenterPolicy::member_type;
        Kokkos::parallel_for("polar chain realspace graph barycenter",
            BarycenterPolicy(
                graphs * barycenter_component_count, Kokkos::AUTO),
            KOKKOS_LAMBDA(const BarycenterMember& team) {
                const int index = team.league_rank();
                const int graph = index / 3;
                const int component = index % 3;
                double sum = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, nodes),
                    [=](const int node, double& value) {
                        if (node_graph_device(node) == graph)
                            value += positions_device(node, component);
                    }, sum);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    const int count = graph_node_counts(graph);
                    barycenter(graph, component) = count == 0
                        ? 0.0
                        : sum / static_cast<double>(count);
                });
            });
    } else {
        auto barycenter = barycenter_device_;
        auto positions_device = cache_.positions_workspace;
        auto node_graph_device = cache_.node_graph_index;
        auto graph_node_counts = graph_node_counts_device_;
        constexpr int barycenter_component_count = 3;
        if (graphs > std::numeric_limits<int>::max()
                / barycenter_component_count)
            throw std::length_error(
                "polar barycenter team count exceeds INT_MAX.");
        using BarycenterPolicy =
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using BarycenterMember = typename BarycenterPolicy::member_type;
        Kokkos::parallel_for("polar chain graph barycenter",
            BarycenterPolicy(
                graphs * barycenter_component_count, Kokkos::AUTO),
            KOKKOS_LAMBDA(const BarycenterMember& team) {
                const int index = team.league_rank();
                const int graph = index / 3;
                const int component = index % 3;
                Precision sum = Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, nodes),
                    [=](const int node, Precision& value) {
                        if (node_graph_device(node) == graph)
                            value += positions_device(node, component);
                    }, sum);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    const int count = graph_node_counts(graph);
                    barycenter(graph, component) = count == 0
                        ? Precision(0)
                        : sum / static_cast<Precision>(count);
                });
            });
    }

    // The backbone's per-layer features and the model's node attributes.
    const int layers = contract_.interaction_count;
    const int width = contract_.node_feature_dimension;
    if (!inputs.node_feature_views.empty()) {
        for (int layer = 0; layer < layers; ++layer) {
            const auto destination = Kokkos::subview(node_features_,
                Kokkos::ALL,
                std::make_pair(layer * width, (layer + 1) * width));
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, destination,
                inputs.node_feature_views[static_cast<std::size_t>(layer)]);
        }
    } else {
        chain_store_values<Precision>(node_features_, inputs.node_features,
            "polar chain backbone features");
    }
    if (!inputs.node_types.empty()) {
        if (staged_node_types_ != inputs.node_types) {
            Kokkos::View<const int*, Kokkos::LayoutRight, Kokkos::HostSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_node_types(
                    inputs.node_types.data(), inputs.node_types.size());
            Kokkos::deep_copy(node_types_, host_node_types);
            record_input_h2d(
                inputs.node_types.size(),sizeof(int),"node type");
            const auto node_types = node_types_;
            auto attrs = node_attrs_;
            Kokkos::deep_copy(
                Kokkos::DefaultExecutionSpace{}, attrs, Precision(0));
            Kokkos::parallel_for("polar chain one-hot node attrs",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
                KOKKOS_LAMBDA(const int node) {
                    attrs(node, node_types(node)) = Precision(1);
                });
            staged_node_types_ = inputs.node_types;
        }
    } else {
        chain_store_values<Precision>(node_attrs_, inputs.node_attrs,
            "polar chain node attrs");
    }
    const auto layer_features = [this, width] (const int layer) {
        return Kokkos::subview(node_features_, Kokkos::ALL,
            std::make_pair(layer * width, (layer + 1) * width));
    };

    // The per-layer spin-source heads sum into the chain's density state and
    // the feature mixer's per-layer linears sum into the mixed features.  The
    // density state's step-0 entry is the pre-loop state the first update
    // consumes, and it stays that entry's value: the pre-loop equilibration
    // advances it in place, and the local-electron readout later reads it as
    // its ``charges_0``.
    auto density_state = density_steps_[0];
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        density_state, Precision(0));
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        features_mixed_, Precision(0));
    const int charge_width = charges_dimension_;
    const int feature_width = width;
    for (int layer = 0; layer < layers; ++layer) {
        const auto features = layer_features(layer);
        spin_source_heads_[static_cast<std::size_t>(layer)].evaluate(
            features, head_row_);
        if (spin_source_zero_charges_[static_cast<std::size_t>(layer)]) {
            auto row = head_row_;
            Kokkos::parallel_for(
                "polar chain zero charges",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
                KOKKOS_LAMBDA(const int node) { row(node, 0) = Precision(0); });
        }
        {
            auto density = density_state;
            auto head = head_row_;
            const std::size_t head_elements =
                static_cast<std::size_t>(nodes) * charge_width;
            Kokkos::parallel_for(
                "polar chain head sum",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, head_elements),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const int node = static_cast<int>(flat / charge_width);
                    const int column = static_cast<int>(flat % charge_width);
                    density(node, column) += head(node, column);
                });
        }
        feature_mixer_[static_cast<std::size_t>(layer)].evaluate(
            features, mixer_row_);
        {
            auto mixed = features_mixed_;
            auto mixer = mixer_row_;
            const std::size_t mixer_elements =
                static_cast<std::size_t>(nodes) * feature_width;
            Kokkos::parallel_for(
                "polar chain mixer sum",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, mixer_elements),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const int node = static_cast<int>(flat / feature_width);
                    const int column = static_cast<int>(flat % feature_width);
                    mixed(node, column) += mixer(node, column);
                });
        }
    }

    // The pre-loop Fukui equilibration.
    const auto last_layer = layer_features(layers - 1);
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, fukui_input_, last_layer);
    {
        auto input = fukui_input_;
        fukui_source_map_.evaluate(input, fukui_sources_);
    }
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        fukui_raw_steps_[0], fukui_sources_);
    normalize_fukui_device(fukui_raw_steps_[0], fukui_steps_[0]);
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        heads_density_device_, density_steps_[0]);
    equilibrate_device(fukui_steps_[0], density_steps_[0]);

    const int steps = this->steps();
    for (int step = 1; step <= steps; ++step) {
        auto potential_device =
            potential_features_steps_[static_cast<std::size_t>(step - 1)];
        potential_features_device(
            density_steps_[static_cast<std::size_t>(step - 1)],
            potential_device);
        field_updates_[static_cast<std::size_t>(step - 1)].evaluate(
            features_mixed_,
            potential_features_steps_[static_cast<std::size_t>(step - 1)],
            node_attrs_,
            density_steps_[static_cast<std::size_t>(step - 1)],
            update_output_);
        auto sources_device = source_steps_[static_cast<std::size_t>(step)];
        auto raw_fukui_device =
            fukui_raw_steps_[static_cast<std::size_t>(step)];
        auto previous_device =
            density_steps_[static_cast<std::size_t>(step - 1)];
        auto next_device = density_steps_[static_cast<std::size_t>(step)];
        auto update_device = update_output_;
        const int output_width = charges_dimension_ + 2;
        const std::size_t update_elements =
            static_cast<std::size_t>(nodes) * output_width;
        Kokkos::parallel_for("polar chain split update",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(0, update_elements),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const int node = static_cast<int>(flat / output_width);
                const int column = static_cast<int>(flat % output_width);
                if (column < charge_width) {
                    const Precision value = update_device(node, column);
                    sources_device(node, column) = value;
                    next_device(node, column) =
                        previous_device(node, column) + value;
                } else {
                    raw_fukui_device(node, column - charge_width) =
                        update_device(node, column);
                }
            });
        auto normalized_fukui_device =
            fukui_steps_[static_cast<std::size_t>(step)];
        normalize_fukui_device(raw_fukui_device, normalized_fukui_device);
        equilibrate_device(normalized_fukui_device, next_device);
    }

    // The post-loop local-electron readout: the last iteration's potential
    // features, the pre-loop density, and the equilibrated state -- all three
    // the resident per-step entries, in the positions the forward left them.
    local_electron_readout_.evaluate(last_layer,
        potential_features_steps_[static_cast<std::size_t>(steps - 1)],
        density_steps_[0], density_steps_[static_cast<std::size_t>(steps)],
        local_energy_);

    State result;
    result.nodes = static_cast<std::size_t>(nodes);
    result.graphs = static_cast<std::size_t>(graphs);
    if (materialize_public_state) {
        result.spin_charge_density =
            to_double(density_steps_[static_cast<std::size_t>(steps)]);
        ++diagnostic_state_materialization_count_;
        result.pre_loop_density = to_double(density_steps_[0]);
        ++diagnostic_state_materialization_count_;
        result.local_electron_energy = to_double(local_energy_);
        ++diagnostic_state_materialization_count_;
        result.charge_density.assign(
            static_cast<std::size_t>(nodes) * density_width_, 0.0);
        result.spin_density.assign(result.charge_density.size(), 0.0);
        for (int node = 0; node < nodes; ++node)
            for (int column = 0; column < density_width_; ++column) {
                const std::size_t row = static_cast<std::size_t>(node);
                const double alpha_value = result.spin_charge_density[
                    row * 2 * density_width_ + column];
                const double beta_value = result.spin_charge_density[
                    row * 2 * density_width_ + density_width_ + column];
                result.charge_density[row * density_width_ + column] =
                    alpha_value + beta_value;
                result.spin_density[row * density_width_ + column] =
                    alpha_value - beta_value;
            }
        for (int step = 0; step <= steps; ++step) {
            density_history_[static_cast<std::size_t>(step)] =
                to_double(density_steps_[static_cast<std::size_t>(step)]);
            ++diagnostic_state_materialization_count_;
            fukui_history_[static_cast<std::size_t>(step)] =
                to_double(fukui_steps_[static_cast<std::size_t>(step)]);
            ++diagnostic_state_materialization_count_;
        }
        for (int step = 0; step < steps; ++step) {
            potential_features_history_[static_cast<std::size_t>(step)] =
                to_double(potential_features_steps_[static_cast<std::size_t>(step)]);
            ++diagnostic_state_materialization_count_;
        }
        result.features_mixed = to_double(features_mixed_);
        ++diagnostic_state_materialization_count_;
        result.potential_features = potential_features_history_;
        result.fukui_sources = fukui_history_;
        result.density_history = density_history_;
    }
    return result;
}

template<typename Precision>
typename PolarDeltaChainKokkosT<Precision>::ComposedForwardResult
PolarDeltaChainKokkosT<Precision>::forward_composed_energy(
    const Inputs& inputs, const bool materialize_public_state,
    const bool materialize_outputs)
{
    ComposedForwardResult result;
    result.state = forward(inputs, materialize_public_state);
    const std::size_t nodes = result.state.nodes;
    const std::size_t graphs = result.state.graphs;
    auto spin_density = density_steps_.back();
    auto density = composed_density_device_;
    const int density_width = density_width_;
    Kokkos::parallel_for("polar composed spin sum",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
        KOKKOS_LAMBDA(const int node) {
            for (int column = 0; column < density_width; ++column)
                density(node, column) = spin_density(node, column)
                    + spin_density(node, density_width + column);
        });

    if (solver_decision_.resolved
            == execution::long_range::LongRangeSolver::realspace
        || solver_decision_.resolved
            == execution::long_range::LongRangeSolver::pme) {
        Kokkos::View<Precision*, Kokkos::LayoutRight> electrostatic;
        if (solver_decision_.resolved
            == execution::long_range::LongRangeSolver::realspace) {
            electrostatic = ops_.realspace_energy_kernel(
                cache_, density,
                contract_.long_range.include_electrostatic_self_interaction,
                /*use_workspace=*/true);
        } else {
#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
            std::array<double, 9> cell{};
            std::copy_n(cell_.data(), cell.size(), cell.begin());
            const std::array<int, 3> pbc{graph_pbc_[0], graph_pbc_[1],
                graph_pbc_[2]};
            try {
            particle_mesh_workspace_->forward_channels_device(
                typename PolarParticleMeshWorkspaceT<Precision>::
                    const_device_double_matrix(cache_.realspace_positions),
                typename PolarParticleMeshWorkspaceT<Precision>::const_device_matrix(
                    density),
                pme_energy_features_, pme_energy_channel_energy_, cell, pbc,
                contract_.long_range.density_smearing_width,
                std::span<const double>(
                    contract_.long_range.feature_smearing_widths.data(),
                    contract_.long_range.feature_smearing_widths.size()),
                current_inputs_pme_mesh_, current_inputs_pme_spline_order_,
                !contract_.subtract_feature_self_terms,
                contract_.long_range.include_electrostatic_self_interaction, 1,
                /*compute_features=*/false);
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string("POLAR PME composed-energy forward failed: ")
                    + error.what());
            }
            electrostatic = realspace_graph_energy_seed_device_;
            auto pme_energies = pme_energy_channel_energy_;
            Kokkos::parallel_for("polar composed PME energy stage", 1,
                KOKKOS_LAMBDA(const int) {
                    electrostatic(0) = static_cast<Precision>(pme_energies(0));
                });
#else
            throw std::logic_error("PME route compiled without particle mesh.");
#endif
        }
        auto multipoles = composed_graph_multipoles_device_;
        auto outputs = composed_graph_outputs_device_;
        auto positions = cache_.realspace_positions;
        auto node_graph = cache_.node_graph_index;
        auto external = external_potential_device_;
        auto local_energy = local_energy_;
        constexpr std::size_t graph_quantity_count = 8;
        if (graphs
            > static_cast<std::size_t>(std::numeric_limits<int>::max())
                / graph_quantity_count)
            throw std::length_error(
                "polar graph-output team count exceeds INT_MAX.");
        const int graph_output_team_count =
            static_cast<int>(graphs * graph_quantity_count);
        using GraphOutputPolicy =
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using GraphOutputMember = typename GraphOutputPolicy::member_type;
        Kokkos::parallel_for("polar composed realspace graph output reductions",
            GraphOutputPolicy(graph_output_team_count, Kokkos::AUTO),
            KOKKOS_LAMBDA(const GraphOutputMember& team) {
                const std::size_t league = team.league_rank();
                const std::size_t graph = league / graph_quantity_count;
                const int quantity = static_cast<int>(
                    league % graph_quantity_count);
                double value = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, nodes),
                    [=](const std::size_t node, double& sum) {
                        if (static_cast<std::size_t>(node_graph(node)) != graph)
                            return;
                        const double q = static_cast<double>(density(node, 0));
                        if (quantity == 0) {
                            sum += q;
                        } else if (quantity <= 3) {
                            const int component = quantity - 1;
                            const int column = component == 0 ? 3 : component;
                            sum += q * positions(node, component)
                                + (column < density_width
                                        ? static_cast<double>(
                                            density(node, column))
                                        : 0.0);
                        } else if (quantity == 5) {
                            sum += static_cast<double>(local_energy(node, 0));
                        }
                    }, value);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    if (quantity < 4)
                        multipoles(graph, quantity) = value;
                    else if (quantity == 4)
                        multipoles(graph, 4) = 0.0;
                    else
                        outputs(graph, quantity - 4) = value;
                });
            });
        Kokkos::parallel_for("polar composed realspace graph output finalize",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs_),
            KOKKOS_LAMBDA(const int graph) {
                const double local = outputs(graph, 1);
                const double electrostatic_value =
                    static_cast<double>(electrostatic(graph));
                double field = 0.0;
                for (int component = 0; component < 3; ++component)
                    field += static_cast<double>(
                        external(graph, 1 + component))
                        * multipoles(graph, 1 + component);
                outputs(graph, 1) = local;
                outputs(graph, 2) = electrostatic_value;
                outputs(graph, 3) = 0.0;
                outputs(graph, 4) = field;
                for (int component = 0; component < 3; ++component)
                    outputs(graph, 5 + component) =
                        multipoles(graph, 1 + component);
                outputs(graph, 0) = local + electrostatic_value + field;
            });
        ++graph_output_diagnostics_.selected_count;
        graph_output_diagnostics_.league_extent =
            graphs * graph_quantity_count;
        graph_output_diagnostics_.node_extent = nodes;
        graph_output_diagnostics_.k_extent = 0;
        if (materialize_outputs) {
            auto host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), outputs);
            result.energy.resize(graphs);
            result.local_electron_energy.resize(graphs);
            result.electrostatic_energy.resize(graphs);
            result.correction_energy.resize(graphs);
            result.external_field_energy.resize(graphs);
            result.dipole.resize(graphs * 3);
            for (std::size_t graph = 0; graph < graphs; ++graph) {
                result.energy[graph] = host(graph, 0);
                result.local_electron_energy[graph] = host(graph, 1);
                result.electrostatic_energy[graph] = host(graph, 2);
                result.correction_energy[graph] = host(graph, 3);
                result.external_field_energy[graph] = host(graph, 4);
                for (int component = 0; component < 3; ++component)
                    result.dipole[3 * graph + component] =
                        host(graph, 5 + component);
            }
        }
        return result;
    }

    const auto reciprocal_density = ops_.assemble_fourier_kernel(
        cache_, density, /*use_workspace=*/true);
    const auto reciprocal_potential = ops_.apply_coulomb_kernel(
        cache_, reciprocal_density, /*use_workspace=*/true);
    auto multipoles = composed_graph_multipoles_device_;
    auto outputs = composed_graph_outputs_device_;
    auto positions = cache_.positions_workspace;
    auto node_graph = cache_.node_graph_index;
    auto k_graph = cache_.k_graph_index;
    auto volumes = cache_.kernel_volumes;
    auto pbc = graph_pbc_device_;
    auto external = external_potential_device_;
    auto local_energy = local_energy_;
    auto self_constants = si_energy_constants_device_;
    auto self_select = si_energy_select_device_;
    const std::size_t n_k = cache_.n_k;
    const bool subtract_self =
        !contract_.long_range.include_electrostatic_self_interaction;
    constexpr double pi = 3.14159265358979323846;
    constexpr double two_pi_cubed =
        (2.0 * pi) * (2.0 * pi) * (2.0 * pi);
    constexpr double energy_scale =
        0.5 / (two_pi_cubed * two_pi_cubed);
    constexpr double constant = (1.0 / 5.526349406e-3) / (4.0 * pi);
    constexpr double madelung = 2.837297;
    constexpr std::size_t graph_quantity_count = 8;
    if (graphs
        > static_cast<std::size_t>(std::numeric_limits<int>::max())
            / graph_quantity_count)
        throw std::length_error(
            "polar graph-output team count exceeds INT_MAX.");
    const int graph_output_team_count =
        static_cast<int>(graphs * graph_quantity_count);
    using GraphOutputPolicy =
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using GraphOutputMember = typename GraphOutputPolicy::member_type;
    Kokkos::parallel_for("polar composed graph output reductions",
        GraphOutputPolicy(graph_output_team_count, Kokkos::AUTO),
        KOKKOS_LAMBDA(const GraphOutputMember& team) {
            const std::size_t league = team.league_rank();
            const std::size_t graph = league / graph_quantity_count;
            const int quantity = static_cast<int>(
                league % graph_quantity_count);
            const std::size_t extent = quantity == 7 ? n_k : nodes;
            double value = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, extent),
                [=](const std::size_t index, double& sum) {
                    if (quantity == 7) {
                        if (static_cast<std::size_t>(k_graph(index)) == graph)
                            sum += 2.0
                                * (double(reciprocal_density(index, 0))
                                        * double(reciprocal_potential(index, 0))
                                    + double(reciprocal_density(index, 1))
                                        * double(reciprocal_potential(index, 1)));
                        return;
                    }
                    if (static_cast<std::size_t>(node_graph(index)) != graph)
                        return;
                    const double q = density(index, 0);
                    if (quantity == 0) {
                        sum += q;
                    } else if (quantity <= 3) {
                        const int component = quantity - 1;
                        const int column = component == 0 ? 3 : component;
                        sum += q * double(positions(index, component))
                            + (column < density_width
                                    ? double(density(index, column)) : 0.0);
                    } else if (quantity == 4) {
                        double r2 = 0.0, p_dot_r = 0.0;
                        for (int component = 0; component < 3; ++component) {
                            const double r = positions(index, component);
                            const int column = component == 0 ? 3 : component;
                            const double local_dipole = column < density_width
                                ? double(density(index, column)) : 0.0;
                            r2 += r * r;
                            p_dot_r += r * local_dipole;
                        }
                        sum += q * r2 + 2.0 * p_dot_r;
                    } else if (quantity == 5) {
                        sum += double(local_energy(index, 0));
                    } else if (subtract_self) {
                        for (std::size_t term = 0;
                            term < self_constants.extent(0); ++term)
                            sum += double(density(index, term))
                                * double(self_constants(term))
                                * double(density(index, self_select(term)));
                    }
                }, value);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                if (quantity < 5)
                    multipoles(graph, quantity) = value;
                else
                    outputs(graph, quantity - 4) = value;
            });
        });
    Kokkos::parallel_for("polar composed graph output finalize",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs_),
        KOKKOS_LAMBDA(const int graph) {
            const double charge = multipoles(graph, 0);
            const double dipole[3] = {multipoles(graph, 1),
                multipoles(graph, 2), multipoles(graph, 3)};
            const double quadrupole = multipoles(graph, 4);
            const double local = outputs(graph, 1);
            const double self_energy = outputs(graph, 2);
            double reciprocal = outputs(graph, 3)
                * double(volumes(graph)) * energy_scale;
            if (subtract_self) reciprocal -= 0.5 * self_energy;
            const double volume = volumes(graph);
            double correction = 0.0;
            const bool molecule = pbc(graph, 0) == 0
                && pbc(graph, 1) == 0 && pbc(graph, 2) == 0;
            const bool slab = pbc(graph, 0) != 0
                && pbc(graph, 1) != 0 && pbc(graph, 2) == 0;
            if (molecule) {
                const double dipole2 = dipole[0] * dipole[0]
                    + dipole[1] * dipole[1] + dipole[2] * dipole[2];
                correction = 0.5 * madelung * constant * charge * charge
                        / Kokkos::pow(volume, 0.3333)
                    + 2.0 * constant * pi * dipole2 / (3.0 * volume)
                    - 2.0 * constant * pi * charge * quadrupole
                        / (3.0 * volume);
            } else if (slab) {
                correction = constant * 2.0 * pi * dipole[2] * dipole[2]
                    / volume;
            }
            double field = 0.0;
            for (int component = 0; component < 3; ++component)
                field += double(external(graph, 1 + component))
                    * dipole[component];
            outputs(graph, 1) = local;
            outputs(graph, 2) = reciprocal;
            outputs(graph, 3) = correction;
            outputs(graph, 4) = field;
            for (int component = 0; component < 3; ++component)
                outputs(graph, 5 + component) = dipole[component];
            outputs(graph, 0) = local + reciprocal + correction + field;
        });
    ++graph_output_diagnostics_.selected_count;
    graph_output_diagnostics_.league_extent =
        graphs * graph_quantity_count;
    graph_output_diagnostics_.node_extent = nodes;
    graph_output_diagnostics_.k_extent = n_k;
    if (materialize_outputs) {
        auto host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), outputs);
        result.energy.resize(graphs);
        result.local_electron_energy.resize(graphs);
        result.electrostatic_energy.resize(graphs);
        result.correction_energy.resize(graphs);
        result.external_field_energy.resize(graphs);
        result.dipole.resize(graphs * 3);
        for (std::size_t graph = 0; graph < graphs; ++graph) {
            result.energy[graph] = host(graph, 0);
            result.local_electron_energy[graph] = host(graph, 1);
            result.electrostatic_energy[graph] = host(graph, 2);
            result.correction_energy[graph] = host(graph, 3);
            result.external_field_energy[graph] = host(graph, 4);
            for (int component = 0; component < 3; ++component)
                result.dipole[3 * graph + component] =
                    host(graph, 5 + component);
        }
    }
    return result;
}


// ---------------------------------------------------------------------------
// The reverse pass
// ---------------------------------------------------------------------------
//
// The chain's own derivative surface.  Every delta block is node-local, so no
// block contributes a position or cell adjoint: the position seeds come from
// the long-range kernels the chain drives (the electrostatic projection chain,
// the analytic correction fields, and the external-field gauge).  Fully
// non-periodic real space has an exact-zero fixed-Cartesian cell adjoint;
// periodic reciprocal geometry and corrections own the cell-dependent
// adjoints that the whole-model result collector composes into stress.  The
// pass follows the forward's own order in reverse -- the post-loop readout,
// then the unrolled updates with their equilibrations, then the pre-loop
// equilibration, the heads, the mixer, and the source map -- and the shared
// cache's table seeds are summed before the geometry adjoint fires, exactly as
// the design note's shared-step rule states.
template<typename Precision>
typename PolarDeltaChainKokkosT<Precision>::Adjoints
PolarDeltaChainKokkosT<Precision>::reverse(
    const Inputs& inputs, const Seeds& seeds,
    const bool materialize_table_adjoints,
    const execution::long_range::LongRangeKernelMatrix<Precision>*
        initial_grad_cos_table,
    const execution::long_range::LongRangeKernelMatrix<Precision>*
        initial_grad_sin_table,
    const bool materialize_learned_adjoints,
    const bool use_resident_output_seeds,
    const bool use_reference_potential_adjoint,
    const bool materialize_input_adjoints)
{
    if (!ready_ || density_steps_.empty())
        throw std::invalid_argument(
            "the polar delta chain's reverse pass requires a forward pass.");
    ++resident_reverse_count_;
    const int nodes = nodes_, graphs = graphs_, steps = this->steps();
    const int layers = contract_.interaction_count;
    const int width = contract_.node_feature_dimension;
    const std::size_t charge_count =
        static_cast<std::size_t>(nodes) * charges_dimension_;
    const std::size_t potential_count =
        static_cast<std::size_t>(nodes) * potential_dimension_;
    const std::size_t fukui_count = static_cast<std::size_t>(nodes) * 2;
    const auto stage_optional = [] (const auto& target,
        const std::vector<double>& supplied, const std::size_t expected,
        const char* what) {
        Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace{}, target, Precision(0));
        if (supplied.empty()) return;
        if (supplied.size() != expected)
            throw std::invalid_argument(std::string("the polar chain's ")
                + what + " seed does not match its state.");
        chain_store_values<Precision>(target, supplied, what);
    };
    for (int step = 0; step <= steps; ++step) {
        if (!(use_resident_output_seeds && step == steps))
            stage_optional(density_seed_steps_[static_cast<std::size_t>(step)],
                step == steps ? seeds.spin_charge_density
                    : std::vector<double>{},
                charge_count, "density");
        stage_optional(fukui_seed_steps_[static_cast<std::size_t>(step)],
            step < static_cast<int>(seeds.fukui_sources.size())
                ? seeds.fukui_sources[static_cast<std::size_t>(step)]
                : std::vector<double>{}, fukui_count, "Fukui-source");
    }
    for (int step = 0; step < steps; ++step)
        stage_optional(potential_seed_steps_[static_cast<std::size_t>(step)],
            step < static_cast<int>(seeds.potential_features.size())
                ? seeds.potential_features[static_cast<std::size_t>(step)]
                : std::vector<double>{}, potential_count, "potential-feature");

    if ((initial_grad_cos_table == nullptr) !=
        (initial_grad_sin_table == nullptr))
        throw std::invalid_argument(
            "the polar chain requires both initial device table seeds.");
    auto grad_cos = grad_cos_workspace_;
    auto grad_sin = grad_sin_workspace_;
    if (initial_grad_cos_table != nullptr) {
        grad_cos = *initial_grad_cos_table;
        grad_sin = *initial_grad_sin_table;
    } else {
        Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace{}, grad_cos, Precision(0));
        Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace{}, grad_sin, Precision(0));
        if (!seeds.table_seed_cos.empty())
            chain_store_long_range_values(grad_cos, seeds.table_seed_cos,
                "cos-table");
        if (!seeds.table_seed_sin.empty())
            chain_store_long_range_values(grad_sin, seeds.table_seed_sin,
                "sin-table");
    }
    auto grad_k = grad_k_workspace_;
    auto grad_volume = grad_volume_workspace_;
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, grad_k, Precision(0));
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, grad_volume, Precision(0));
    if (!use_resident_output_seeds) {
        if (solver_decision_.resolved
            == execution::long_range::LongRangeSolver::realspace)
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
                realspace_position_seed_device_, 0.0);
        else
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
                position_seed_device_, Precision(0));
    }
    if (!use_resident_output_seeds)
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            external_seed_device_, Precision(0));
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        target_seed_device_, Precision(0));
    if (use_resident_output_seeds
        && solver_decision_.resolved
            == execution::long_range::LongRangeSolver::realspace) {
        auto target_seed = target_seed_device_;
        auto external = external_potential_device_;
        auto barycenter = realspace_barycenter_device_;
        Kokkos::parallel_for("polar realspace field charge compensation",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs),
            KOKKOS_LAMBDA(const int graph) {
                double common_seed = 0.0;
                for (int component = 0; component < 3; ++component)
                    common_seed += static_cast<double>(
                        external(graph, 1 + component))
                        * barycenter(graph, component);
                target_seed(graph, 0) =
                    static_cast<Precision>(common_seed);
            });
    }
    if (!use_resident_output_seeds)
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            correction_strain_device_, Precision(0));
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        node_feature_seed_, Precision(0));
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, attrs_seed_, Precision(0));
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, mixed_seed_, Precision(0));

    const auto layer_features = [this, width] (const int layer) {
        return Kokkos::subview(node_features_, Kokkos::ALL,
            std::make_pair(layer * width, (layer + 1) * width));
    };
    auto node_feature_seed = node_feature_seed_;
    const auto layer_seed = [&node_feature_seed, width] (const int layer) {
        return Kokkos::subview(node_feature_seed, Kokkos::ALL,
            std::make_pair(layer * width, (layer + 1) * width));
    };

    auto energy_seed = energy_seed_;
    if (use_resident_output_seeds)
        Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace{}, energy_seed, Precision(1));
    else
        stage_optional(energy_seed, seeds.local_electron_energy,
            static_cast<std::size_t>(nodes), "local-electron-energy");
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        pre_loop_seed_, Precision(0));
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        charge_seed_, Precision(0));
    local_electron_readout_.reverse(energy_seed, layer_features(layers - 1),
        potential_features_steps_.back(), density_steps_[0],
        density_steps_.back(), layer_seed(layers - 1),
        potential_seed_steps_.back(), pre_loop_seed_, charge_seed_, false);
    {
        auto destination = density_seed_steps_.back();
        auto source = charge_seed_;
        Kokkos::parallel_for("polar chain final density seed",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                for (int column = 0; column < destination.extent_int(1); ++column)
                    destination(node, column) += source(node, column);
            });
    }

    for (int step = steps; step >= 1; --step) {
        const std::size_t index = static_cast<std::size_t>(step);
        auto working = density_seed_steps_[index];
        auto source_seed = fukui_seed_steps_[index];
        auto unequilibrated = charge_seed_;
        auto previous = density_steps_[index - 1];
        auto update_source = source_steps_[index];
        Kokkos::parallel_for("polar chain replay unequilibrated state",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                for (int column = 0; column < working.extent_int(1); ++column)
                    unequilibrated(node, column) = previous(node, column)
                        + update_source(node, column);
            });
        adjoint_equilibrate_device(fukui_steps_[index], unequilibrated,
            working, source_seed, target_seed_device_);
        adjoint_normalize_fukui_device(fukui_raw_steps_[index], source_seed);

        field_updates_[index - 1].evaluate(features_mixed_,
            potential_features_steps_[index - 1], node_attrs_, previous,
            update_output_);
        auto update_seed = update_seed_;
        Kokkos::parallel_for("polar chain pack update seed",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                for (int column = 0; column < working.extent_int(1); ++column)
                    update_seed(node, column) = working(node, column);
                update_seed(node, working.extent_int(1)) = source_seed(node, 0);
                update_seed(node, working.extent_int(1) + 1) = source_seed(node, 1);
            });
        Kokkos::deep_copy(
            Kokkos::DefaultExecutionSpace{}, row_seed_, Precision(0));
        field_updates_[index - 1].reverse(update_seed, features_mixed_,
            potential_features_steps_[index - 1], node_attrs_, previous,
            mixed_seed_, potential_seed_steps_[index - 1], attrs_seed_,
            row_seed_, false);
        auto previous_seed = density_seed_steps_[index - 1];
        auto local_seed = row_seed_;
        Kokkos::parallel_for("polar chain update identity adjoint",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                for (int column = 0; column < working.extent_int(1); ++column)
                    previous_seed(node, column) += working(node, column)
                        + local_seed(node, column);
            });
        adjoint_potential_features_device(previous,
            potential_seed_steps_[index - 1], previous_seed,
            position_seed_device_, external_seed_device_,
            correction_strain_device_, grad_cos, grad_sin, grad_k,
            grad_volume, use_reference_potential_adjoint);
    }

    auto heads_seed = density_seed_steps_[0];
    {
        auto pre = pre_loop_seed_;
        Kokkos::parallel_for("polar chain pre-loop seed merge",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                for (int column = 0; column < heads_seed.extent_int(1); ++column)
                    heads_seed(node, column) += pre(node, column);
            });
    }
    auto initial_fukui_seed = fukui_seed_steps_[0];
    adjoint_equilibrate_device(fukui_steps_[0], heads_density_device_,
        heads_seed, initial_fukui_seed, target_seed_device_);
    adjoint_normalize_fukui_device(fukui_raw_steps_[0], initial_fukui_seed);
    for (int layer = 0; layer < layers; ++layer) {
        if (spin_source_zero_charges_[static_cast<std::size_t>(layer)]) {
            auto seed = heads_seed;
            Kokkos::parallel_for("polar chain zero charge adjoint",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
                KOKKOS_LAMBDA(const int node) { seed(node, 0) = Precision(0); });
        }
        spin_source_heads_[static_cast<std::size_t>(layer)].reverse(
            heads_seed, layer_seed(layer), false);
        feature_mixer_[static_cast<std::size_t>(layer)].reverse(
            mixed_seed_, layer_seed(layer), false);
    }
    fukui_source_map_.reverse(initial_fukui_seed, fukui_input_,
        layer_seed(layers - 1), false);

    constexpr std::size_t stress_component_count = 9;
    if (static_cast<std::size_t>(graphs)
        > static_cast<std::size_t>(std::numeric_limits<int>::max())
            / stress_component_count)
        throw std::length_error(
            "polar resident stress team count exceeds INT_MAX.");
    const int stress_team_count = static_cast<int>(
        static_cast<std::size_t>(graphs) * stress_component_count);
    using StressPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using StressMember = typename StressPolicy::member_type;
    execution::long_range::LongRangeKernelTensor3<Precision> geometry_cell;
    if (solver_decision_.resolved
        == execution::long_range::LongRangeSolver::realspace) {
        auto position_seed = realspace_position_seed_device_;
        auto public_position_seed = public_position_seed_device_;
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            public_position_seed, position_seed);
        auto stress = composed_stress_device_;
        auto positions = cache_.realspace_positions;
        auto node_graph = cache_.node_graph_index;
        auto volumes = reporting_volumes_device_;
        Kokkos::parallel_for("polar composed realspace resident stress",
            StressPolicy(stress_team_count, Kokkos::AUTO),
            KOKKOS_LAMBDA(const StressMember& team) {
                const int index = team.league_rank();
                const int graph = index / 9;
                const int row = (index % 9) / 3;
                const int column = index % 3;
                double virial = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, nodes),
                    [=](const int node, double& sum) {
                        if (node_graph(node) == graph)
                            sum += positions(node, row)
                                * position_seed(node, column);
                    }, virial);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    const double volume = volumes(graph);
                    stress(graph, 3 * row + column) =
                        volume == 0.0 ? 0.0 : virial / volume;
                });
            });
    } else if (solver_decision_.resolved
        == execution::long_range::LongRangeSolver::pme) {
        auto position_seed = position_seed_device_;
        auto public_position_seed = public_position_seed_device_;
        Kokkos::parallel_for("polar chain publish PME position adjoint",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                for (int component = 0; component < 3; ++component)
                    public_position_seed(node, component) =
                        static_cast<double>(position_seed(node, component));
            });
        auto correction_strain = correction_strain_device_;
        auto stress = composed_stress_device_;
        auto positions = cache_.realspace_positions;
        auto node_graph = cache_.node_graph_index;
        auto volumes = reporting_volumes_device_;
        Kokkos::parallel_for("polar composed PME resident stress",
            StressPolicy(stress_team_count, Kokkos::AUTO),
            KOKKOS_LAMBDA(const StressMember& team) {
                const int index = team.league_rank();
                const int graph = index / 9;
                const int row = (index % 9) / 3;
                const int column = index % 3;
                double virial = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, nodes),
                    [=](const int node, double& sum) {
                        if (node_graph(node) == graph)
                            sum += positions(node, row)
                                * static_cast<double>(
                                    position_seed(node, column));
                    }, virial);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    const double volume = volumes(graph);
                    stress(graph, 3 * row + column) = volume == 0.0 ? 0.0
                        : (static_cast<double>(correction_strain(
                                graph, 3 * row + column))
                            + virial)
                            / volume;
                });
            });
    } else {
        auto geometry = ops_.adjoint_geometry_device(
            cache_, grad_cos, grad_sin);
        auto position_seed = position_seed_device_;
        auto public_position_seed = public_position_seed_device_;
        Kokkos::parallel_for("polar chain combine position adjoint",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes),
            KOKKOS_LAMBDA(const int node) {
                for (int component = 0; component < 3; ++component) {
                    position_seed(node, component) +=
                        static_cast<Precision>(geometry(node, component));
                    public_position_seed(node, component) =
                        static_cast<double>(position_seed(node, component));
                }
            });
        geometry_cell = ops_.adjoint_geometry_cell_all_device(cache_, grad_k,
            grad_volume, grad_cos, grad_sin,
            execution::long_range::LongRangeConstMatrix<double>(
                positions_.data(), nodes, 3), true);
        auto correction_strain = correction_strain_device_;
        auto stress = composed_stress_device_;
        auto positions = cache_.positions_workspace;
        auto node_graph = cache_.node_graph_index;
        auto volumes = cache_.kernel_volumes;
        Kokkos::parallel_for("polar composed resident stress",
            StressPolicy(stress_team_count, Kokkos::AUTO),
            KOKKOS_LAMBDA(const StressMember& team) {
                const int index = team.league_rank();
                const int graph = index / 9;
                const int row = (index % 9) / 3;
                const int column = index % 3;
                double virial = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, nodes),
                    [=](const int node, double& sum) {
                        if (node_graph(node) == graph)
                            sum += double(positions(node, row))
                                * double(position_seed(node, column));
                    }, virial);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    stress(graph, 3 * row + column) =
                        (geometry_cell(graph, row, column)
                            + double(correction_strain(
                                graph, 3 * row + column))
                            + virial)
                        / double(volumes(graph));
                });
            });
    }

    Adjoints result;
    if (materialize_learned_adjoints) {
        result.node_features = to_double(node_feature_seed_);
        result.node_attrs = to_double(attrs_seed_);
    }
    if (materialize_input_adjoints) {
        const auto position_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), public_position_seed_device_);
        result.positions.assign(position_host.data(),
            position_host.data() + position_host.size());
        result.external_potential = to_double(external_seed_device_);
        const auto targets = to_double(target_seed_device_);
        result.total_charge.resize(static_cast<std::size_t>(graphs));
        result.total_spin.resize(static_cast<std::size_t>(graphs));
        for (int graph = 0; graph < graphs; ++graph) {
            result.total_charge[static_cast<std::size_t>(graph)] =
                targets[static_cast<std::size_t>(graph) * 2];
            result.total_spin[static_cast<std::size_t>(graph)] =
                targets[static_cast<std::size_t>(graph) * 2 + 1];
        }
        result.strain.resize(static_cast<std::size_t>(graphs) * 9);
        if (solver_decision_.resolved
            == execution::long_range::LongRangeSolver::realspace) {
            std::fill(result.strain.begin(), result.strain.end(), 0.0);
        } else if (solver_decision_.resolved
            == execution::long_range::LongRangeSolver::pme) {
            const auto correction_strain_host =
                Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), correction_strain_device_);
            for (int graph = 0; graph < graphs; ++graph)
                for (int component = 0; component < 9; ++component)
                    result.strain[
                        static_cast<std::size_t>(graph) * 9 + component]
                        = double(correction_strain_host(graph, component));
        } else {
            const auto geometry_cell_host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), geometry_cell);
            const auto correction_strain_host =
                Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), correction_strain_device_);
            for (int graph = 0; graph < graphs; ++graph)
                for (int component = 0; component < 9; ++component)
                    result.strain[
                        static_cast<std::size_t>(graph) * 9 + component]
                        = geometry_cell_host(
                            graph, component / 3, component % 3)
                        + double(correction_strain_host(graph, component));
        }
    }
    if (materialize_table_adjoints) {
        result.grad_cos_table = chain_long_range_values<Precision>(grad_cos);
        result.grad_sin_table = chain_long_range_values<Precision>(grad_sin);
    }
    (void)inputs;
    return result;
}

template<typename Precision>
typename PolarDeltaChainKokkosT<Precision>::ComposedReverseResult
PolarDeltaChainKokkosT<Precision>::reverse_composed_energy(
    const Inputs& inputs, const State& state,
    const bool materialize_learned_adjoints,
    const bool materialize_input_adjoints)
{
    if (state.nodes != static_cast<std::size_t>(nodes_)
        || state.graphs != static_cast<std::size_t>(graphs_))
        throw std::invalid_argument(
            "the polar composed reverse requires the state from the current "
            "forward pass.");
    const std::size_t nodes = state.nodes;
    const std::size_t graphs = state.graphs;
    if (inputs.positions.size() != nodes * 3
        || inputs.cell.size() != graphs * 9
        || inputs.graph_pbc.size() != graphs * 3
        || inputs.external_potential.size() != graphs * 4)
        throw std::invalid_argument(
            "the polar composed reverse inputs do not match the forward "
            "state's dimensions.");

#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
    if (solver_decision_.resolved
        == execution::long_range::LongRangeSolver::pme) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            pme_energy_feature_seeds_, Precision(0));
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            pme_energy_seed_, 1.0);
        std::array<double, 9> cell{};
        std::copy_n(cell_.data(), cell.size(), cell.begin());
        const std::array<int, 3> pbc{graph_pbc_[0], graph_pbc_[1],
            graph_pbc_[2]};
        particle_mesh_workspace_->reverse_channels_device(
            typename PolarParticleMeshWorkspaceT<Precision>::
                const_device_double_matrix(cache_.realspace_positions),
            typename PolarParticleMeshWorkspaceT<Precision>::const_device_matrix(
                composed_density_device_),
            typename PolarParticleMeshWorkspaceT<Precision>::const_device_matrix(
                pme_energy_feature_seeds_),
            typename PolarParticleMeshWorkspaceT<Precision>::const_device_vector(
                pme_energy_seed_),
            pme_energy_features_, pme_energy_channel_energy_,
            pme_energy_source_adjoints_, pme_position_adjoints_,
            pme_cell_adjoints_, cell, pbc,
            contract_.long_range.density_smearing_width,
            std::span<const double>(
                contract_.long_range.feature_smearing_widths.data(),
                contract_.long_range.feature_smearing_widths.size()),
            current_inputs_pme_mesh_, current_inputs_pme_spline_order_,
            !contract_.subtract_feature_self_terms,
            contract_.long_range.include_electrostatic_self_interaction, 1,
            false, true);

        auto final_seed = density_seed_steps_.back();
        auto density = composed_density_device_;
        auto stage_source = pme_energy_source_adjoints_;
        auto stage_position = pme_position_adjoints_;
        auto position_seed = position_seed_device_;
        auto positions = cache_.realspace_positions;
        auto node_graph = cache_.node_graph_index;
        auto external = external_potential_device_;
        auto external_seed = external_seed_device_;
        auto cell_seed = correction_strain_device_;
        const int density_width = density_width_;
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            external_seed, Precision(0));
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            cell_seed, Precision(0));
        Kokkos::parallel_for("polar composed PME resident output seeds",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
            KOKKOS_LAMBDA(const int node) {
                const int graph = node_graph(node);
                const double q = static_cast<double>(density(node, 0));
                double field_displacement_seed = 0.0;
                for (int component = 0; component < 3; ++component)
                    field_displacement_seed += static_cast<double>(
                        external(graph, 1 + component))
                        * positions(node, component);
                for (int column = 0; column < density_width; ++column) {
                    Precision value = stage_source(node, column);
                    if (column == 0) {
                        value += static_cast<Precision>(field_displacement_seed);
                    } else {
                        const int component = column == 3
                            ? 0 : (column == 1 ? 1 : (column == 2 ? 2 : -1));
                        if (component >= 0)
                            value += external(graph, 1 + component);
                    }
                    final_seed(node, column) = value;
                    final_seed(node, density_width + column) = value;
                }
                for (int component = 0; component < 3; ++component)
                    position_seed(node, component) =
                        static_cast<Precision>(stage_position(node, component)
                            + q * static_cast<double>(
                                external(graph, 1 + component)));
            });
        auto aggregates = composed_graph_multipoles_device_;
        Kokkos::parallel_for("polar composed PME direct field seed",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs_),
            KOKKOS_LAMBDA(const int graph) {
                for (int component = 0; component < 3; ++component)
                    external_seed(graph, 1 + component) =
                        static_cast<Precision>(
                            aggregates(graph, 1 + component));
            });
        auto pme_cell = pme_cell_adjoints_;
        using CellPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using CellMember = typename CellPolicy::member_type;
        const int node_count = nodes_;
        Kokkos::parallel_for("polar composed PME fixed-Cartesian cell seed",
            CellPolicy(9, Kokkos::AUTO),
            KOKKOS_LAMBDA(const CellMember& team) {
                const int index = team.league_rank();
                const int row = index / 3;
                const int column = index % 3;
                double virial = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, node_count),
                    [=](const int node, double& sum) {
                        sum += positions(node, row)
                            * stage_position(node, column);
                    }, virial);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    cell_seed(0, index) = static_cast<Precision>(
                        pme_cell(index) - virial);
                });
            });

        ComposedReverseResult composed;
        composed.adjoints = reverse(inputs, Seeds{},
            /*materialize_table_adjoints=*/false,
            /*initial_grad_cos_table=*/nullptr,
            /*initial_grad_sin_table=*/nullptr,
            materialize_learned_adjoints,
            /*use_resident_output_seeds=*/true,
            /*use_reference_potential_adjoint=*/false,
            materialize_input_adjoints);
        if (materialize_input_adjoints) {
            const auto host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), composed_stress_device_);
            composed.stress.resize(graphs * 9);
            for (std::size_t graph = 0; graph < graphs; ++graph)
                for (int component = 0; component < 9; ++component)
                    composed.stress[9 * graph + component] =
                        host(graph, component);
        }
        return composed;
    }
#endif

    if (solver_decision_.resolved
        == execution::long_range::LongRangeSolver::realspace) {
        auto graph_energy_seed = realspace_graph_energy_seed_device_;
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            graph_energy_seed, Precision(1));
        const auto electrostatic = ops_.adjoint_realspace_energy_kernel(
            cache_, composed_density_device_, graph_energy_seed,
            contract_.long_range.include_electrostatic_self_interaction,
            /*use_workspace=*/true);
        auto final_seed = density_seed_steps_.back();
        auto density = composed_density_device_;
        auto positions = cache_.realspace_positions;
        auto barycenter = realspace_barycenter_device_;
        auto node_graph = cache_.node_graph_index;
        auto stage_multipole_seed = electrostatic.grad_multipoles;
        auto stage_position_seed = electrostatic.grad_positions;
        auto position_seed = realspace_position_seed_device_;
        auto external_seed = external_seed_device_;
        auto external = external_potential_device_;
        auto cell_seed = correction_strain_device_;
        const int density_width = density_width_;
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            external_seed, Precision(0));
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
            cell_seed, Precision(0));
        Kokkos::parallel_for("polar composed realspace resident output seeds",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
            KOKKOS_LAMBDA(const int node) {
                const int graph = node_graph(node);
                const double q = static_cast<double>(density(node, 0));
                // Charge normalization annihilates a graph-common seed.  Remove
                // it in fp64 before narrowing the learned adjoint to Precision.
                double field_displacement_seed = 0.0;
                for (int component = 0; component < 3; ++component)
                    field_displacement_seed += static_cast<double>(
                        external(graph, 1 + component))
                        * (positions(node, component)
                            - barycenter(graph, component));
                for (int column = 0; column < density_width; ++column) {
                    Precision value = stage_multipole_seed(node, column);
                    if (column == 0) {
                        value += static_cast<Precision>(
                            field_displacement_seed);
                    } else {
                        const int component = column == 3
                            ? 0 : (column == 1 ? 1 : (column == 2 ? 2 : -1));
                        if (component >= 0)
                            value += external(graph, 1 + component);
                    }
                    final_seed(node, column) = value;
                    final_seed(node, density_width + column) = value;
                }
                for (int component = 0; component < 3; ++component)
                    position_seed(node, component) =
                        stage_position_seed(node, component)
                        + q * static_cast<double>(
                            external(graph, 1 + component));
            });
        auto aggregates = composed_graph_multipoles_device_;
        Kokkos::parallel_for("polar composed realspace direct field seed",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs_),
            KOKKOS_LAMBDA(const int graph) {
                for (int component = 0; component < 3; ++component)
                    external_seed(graph, 1 + component) =
                        static_cast<Precision>(
                            aggregates(graph, 1 + component));
            });

        ComposedReverseResult composed;
        composed.adjoints = reverse(inputs, Seeds{},
            /*materialize_table_adjoints=*/false,
            /*initial_grad_cos_table=*/nullptr,
            /*initial_grad_sin_table=*/nullptr,
            materialize_learned_adjoints,
            /*use_resident_output_seeds=*/true,
            /*use_reference_potential_adjoint=*/false,
            materialize_input_adjoints);
        if (materialize_input_adjoints) {
            const auto host = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), composed_stress_device_);
            composed.stress.resize(graphs * 9);
            for (std::size_t graph = 0; graph < graphs; ++graph)
                for (int component = 0; component < 9; ++component)
                    composed.stress[9 * graph + component] =
                        host(graph, component);
        }
        return composed;
    }

    auto grad_cos_table = grad_cos_workspace_;
    auto grad_sin_table = grad_sin_workspace_;
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, grad_cos_table, 0.0);
    Kokkos::deep_copy(
        Kokkos::DefaultExecutionSpace{}, grad_sin_table, 0.0);
    auto reciprocal_seed = ops_.adjoint_coulomb_energy_composed_device(
        cache_, composed_density_device_,
        !contract_.long_range.include_electrostatic_self_interaction,
        si_energy_constants_device_, si_energy_select_device_,
        grad_cos_table, grad_sin_table, composed_energy_strain_device_);

    auto aggregates = composed_graph_multipoles_device_;
    auto graph_seeds = composed_graph_seeds_device_;
    auto pbc = graph_pbc_device_;
    auto volumes = cache_.kernel_volumes;
    auto external = external_potential_device_;
    auto cell_seed = correction_strain_device_;
    auto energy_strain = composed_energy_strain_device_;
    constexpr double pi = 3.14159265358979323846;
    constexpr double constant = (1.0 / 5.526349406e-3) / (4.0 * pi);
    constexpr double madelung = 2.837297;
    Kokkos::parallel_for("polar composed correction graph adjoint",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs_),
        KOKKOS_LAMBDA(const int graph) {
            const double volume = volumes(graph);
            const double charge = aggregates(graph, 0);
            const double quadrupole = aggregates(graph, 4);
            const bool molecule = pbc(graph, 0) == 0
                && pbc(graph, 1) == 0 && pbc(graph, 2) == 0;
            const bool slab = pbc(graph, 0) != 0
                && pbc(graph, 1) != 0 && pbc(graph, 2) == 0;
            double seed_charge = 0.0, seed_quadrupole = 0.0;
            double seed_dipole[3] = {0.0, 0.0, 0.0};
            double d_energy_d_volume = 0.0;
            if (molecule) {
                const double box = Kokkos::pow(volume, 0.3333);
                double dipole2 = 0.0;
                for (int d = 0; d < 3; ++d)
                    dipole2 += aggregates(graph, 1 + d)
                        * aggregates(graph, 1 + d);
                seed_charge = madelung * constant * charge / box
                    - 2.0 * constant * pi * quadrupole / (3.0 * volume);
                for (int d = 0; d < 3; ++d)
                    seed_dipole[d] = 4.0 * constant * pi
                        * aggregates(graph, 1 + d) / (3.0 * volume);
                seed_quadrupole =
                    -2.0 * constant * pi * charge / (3.0 * volume);
                d_energy_d_volume = -0.3333 * 0.5 * madelung * constant
                        * charge * charge
                        * Kokkos::pow(volume, -1.3333)
                    - 2.0 * constant * pi * dipole2
                        / (3.0 * volume * volume)
                    + 2.0 * constant * pi * charge * quadrupole
                        / (3.0 * volume * volume);
            } else if (slab) {
                seed_dipole[2] = 4.0 * constant * pi
                    * aggregates(graph, 3) / volume;
                d_energy_d_volume = -constant * 2.0 * pi
                    * aggregates(graph, 3) * aggregates(graph, 3)
                    / (volume * volume);
            }
            graph_seeds(graph, 0) = seed_charge;
            for (int d = 0; d < 3; ++d)
                graph_seeds(graph, 1 + d) = seed_dipole[d];
            graph_seeds(graph, 4) = seed_quadrupole;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    cell_seed(graph, 3 * a + b) = static_cast<Precision>(
                        energy_strain(graph, a, b)
                        + (a == b ? d_energy_d_volume * volume : 0.0));
        });

    auto final_seed = density_seed_steps_.back();
    auto density = composed_density_device_;
    auto positions = cache_.positions_workspace;
    auto node_graph = cache_.node_graph_index;
    auto position_seed = position_seed_device_;
    auto external_seed = external_seed_device_;
    const int density_width = density_width_;
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},
        external_seed, Precision(0));
    Kokkos::parallel_for("polar composed resident output seeds",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nodes_),
        KOKKOS_LAMBDA(const int node) {
            const int graph = node_graph(node);
            const double q = density(node, 0);
            const double local_dipole[3] = {
                density_width > 3 ? double(density(node, 3)) : 0.0,
                density_width > 1 ? double(density(node, 1)) : 0.0,
                density_width > 2 ? double(density(node, 2)) : 0.0};
            double r2 = 0.0, displacement_seed = 0.0;
            for (int d = 0; d < 3; ++d) {
                r2 += double(positions(node, d)) * positions(node, d);
                displacement_seed += graph_seeds(graph, 1 + d)
                    * positions(node, d);
            }
            Precision combined[4] = {Precision(0), Precision(0),
                Precision(0), Precision(0)};
            combined[0] = reciprocal_seed(node, 0)
                + static_cast<Precision>(graph_seeds(graph, 0)
                    + displacement_seed + graph_seeds(graph, 4) * r2);
            double field_displacement_seed = 0.0;
            for (int d = 0; d < 3; ++d)
                field_displacement_seed += double(external(graph, 1 + d))
                    * positions(node, d);
            combined[0] += static_cast<Precision>(field_displacement_seed);
            for (int d = 0; d < 3; ++d) {
                const int column = d == 0 ? 3 : (d == 1 ? 1 : 2);
                if (column < density_width)
                    combined[column] = reciprocal_seed(node, column)
                        + static_cast<Precision>(graph_seeds(graph, 1 + d)
                            + 2.0 * graph_seeds(graph, 4)
                                * positions(node, d)
                            + double(external(graph, 1 + d)));
                position_seed(node, d) = static_cast<Precision>(
                    graph_seeds(graph, 1 + d) * q
                    + graph_seeds(graph, 4)
                        * (2.0 * q * positions(node, d)
                            + 2.0 * local_dipole[d])
                    + q * double(external(graph, 1 + d)));
            }
            for (int column = 0; column < density_width; ++column) {
                const Precision value = column < 4
                    ? combined[column] : reciprocal_seed(node, column);
                final_seed(node, column) = value;
                final_seed(node, density_width + column) = value;
            }
        });
    // The direct derivative of E_field dot dipole with respect to the public
    // graph field is the already resident graph dipole.
    Kokkos::parallel_for("polar composed direct field seed",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, graphs_),
        KOKKOS_LAMBDA(const int graph) {
            for (int d = 0; d < 3; ++d)
                external_seed(graph, 1 + d) = static_cast<Precision>(
                    aggregates(graph, 1 + d));
        });

    ComposedReverseResult composed;
    composed.adjoints = reverse(inputs, Seeds{},
        /*materialize_table_adjoints=*/false,
        &grad_cos_table, &grad_sin_table,
        materialize_learned_adjoints,
        /*use_resident_output_seeds=*/true,
        /*use_reference_potential_adjoint=*/false,
        materialize_input_adjoints);

    if (materialize_input_adjoints) {
        const auto host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), composed_stress_device_);
        composed.stress.resize(graphs * 9);
        for (std::size_t graph = 0; graph < graphs; ++graph)
            for (int component = 0; component < 9; ++component)
                composed.stress[9 * graph + component] =
                    host(graph, component);
    }
    return composed;
}

} // namespace symmetrix::mace_polar

// The delta layer's instantiations: the family's evaluator is templated on the
// model's precision exactly as the rest of the backbone is.
template class symmetrix::mace_polar::NodeLocalUvuTensorProductKokkosT<float>;
template class symmetrix::mace_polar::NodeLocalUvuTensorProductKokkosT<double>;
template class symmetrix::mace_polar::PolarGateKokkosT<float>;
template class symmetrix::mace_polar::PolarGateKokkosT<double>;
template class symmetrix::mace_polar::PolarGatedReadoutKokkosT<float>;
template class symmetrix::mace_polar::PolarGatedReadoutKokkosT<double>;
template class symmetrix::mace_polar::PolarNonLinearBiasReadoutKokkosT<float>;
template class symmetrix::mace_polar::PolarNonLinearBiasReadoutKokkosT<double>;
template class symmetrix::mace_polar::PolarPotentialEmbeddingKokkosT<float>;
template class symmetrix::mace_polar::PolarPotentialEmbeddingKokkosT<double>;
template class symmetrix::mace_polar::PolarFieldUpdateKokkosT<float>;
template class symmetrix::mace_polar::PolarFieldUpdateKokkosT<double>;
template class symmetrix::mace_polar::PolarLocalElectronReadoutKokkosT<float>;
template class symmetrix::mace_polar::PolarLocalElectronReadoutKokkosT<double>;
template class symmetrix::mace_polar::PolarDeltaChainKokkosT<float>;
template class symmetrix::mace_polar::PolarDeltaChainKokkosT<double>;
