#include "mace_les_kokkos.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

template <typename Precision>
void MACELESKokkos<Precision>::begin_model_evaluation(const int num_nodes)
{
    this->les_.charge_readouts_ready = false;
    this->les_.charge_readout_nodes = 0;
    if (this->streamed_edges != ExecutionMode::generic
        && !uses_prepared_execution(this->streamed_edges))
        throw std::invalid_argument(
            "MACELES requires reference or prepared direct execution.");
    if (uses_prepared_execution(this->streamed_edges)) {
        if (this->model_evaluation_graph_generation == 0)
            throw std::invalid_argument(
                "LES direct evaluation requires the current prepared graph token.");
        if (this->execution_prepared_num_feature_nodes != num_nodes)
            throw std::invalid_argument(
                "LES direct evaluation requires one feature node per atom.");
        if (this->single_layer_tiled_plan_active
            || this->dual_layer_tiled_plan_active)
            throw std::invalid_argument(
                "LES fixed-workspace tiled execution is unavailable.");
    }
    if (!this->les_.geometry_ready
        || this->les_.geometry_active_nodes != num_nodes)
        throw std::invalid_argument(
            "LES evaluation requires fresh LES geometry for the active node count.");
}

template <typename Precision>
void MACELESKokkos<Precision>::prepare_model_readouts(const int num_nodes)
{
    this->prepare_les_charge_readouts(num_nodes);
}

template <typename Precision>
void MACELESKokkos<Precision>::evaluate_model_after_readouts(const int num_nodes)
{
    this->evaluate_les_term(num_nodes);
}

template <typename Precision>
void MACELESKokkos<Precision>::before_model_evaluation_fence()
{
    if (this->use_factorized_async_inference())
        this->enqueue_les_energy_copy();
}

template <typename Precision>
void MACELESKokkos<Precision>::after_model_evaluation_fence()
{
    if (this->use_factorized_async_inference())
        this->publish_les_energy();
}

template <typename Precision>
void MACELESKokkos<Precision>::finish_model_evaluation(int)
{
    this->les_.charge_readouts_ready = false;
    this->les_.charge_readout_nodes = 0;
    this->les_.geometry_ready = false;
    this->les_.results_ready = true;
}

template <typename Precision>
void MACELESKokkos<Precision>::compose_model_forces(const int num_nodes)
{
    this->add_les_explicit_forces(num_nodes);
}

template <typename Precision>
void MACELESKokkos<Precision>::compose_model_stress(const double volume)
{
    this->add_les_explicit_stress(volume);
}

template void MACELESKokkos<float>::begin_model_evaluation(int);
template void MACELESKokkos<double>::begin_model_evaluation(int);
template void MACELESKokkos<float>::prepare_model_readouts(int);
template void MACELESKokkos<double>::prepare_model_readouts(int);
template void MACELESKokkos<float>::evaluate_model_after_readouts(int);
template void MACELESKokkos<double>::evaluate_model_after_readouts(int);
template void MACELESKokkos<float>::before_model_evaluation_fence();
template void MACELESKokkos<double>::before_model_evaluation_fence();
template void MACELESKokkos<float>::after_model_evaluation_fence();
template void MACELESKokkos<double>::after_model_evaluation_fence();
template void MACELESKokkos<float>::finish_model_evaluation(int);
template void MACELESKokkos<double>::finish_model_evaluation(int);
template void MACELESKokkos<float>::compose_model_forces(int);
template void MACELESKokkos<double>::compose_model_forces(int);
template void MACELESKokkos<float>::compose_model_stress(double);
template void MACELESKokkos<double>::compose_model_stress(double);

template <typename Precision>
void MACELESKokkos<Precision>::compute_les_charges(const int num_nodes)
{
    if (!les_.parameters)
        throw std::logic_error("LES charge evaluation requires MACELES parameters.");
    if (num_nodes < 0)
        throw std::invalid_argument("LES charge node count cannot be negative.");
    if (this->H1.extent(0) < static_cast<std::size_t>(num_nodes)
        || this->H1.extent(1) == 0
        || this->H1.extent_int(2) != this->num_channels
        || this->H2.extent(0) < static_cast<std::size_t>(num_nodes)
        || this->H2.extent_int(1) != this->num_channels)
        throw std::invalid_argument(
            "MACE feature extents do not match the LES charge readouts.");

    const auto old_capacity = les_.latent_charges.extent(0);
    const auto capacity = std::max<std::size_t>(
        old_capacity, static_cast<std::size_t>(num_nodes));
    const bool replace = les_.charge_readout_1.extent(0) < capacity
        || les_.charge_readout_2.extent(0) < capacity
        || les_.charge_readout_2_gradient.extent(0) < capacity
        || les_.charge_readout_2_gradient.extent_int(1) != this->num_channels
        || les_.latent_charges.extent(0) < capacity
        || les_.charge_adjoints.extent(0) < capacity;
    if (replace) {
        Kokkos::realloc(
            Kokkos::WithoutInitializing,
            les_.charge_readout_1, capacity);
        Kokkos::realloc(
            Kokkos::WithoutInitializing,
            les_.charge_readout_2, capacity);
        Kokkos::realloc(
            Kokkos::WithoutInitializing,
            les_.charge_readout_2_gradient, capacity, this->num_channels);
        Kokkos::realloc(
            Kokkos::WithoutInitializing,
            les_.latent_charges, capacity);
        Kokkos::realloc(
            Kokkos::WithoutInitializing,
            les_.charge_adjoints, capacity);
        ++les_.charge_workspace_replacements;
    } else {
        ++les_.charge_workspace_reuses;
    }
    les_.charge_active_nodes = num_nodes;

    const bool async_inference = this->use_factorized_async_inference();
    const auto execution_space = async_inference
        ? this->factorized_execution_space : Kokkos::DefaultExecutionSpace();
    const auto h1 = this->H1;
    const auto weights = les_.readout_1_weights;
    const auto charge_1 = les_.charge_readout_1;
    const int channels = this->num_channels;
    Kokkos::parallel_for(
        "Compute LES first charge readout",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int node) {
            double value = 0.0;
            for (int channel=0; channel<channels; ++channel)
                value += weights(channel)
                    *static_cast<double>(h1(node,0,channel));
            charge_1(node) = value;
        });

    const auto h2 = Kokkos::subview(
        this->H2, Kokkos::make_pair(0,num_nodes), Kokkos::ALL);
    const auto charge_2 = Kokkos::subview(
        les_.charge_readout_2, Kokkos::make_pair(0,num_nodes));
    const auto charge_2_gradient = Kokkos::subview(
        les_.charge_readout_2_gradient,
        Kokkos::make_pair(0,num_nodes), Kokkos::ALL);
    les_.readout_2.evaluate_gradient(
        execution_space, h2, charge_2, charge_2_gradient, false);

    const auto charges = les_.latent_charges;
    const double output_scale = les_.parameters->output_scale;
    Kokkos::parallel_for(
        "Sum LES charge readouts",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int node) {
            charges(node) = output_scale*(charge_1(node)+charge_2(node));
        });
    if (!async_inference)
        execution_space.fence("LES charge readout evaluation");
}


template <typename Precision>
void MACELESKokkos<Precision>::prepare_les_charge_readouts(const int num_nodes)
{
    if (!is_mace_les())
        return;
    les_.charge_readouts_ready = false;
    les_.charge_readout_nodes = 0;
    if (this->streamed_edges != ExecutionMode::generic
        && !uses_prepared_execution(this->streamed_edges))
        throw std::invalid_argument(
            "MACELES requires reference or prepared direct execution.");
    if (!les_.geometry_ready || les_.geometry_active_nodes != num_nodes)
        throw std::invalid_argument(
            "LES evaluation requires fresh LES geometry for the active node count.");

    compute_les_charges(num_nodes);
    les_.charge_readout_nodes = num_nodes;
    les_.charge_readouts_ready = true;
}

template <typename Precision>
void MACELESKokkos<Precision>::evaluate_les_term(const int num_nodes)
{
    if (!is_mace_les())
        return;
    if (this->streamed_edges != ExecutionMode::generic
        && !uses_prepared_execution(this->streamed_edges))
        throw std::invalid_argument(
            "MACELES requires reference or prepared direct execution.");
    if (!les_.charge_readouts_ready || les_.charge_readout_nodes != num_nodes)
        throw std::logic_error(
            "LES charge readout state is unavailable or does not match the active node count.");

    les_.charge_readouts_ready = false;
    les_.charge_readout_nodes = 0;
    les_.evaluation_execution_space_selected = true;
    les_.evaluation_execution_space_factorized =
        uses_prepared_execution(this->streamed_edges);
    const auto execution_space = les_.evaluation_execution_space_factorized
        ? this->factorized_execution_space : Kokkos::DefaultExecutionSpace();
    const auto result = les_.periodic
        ? les_.monopole->compute_periodic_device(
            execution_space, les_.latent_charges, les_.positions,
            static_cast<std::size_t>(num_nodes))
        : les_.monopole->compute_nonperiodic_device(
            execution_space, les_.latent_charges, les_.positions,
            static_cast<std::size_t>(num_nodes));
    les_.atomic_energies = result.atomic_energies;
    les_.energy_device = result.energy;
    les_.explicit_forces = result.forces;
    les_.charge_adjoints = result.charge_adjoints;
    les_.strain_derivative = result.strain_derivative;
    if (!this->use_factorized_async_inference())
        publish_les_energy();

    const auto energies = this->node_energies;
    const auto les_energies = les_.atomic_energies;
    Kokkos::parallel_for(
        "Compose LES atomic energies",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int node) {
            energies(node) += les_energies(node);
        });
    if (les_.charge_adjoint_accumulation_enabled)
        accumulate_les_charge_readout_adjoints(num_nodes);
    if (!this->use_factorized_async_inference())
        execution_space.fence("Compose LES energy and charge adjoints");
}

template <typename Precision>
void MACELESKokkos<Precision>::enqueue_les_energy_copy()
{
    if (les_.energy_device.data() == nullptr)
        throw std::logic_error("LES device energy is unavailable.");
    Kokkos::deep_copy(
        this->factorized_execution_space, les_.energy_host, les_.energy_device);
    les_.energy_copy_pending = true;
}

template <typename Precision>
void MACELESKokkos<Precision>::publish_les_energy()
{
    if (les_.energy_device.data() == nullptr)
        throw std::logic_error("LES device energy is unavailable.");
    if (!les_.energy_copy_pending)
        Kokkos::deep_copy(les_.energy_host, les_.energy_device);
    les_.energy = les_.energy_host();
    les_.energy_copy_pending = false;
}

template <typename Precision>
void MACELESKokkos<Precision>::add_les_explicit_forces(const int num_nodes)
{
    if (!is_mace_les())
        return;
    if (!les_.results_ready
        || les_.explicit_forces.extent(0) < 3*static_cast<std::size_t>(num_nodes))
        throw std::logic_error("LES explicit forces are not ready.");
    const auto forces = this->atom_forces;
    const auto les_forces = les_.explicit_forces;
    Kokkos::parallel_for(
        "Compose LES explicit atom forces",
        Kokkos::RangePolicy<decltype(this->factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
            this->factorized_execution_space, 0, 3*static_cast<std::size_t>(num_nodes)),
        KOKKOS_LAMBDA (const std::size_t index) {
            forces(index) += les_forces(index);
        });
}

template <typename Precision>
void MACELESKokkos<Precision>::add_les_explicit_stress(const double volume)
{
    if (!is_mace_les())
        return;
    if (!les_.results_ready || les_.strain_derivative.extent(0) != 9)
        throw std::logic_error("LES explicit stress is not ready.");
    const auto stress = this->stress_tensor;
    const auto derivative = les_.strain_derivative;
    Kokkos::parallel_for(
        "Compose LES explicit stress",
        Kokkos::RangePolicy<decltype(this->factorized_execution_space)>(
            this->factorized_execution_space, 0, 9),
        KOKKOS_LAMBDA (const int component) {
            stress(component) += derivative(component)/volume;
        });
}

template <typename Precision>
void MACELESKokkos<Precision>::accumulate_les_charge_readout_adjoints(
    const int num_nodes)
{
    if (!les_.parameters)
        throw std::logic_error("LES charge reverse requires MACELES parameters.");
    if (num_nodes < 0 || num_nodes != les_.charge_active_nodes)
        throw std::invalid_argument(
            "LES charge reverse node count does not match the saved readout.");
    if (les_.charge_adjoints.extent(0) < static_cast<std::size_t>(num_nodes)
        || les_.charge_readout_2_gradient.extent(0)
            < static_cast<std::size_t>(num_nodes)
        || les_.charge_readout_2_gradient.extent_int(1) != this->num_channels
        || this->H1_adj.extent(0) < static_cast<std::size_t>(num_nodes)
        || this->H1_adj.extent(1) == 0
        || this->H1_adj.extent_int(2) != this->num_channels
        || this->H2_adj.extent(0) < static_cast<std::size_t>(num_nodes)
        || this->H2_adj.extent_int(1) != this->num_channels)
        throw std::invalid_argument(
            "LES charge reverse extents do not match the MACE adjoint buffers.");

    const auto execution_space = this->use_factorized_async_inference()
        ? this->factorized_execution_space : Kokkos::DefaultExecutionSpace();
    const auto h1_adjoint = this->H1_adj;
    const auto h2_adjoint = this->H2_adj;
    const auto charge_adjoint = les_.charge_adjoints;
    const auto readout_1_weights = les_.readout_1_weights;
    const auto readout_2_gradient = les_.charge_readout_2_gradient;
    const double output_scale = les_.parameters->output_scale;
    const int channels = this->num_channels;
    const std::size_t entries = static_cast<std::size_t>(num_nodes)
        *static_cast<std::size_t>(channels);
    Kokkos::parallel_for(
        "Accumulate LES charge readout adjoints",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, std::size_t(0), entries),
        KOKKOS_LAMBDA (const std::size_t index) {
            const int node = static_cast<int>(
                index/static_cast<std::size_t>(channels));
            const int channel = static_cast<int>(
                index%static_cast<std::size_t>(channels));
            const double seed = output_scale*charge_adjoint(node);
            h1_adjoint(node,0,channel) += static_cast<Precision>(
                seed*readout_1_weights(channel));
            h2_adjoint(node,channel) +=
                seed*readout_2_gradient(node,channel);
        });
    if (!this->use_factorized_async_inference())
        execution_space.fence("LES charge readout reverse");
}

template <typename Precision>
std::size_t MACELESKokkos<Precision>::les_charge_workspace_active_bytes() const
{
    return sizeof(double)*static_cast<std::size_t>(les_.charge_active_nodes)
            *static_cast<std::size_t>(4+this->num_channels)
        +les_.readout_2.estimated_workspace_bytes(
            static_cast<std::size_t>(les_.charge_active_nodes), true);
}

template <typename Precision>
std::size_t MACELESKokkos<Precision>::les_charge_workspace_capacity_bytes() const
{
    return sizeof(double)*(les_.charge_readout_1.size()
        +les_.charge_readout_2.size()+les_.charge_readout_2_gradient.size()
        +les_.latent_charges.size()+les_.charge_adjoints.size())
        +les_.readout_2.workspace_bytes();
}


template void MACELESKokkos<float>::compute_les_charges(int);
template void MACELESKokkos<double>::compute_les_charges(int);
template void MACELESKokkos<float>::prepare_les_charge_readouts(int);
template void MACELESKokkos<double>::prepare_les_charge_readouts(int);
template void MACELESKokkos<float>::evaluate_les_term(int);
template void MACELESKokkos<double>::evaluate_les_term(int);
template void MACELESKokkos<float>::enqueue_les_energy_copy();
template void MACELESKokkos<double>::enqueue_les_energy_copy();
template void MACELESKokkos<float>::publish_les_energy();
template void MACELESKokkos<double>::publish_les_energy();
template void MACELESKokkos<float>::add_les_explicit_forces(int);
template void MACELESKokkos<double>::add_les_explicit_forces(int);
template void MACELESKokkos<float>::add_les_explicit_stress(double);
template void MACELESKokkos<double>::add_les_explicit_stress(double);
template void MACELESKokkos<float>::accumulate_les_charge_readout_adjoints(int);
template void MACELESKokkos<double>::accumulate_les_charge_readout_adjoints(int);
template std::size_t MACELESKokkos<float>::les_charge_workspace_active_bytes() const;
template std::size_t MACELESKokkos<double>::les_charge_workspace_active_bytes() const;
template std::size_t MACELESKokkos<float>::les_charge_workspace_capacity_bytes() const;
template std::size_t MACELESKokkos<double>::les_charge_workspace_capacity_bytes() const;
