#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "affine_mlp_kokkos.hpp"
#include "e3nn_kokkos.hpp"
#include "long_range_kernels.hpp"
#include "long_range_solver_policy.hpp"
#include "long_range_operator_kinds.hpp"
#include "mace_polar_program.hpp"
#include "nlohmann/json.hpp"
#include "mace_polar_particle_mesh.hpp"

// The PolarMACE backbone deltas: the family's node-local execution layer.
//
// The neutral internal backbone engine produces the per-layer node features;
// everything the Polar sibling adds on top is a node-local operator with no
// neighbor-list (and no position) dependence -- the per-layer spin-source
// multipole heads, the multi-layer feature mixer, the pre-loop Fukui source
// map, the field-dependent charge-update chain (potential/source embeddings,
// node-local sparse uvu tensor products, an affine MLP, and a gated readout),
// and the post-loop local-electron-energy readout -- plus the unrolled
// non-self-consistent chain that drives them against the long-range operator
// group's own kernels (long_range_kernels.hpp).
//
// Because every delta block is node-local, none of them contributes a position
// or cell adjoint: the geometry/cell reverse stays exactly where it was (the
// long-range kernels and the backbone).  Their derivative surface is the
// *feature* adjoint -- the seeds of their inputs' feature spaces -- which is
// what the per-block reverse() entry points return.
//
// The oracle is mace's PolarMACE.forward and its field blocks at the pinned
// checkout; see docs/user/models.md#mace-polar-1 for the model contract:
// ``SparseUvuTensorProduct.forward`` for the tied-channel tensor products,
// ``GatedEquivariantBlock`` for the gated nonlinearity, and
// ``GeneralNonLinearBiasReadoutBlock`` / ``NonLinearBiasReadoutBlock`` /
// ``RadialMLP`` for the readouts and MLPs.  The composition is the oracle's
// bit-pinned feature rule: ``(project(coulomb(assemble(multipoles)))
// - feature_self_terms + correction_fields + 0.5*external_field) /
// field_feature_norms``.
namespace symmetrix::mace_polar {

// The oracle's dense equivariant linears carry a *partial* bias when they are
// declared with ``biases=True``: e3nn biases only the scalar output blocks,
// one entry per multiplicity channel, while the shared e3 linear kernel
// applies its bias per output element.  Expanding the partial bias to the
// full output dimension (zeros off the scalar blocks, the channel's own entry
// on them) is exactly the oracle's arithmetic, and it lets the family's delta
// layer reuse the qualified e3 linear kernel instead of a second one.
inline nlohmann::json linear_with_full_bias(const nlohmann::json& definition)
{
    nlohmann::json result = definition;
    if (!result.contains("bias"))
        return result;
    const std::vector<double> bias =
        result.at("bias").at("values").get<std::vector<double>>();
    if (bias.empty())
        return result;
    const Irreps output(definition.at("irreps_out").get<std::string>());
    std::vector<double> full(static_cast<std::size_t>(output.dimension()), 0.0);
    std::size_t cursor = 0;
    for (const IrrepBlock& block : output.blocks) {
        if (block.l != 0 || block.parity != 1)
            continue;
        for (int channel = 0; channel < block.multiplicity; ++channel) {
            if (cursor >= bias.size())
                throw std::invalid_argument(
                    "a polar e3 linear's bias does not match its scalar "
                    "output blocks.");
            full[static_cast<std::size_t>(block.offset + channel)] =
                bias[cursor++];
        }
    }
    if (cursor != bias.size())
        throw std::invalid_argument(
            "a polar e3 linear's bias does not match its scalar output "
            "blocks.");
    result["bias"]["values"] = full;
    result["bias"]["shape"] = std::vector<std::size_t>{full.size()};
    return result;
}

inline nlohmann::json polar_specialized_linear(
    const nlohmann::json& definition,const char* role)
{
    nlohmann::json result=linear_with_full_bias(definition);
    result["execution_specialization"]={
        {"family","polar_e3linear"},{"role",role}};
    return result;
}

// The tied-channel node-local tensor product of the oracle's
// ``SparseUvuTensorProduct``: per path, a weighted bilinear over the two
// inputs' multiplicity channels with the angular coupling collapsed
// (``l x l -> 0``) or carried through (``l x 0 -> l``).  The forward is exactly
// the oracle's einsum pair (including the path weight and the ``1/sqrt(2l+1)``
// tied-channel normalization and the elementwise output mask); the reverse is
// its transpose, which is what makes the block's adjoint available without a
// Wigner tensor of its own.
template<typename Precision>
class NodeLocalUvuTensorProductKokkosT {
public:
    struct Path {
        int input_1_offset = 0, input_2_offset = 0, output_offset = 0;
        int multiplicity_1 = 0, multiplicity_2 = 0;
        // The tied angular width: components per multiplicity channel.  It is
        // 1 for a contraction's input_2/output and d for a modulation.
        int component_width = 0;
        int weight_offset = 0;
        Precision path_weight = Precision(0);
        Precision inverse_sqrt_width = Precision(1);
        // True for ``l x l -> 0`` (the output is a scalar per in-1 channel),
        // false for ``l x 0 -> l`` (the in-2 scalars modulate in-1).
        bool contraction = true;
    };
    // Public for the same reason Path is: an extended __device__ lambda cannot
    // declare a captured variable of a private nested type.
    struct ElementEntry {
        // Index into path_table_.
        int path = 0;
        // The contributing channel: the in-1 channel for kinds that emit one
        // channel's contribution, the fixed in-2 ``other`` for the two kinds
        // that fold a channel loop into the entry.
        int slot = 0;
        // Address component within the channel (0 for whole-channel kinds).
        int component = 0;
        int kind = 0;
    };
    NodeLocalUvuTensorProductKokkosT() = default;
    explicit NodeLocalUvuTensorProductKokkosT(const nlohmann::json& definition);

    int input_1_dimension() const { return input_1_dimension_; }
    int input_2_dimension() const { return input_2_dimension_; }
    int output_dimension() const { return output_dimension_; }
    int weight_count() const { return weight_count_; }
    std::size_t path_count() const { return paths_.size(); }
    const std::vector<Path>& paths() const { return paths_; }
    void prepare(int samples);
    std::size_t workspace_replacement_count() const
    {
        return workspace_replacement_count_;
    }
    // out = tensor_product(input_1, input_2) * output_mask.  ``output`` is
    // overwritten (the oracle builds it from zeros and accumulates the paths).
    void evaluate(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> input_1,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> input_2,
        Kokkos::View<Precision**, Kokkos::LayoutRight> output) const;
    // The transpose: accumulates the seeds of both inputs from the output
    // seed (the output mask is differentiated as the forward applies it).
    // ``initialize_adjoints`` clears both adjoints first.
    void reverse(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> input_1,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> input_2,
        Kokkos::View<Precision**, Kokkos::LayoutRight> input_1_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> input_2_adjoint,
        bool initialize_adjoints = true) const;

private:
    // Inverts the path table into the per-element contribution lists the
    // element-parallel kernels walk (see the public ElementEntry above).
    void build_contribution_tables();

    // The path table lives in a shared-space view as well as the host vector:
    // the kernels read it every call, so it is staged once at parse time
    // rather than refilled per evaluation.
    std::vector<Path> paths_;
    Kokkos::View<Path*, Kokkos::SharedSpace> path_table_;
    Kokkos::View<Precision*> weights_;
    Kokkos::View<Precision*> output_mask_;
    // Per-element inverted contribution lists (entries between offsets[e] and
    // offsets[e + 1] contribute to element e, in the original accumulation
    // order) for the forward output, the reverse's in-1 adjoint, and the
    // reverse's in-2 adjoint.
    Kokkos::View<ElementEntry*, Kokkos::SharedSpace> forward_entries_;
    Kokkos::View<int*, Kokkos::SharedSpace> forward_offsets_;
    Kokkos::View<ElementEntry*, Kokkos::SharedSpace> reverse_in1_entries_;
    Kokkos::View<int*, Kokkos::SharedSpace> reverse_in1_offsets_;
    Kokkos::View<ElementEntry*, Kokkos::SharedSpace> reverse_in2_entries_;
    Kokkos::View<int*, Kokkos::SharedSpace> reverse_in2_offsets_;
    // Dense paths execute serially within a block, so one extent-keyed set of
    // operands is sufficient for every path.  Keeping it with the parsed
    // block avoids allocating four large views for every forward/reverse.
    Kokkos::View<Precision**, Kokkos::LayoutRight> packed_input_2_workspace_,
        mixed_workspace_, left_workspace_, grad_input_2_workspace_;
    int workspace_samples_ = 0, workspace_rows_ = 0;
    std::size_t workspace_replacement_count_ = 0;
    // POLAR-1's learned uvu blocks are dense 512x512 channel maps.  They use
    // the GEMM-based specialization instead of making one thread serially
    // walk all 512 partner channels.
    bool dense_512_specialization_ = false;
    int input_1_dimension_ = 0, input_2_dimension_ = 0, output_dimension_ = 0;
    int weight_count_ = 0;
};

// The gated equivariant nonlinearity of the oracle's ``GatedEquivariantBlock``
// (mace's own, which -- unlike e3nn's -- carries no learnable weights): the
// input's leading scalar channels are activated and scaled, its gate channels
// are activated and scale the trailing gated channels per multiplicity.  The
// split and its order are the oracle's own (a stable sort of
// scalars + gates + gated by (l, -parity, multiplicity), merged on simplify),
// so the kernel reproduces the block rather than a convention about it.
template<typename Precision>
class PolarGateKokkosT {
public:
    struct GatedGroup {
        int offset = 0, length = 0;
        int component_width = 0, multiplicity = 0;
        // Where the group's gate channels start inside the gate slice.
        int gate_offset = 0;
    };
    // Pre-resolved element ownership lets the device kernels expose the full
    // row width as parallel work instead of having one worker serially walk a
    // complete gate row.  Reverse kind 1 retains the small component dot
    // product in its original order for the gate-channel adjoint.
    struct ForwardElement {
        int input_element = -1;
        int gate_element = -1;
    };
    struct ReverseElement {
        int kind = -1;
        int output_element = 0;
        int source_offset = 0;
        int component_width = 0;
        int gate_element = -1;
    };
    PolarGateKokkosT() = default;
    explicit PolarGateKokkosT(const nlohmann::json& definition);

    int input_dimension() const { return input_dimension_; }
    int output_dimension() const { return output_dimension_; }

    void evaluate(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
        Kokkos::View<Precision**, Kokkos::LayoutRight> output) const;
    void reverse(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
        Kokkos::View<Precision**, Kokkos::LayoutRight> input_adjoint,
        bool initialize_input_adjoint = true) const;

    // nvcc refuses a private or protected member type in a variable captured by
    // an extended __host__ __device__ lambda, so the activation selectors the
    // device kernels capture live in the public section.
    enum class Activation { identity, silu, sigmoid };

private:
    Activation scalar_activation_ = Activation::identity;
    Activation gate_activation_ = Activation::identity;
    Precision scalar_constant_ = Precision(1);
    Precision gate_constant_ = Precision(1);
    int scalar_offset_ = 0, scalar_length_ = 0;
    int gate_offset_ = 0, gate_length_ = 0;
    int input_dimension_ = 0, output_dimension_ = 0;
    std::vector<GatedGroup> gated_;
    // Staged once at parse time, like the tensor product's path table.
    Kokkos::View<GatedGroup*, Kokkos::SharedSpace> group_table_;
    Kokkos::View<ForwardElement*, Kokkos::SharedSpace> forward_elements_;
    Kokkos::View<ReverseElement*, Kokkos::SharedSpace> reverse_elements_;
};

// The oracle's ``GeneralNonLinearBiasReadoutBlock``: a dense linear into the
// gate's input space, the gated nonlinearity, a second dense linear, the
// nonlinearity again, and the output linear.  The field update's readout uses
// it to emit the charge channels followed by the iteration's Fukui sources.
template<typename Precision>
class PolarGatedReadoutKokkosT {
public:
    PolarGatedReadoutKokkosT() = default;
    explicit PolarGatedReadoutKokkosT(const nlohmann::json& definition);

    int input_dimension() const { return linear_1_.input_dimension(); }
    int output_dimension() const { return linear_2_.output_dimension(); }
    int hidden_dimension() const { return linear_1_.output_dimension(); }
    std::vector<const E3LinearKokkosT<Precision>*> e3linears() const {
        return {&linear_1_,&linear_mid_,&linear_2_};
    }
    // The gated nonlinearity itself, so its own adjoint is exercisable.
    PolarGateKokkosT<Precision>& gate() { return gate_; }
    void prepare(int nodes);

    void evaluate(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
        Kokkos::View<Precision**, Kokkos::LayoutRight> output);
    void reverse(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
        Kokkos::View<Precision**, Kokkos::LayoutRight> input_adjoint,
        bool initialize_input_adjoint);

private:
    E3LinearKokkosT<Precision> linear_1_, linear_mid_, linear_2_;
    PolarGateKokkosT<Precision> gate_;
    // hidden_ holds linear_1's output (the second gate's input), gated_ the
    // first gate's output; the two adjoint buffers carry the reverse's
    // intermediate seeds, which is why the reverse needs no third buffer.
    Kokkos::View<Precision**, Kokkos::LayoutRight> hidden_, gated_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> gate_input_, gate_output_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> adjoint_a_, adjoint_b_;
};

// The oracle's ``NonLinearBiasReadoutBlock``: the same three-line linear stack
// with a plain normalized activation between them.  The Fukui source map uses
// it to turn the backbone's last-layer features into the per-node source pair.
template<typename Precision>
class PolarNonLinearBiasReadoutKokkosT {
public:
    PolarNonLinearBiasReadoutKokkosT() = default;
    explicit PolarNonLinearBiasReadoutKokkosT(const nlohmann::json& definition);

    int input_dimension() const { return linear_1_.input_dimension(); }
    int output_dimension() const { return linear_2_.output_dimension(); }
    std::vector<const E3LinearKokkosT<Precision>*> e3linears() const {
        return {&linear_1_,&linear_mid_,&linear_2_};
    }
    void prepare(int nodes);

    void evaluate(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output);
    void reverse(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
        bool initialize_input_adjoint);

    // nvcc refuses a private or protected member type in a variable captured by
    // an extended __host__ __device__ lambda, so the activation selector the
    // device kernels capture lives in the public section.
    enum class Activation { identity, silu, sigmoid };

    // The forward's activation with the oracle's normalization constant, and
    // its transpose with respect to the seed (the pre-activation value is the
    // derivative's input).  Public for the same nvcc rule as above: both
    // enclose extended lambdas.
    void activate(
        Kokkos::View<Precision**, Kokkos::LayoutRight> pre,
        Kokkos::View<Precision**, Kokkos::LayoutRight> activated) const;
    void reverse_activate(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> pre,
        Kokkos::View<Precision**, Kokkos::LayoutRight> seed) const;

private:
    E3LinearKokkosT<Precision> linear_1_, linear_mid_, linear_2_;
    Activation activation_ = Activation::silu;
    Precision activation_constant_ = Precision(1);
    // The forward stores each stage's pre-activation and activated values: the
    // normalization constant multiplies the activated value, and the reverse
    // needs the pre-activation to differentiate.
    Kokkos::View<Precision**, Kokkos::LayoutRight> first_pre_, first_activated_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> second_pre_, second_activated_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> adjoint_a_, adjoint_b_;
};

// The update chain's potential embedding: three dense equivariant linears
// summed (the potential features, the mixed node features, and the local
// charges), exactly the oracle's ``AgnosticChargeBiasedLinearPotentialEmbedding``.
struct PolarPotentialEmbeddingPackedInputSource {
    int input_kind = 0;
    int input_offset = 0;
    int component_width = 0;
    int source = 0;
};

template<typename Precision>
class PolarPotentialEmbeddingKokkosT {
public:
    PolarPotentialEmbeddingKokkosT() = default;
    explicit PolarPotentialEmbeddingKokkosT(const nlohmann::json& definition);

    int output_dimension() const { return potential_linear_.output_dimension(); }
    int potential_input_dimension() const
    {
        return potential_linear_.input_dimension();
    }
    int node_input_dimension() const
    {
        return node_feats_linear_.input_dimension();
    }
    int charge_input_dimension() const
    {
        return charge_embedding_.input_dimension();
    }
    std::vector<const E3LinearKokkosT<Precision>*> e3linears() const {
        return {&potential_linear_,&node_feats_linear_,&charge_embedding_};
    }
    void prepare(int nodes);

    void evaluate(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> potential_features,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> node_features,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> charges,
        Kokkos::View<Precision**, Kokkos::LayoutRight> output);
    void reverse(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> potential_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> node_features_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> charges_adjoint,
        bool initialize_adjoints);

private:
    E3LinearKokkosT<Precision> potential_linear_, node_feats_linear_,
        charge_embedding_;
    // Only the released POLAR-1/M irreps and ordered path topology enter the
    // packed-GEMM route.  Every other valid definition retains the three
    // E3Linear calls and sum above as its reference schedule.
    bool use_composed_schedule_ = false;
    int composed_workspace_nodes_ = 0;
    Kokkos::View<PolarPotentialEmbeddingPackedInputSource*, Kokkos::SharedSpace>
        scalar_input_sources_, vector_input_sources_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> scalar_weights_,
        vector_weights_, packed_input_, packed_output_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> potential_part_, node_part_,
        charge_part_, potential_part_adjoint_, node_part_adjoint_,
        charge_part_adjoint_;
};

// One iteration of the oracle's ``AgnosticEmbeddedOneBodyVariableUpdate``: the
// potential embedding, the node-local contraction against the mixed features,
// the invariant source embedding, the affine MLP over their concatenation, the
// modulation back into the node-feature space, and the gated readout that
// emits the charge channels plus the iteration's Fukui sources.  Every edge
// argument of the oracle's signature is unused (``edge_set: "empty"``), so the
// whole block is node-local.
template<typename Precision>
class PolarFieldUpdateKokkosT {
public:
    PolarFieldUpdateKokkosT() = default;
    explicit PolarFieldUpdateKokkosT(const nlohmann::json& definition);
    // The node-feature width the block consumes/produces and the invariant
    // width its contraction emits.
    int node_feature_dimension() const { return dot_products_.input_1_dimension(); }
    int source_dimension() const { return source_embedding_.output_dimension(); }
    int output_dimension() const { return readout_.output_dimension(); }
    std::vector<const E3LinearKokkosT<Precision>*> e3linears() const {
        auto result=embedding_.e3linears();
        result.push_back(&source_embedding_);
        const auto readout=readout_.e3linears();
        result.insert(result.end(),readout.begin(),readout.end());
        return result;
    }
    void prepare(int nodes);
    std::size_t uvu_workspace_replacement_count() const
    {
        return dot_products_.workspace_replacement_count()
            + tp_out_.workspace_replacement_count();
    }

    // ``output`` carries the readout's channels: the charge channels first,
    // then the iteration's Fukui source pair.
    // The block's own parts, so each one's forward and reverse are
    // exercisable (and FD-verified) in isolation.
    PolarPotentialEmbeddingKokkosT<Precision>& embedding() { return embedding_; }
    E3LinearKokkosT<Precision>& source_embedding() { return source_embedding_; }
    NodeLocalUvuTensorProductKokkosT<Precision>& dot_products()
    {
        return dot_products_;
    }
    NodeLocalUvuTensorProductKokkosT<Precision>& tp_out() { return tp_out_; }
    AffineMLPKokkosT<Precision>& nonlinearity() { return nonlinearity_; }
    PolarGatedReadoutKokkosT<Precision>& readout() { return readout_; }

    void evaluate(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> node_features,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> potential_features,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> node_attrs,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> local_charges,
        Kokkos::View<Precision**, Kokkos::LayoutRight> output);
    void reverse(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> node_features,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> potential_features,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> node_attrs,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> local_charges,
        Kokkos::View<Precision**, Kokkos::LayoutRight> node_features_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> potential_features_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> node_attrs_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> local_charges_adjoint,
        bool initialize_adjoints);

private:
    PolarPotentialEmbeddingKokkosT<Precision> embedding_;
    E3LinearKokkosT<Precision> source_embedding_;
    NodeLocalUvuTensorProductKokkosT<Precision> dot_products_;
    AffineMLPKokkosT<Precision> nonlinearity_;
    NodeLocalUvuTensorProductKokkosT<Precision> tp_out_;
    PolarGatedReadoutKokkosT<Precision> readout_;
    // The forward's own intermediates (the reverse re-reads them), and the two
    // adjoint buffers the reverse stages in.
    Kokkos::View<Precision**, Kokkos::LayoutRight> mixed_, mixed_adjoint_,
        invariants_, invariants_adjoint_, embedded_, embedded_seed_,
        concatenated_, mlp_input_, mlp_input_adjoint_, mlp_output_,
        mlp_output_adjoint_, new_features_, new_features_adjoint_,
        readout_adjoint_;
};

// The oracle's ``OneBodyMLPFieldReadout``: the post-loop local-electron-energy
// readout.  Two biased dense embeddings (of the summed charge density and of
// the last iteration's potential features) are contracted node-locally against
// the backbone's node features, concatenated, and reduced to one energy per
// node by an affine MLP.
template<typename Precision>
class PolarLocalElectronReadoutKokkosT {
public:
    PolarLocalElectronReadoutKokkosT() = default;
    explicit PolarLocalElectronReadoutKokkosT(const nlohmann::json& definition);
    int node_feature_dimension() const
    {
        return dot_products_q_.input_1_dimension();
    }
    int field_dimension() const { return linear_up_v_.input_dimension(); }
    int charge_dimension() const { return linear_up_q_.input_dimension(); }
    std::vector<const E3LinearKokkosT<Precision>*> e3linears() const {
        return {&linear_up_q_,&linear_up_v_};
    }
    // The readout's own parts, for the block-level gates.
    E3LinearKokkosT<Precision>& linear_up_q() { return linear_up_q_; }
    E3LinearKokkosT<Precision>& linear_up_v() { return linear_up_v_; }
    NodeLocalUvuTensorProductKokkosT<Precision>& dot_products_q()
    {
        return dot_products_q_;
    }
    NodeLocalUvuTensorProductKokkosT<Precision>& dot_products_v()
    {
        return dot_products_v_;
    }
    AffineMLPKokkosT<Precision>& mlp() { return mlp_; }
    void prepare(int nodes);
    std::size_t uvu_workspace_replacement_count() const
    {
        return dot_products_q_.workspace_replacement_count()
            + dot_products_v_.workspace_replacement_count();
    }

    void evaluate(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> node_features,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> field_features,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> charges_0,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> charges_induced,
        Kokkos::View<Precision**, Kokkos::LayoutRight> output);
    void reverse(
        Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> node_features,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> field_features,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> charges_0,
        Kokkos::View<const Precision**, Kokkos::LayoutRight> charges_induced,
        Kokkos::View<Precision**, Kokkos::LayoutRight> node_features_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> field_features_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> charges_0_adjoint,
        Kokkos::View<Precision**, Kokkos::LayoutRight> charges_induced_adjoint,
        bool initialize_adjoints);

private:
    E3LinearKokkosT<Precision> linear_up_q_, linear_up_v_;
    NodeLocalUvuTensorProductKokkosT<Precision> dot_products_q_,
        dot_products_v_;
    AffineMLPKokkosT<Precision> mlp_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> charges_, charges_seed_,
        charge_embedding_, charge_embedding_adjoint_, field_embedding_,
        field_embedding_adjoint_, invariants_, invariants_adjoint_, energy_,
        invariant_q_, invariant_q_adjoint_, invariant_v_,
        invariant_v_adjoint_;
};

// The polar delta chain: the parsed blocks plus the unrolled
// non-self-consistent orchestration the contract's schedule *describes*
// (execution::polar::polar_execution_schedule; mace_polar_program.hpp).  The
// chain runs the same fixed linear order the schedule names -- the interaction
// layers' heads, the shared geometry cache, the feature mixer, the pre-loop
// equilibration, U unrolled updates (long-range electrostatic features, external-field
// coupling, the local charge update, the equilibration), and the post-loop
// local-electron readout -- with no data-dependent control flow, but this
// class walks it as a written sequence over its own parsed blocks rather than
// dispatching over the schedule's entries: the two are pinned equal by the
// program's schedule-coverage rule and by this layer's fixtures, not by a
// shared iterator.  Driving the loop off the schedule's entry list is the
// end-to-end stage's migration (the plan's task 11 note), and the difference
// is recorded in the task-13a report.
template<typename Precision>
class PolarDeltaChainKokkosT {
public:
    struct E3LinearDiagnostics {
        std::size_t linear_count=0,selected_count=0,forward_count=0,
            reverse_count=0,fallback_count=0;
        std::vector<std::pair<std::string,std::size_t>> roles;
    };

    struct PotentialAdjointDiagnostics {
        std::size_t selected_count = 0;
        std::size_t reference_count = 0;
        std::uint64_t selected_node_extent = 0;
        std::uint64_t selected_graph_extent = 0;
    };
    struct PotentialAdjointExtents {
        std::uint64_t nodes = 0;
        std::uint64_t graphs = 0;
        std::uint64_t external_node_elements = 0;
        std::uint64_t external_graph_elements = 0;
        std::uint64_t correction_node_elements = 0;
        std::uint64_t correction_graph_elements = 0;
    };
    struct GraphOutputDiagnostics {
        std::size_t selected_count = 0;
        std::uint64_t league_extent = 0;
        std::uint64_t node_extent = 0;
        std::uint64_t k_extent = 0;
    };
    using LongRangeContract = execution::polar::PolarLongRangeContract;

    // The chain's contract data: the long-range slice (basis, cutoffs, depth,
    // precision policy) plus the delta layer's own dimensions, all of which
    // come from the model file the evaluator parsed.
    struct Contract {
        LongRangeContract long_range;
        int interaction_count = 0;
        int node_feature_dimension = 0;
        int num_elements = 0;
        // The checkpoint's `field_si`: when false the electrostatic features carry
        // the self-interaction subtraction (the oracle's own rule).
        bool subtract_feature_self_terms = true;
    };

    struct Inputs {
        execution::long_range::LongRangeSolverRequest solver_request =
            execution::long_range::LongRangeSolverRequest::direct_reciprocal;
        // The backbone's per-layer node features, flattened layer-major.
        std::vector<double> node_features;      // [nodes, layers * width]
        // Production callers may hand the chain the evaluator-owned retained
        // layers directly.  These device views are mutually exclusive with
        // ``node_features`` and avoid the diagnostic NumPy materialization.
        std::vector<Kokkos::View<const Precision**, Kokkos::LayoutRight>>
            node_feature_views;                 // [layers][nodes, width]
        std::vector<double> node_attrs;         // [nodes, num_elements]
        // When node_attrs is absent, build its one-hot spelling directly in
        // the kernel space from these compact model element indices.
        std::vector<int> node_types;             // [nodes]
        std::vector<double> positions;          // [nodes, 3]
        std::vector<double> cell;               // [graphs, 3, 3]
        std::vector<int> graph_pbc;             // [graphs, 3]
        std::vector<int> node_graph_index;      // [nodes]
        std::vector<double> total_charge;       // [graphs]
        std::vector<double> total_spin;         // [graphs]
        std::vector<double> external_potential; // [graphs, 4]
        // Runtime/backend PME policy; never serialized into the model.
        std::array<std::int64_t, 3> pme_mesh{64, 64, 64};
        std::int64_t pme_spline_order = 6;
    };

    // The chain state the oracle records: the equilibrated spin-resolved
    // density, its spin-summed and spin-difference reductions, the pre-loop
    // density the local-electron readout subtracts, the per-iteration
    // potential features and Fukui sources, and the local-electron energy.
    struct State {
        std::size_t nodes = 0, graphs = 0;
        std::vector<double> spin_charge_density;    // [nodes, charges]
        std::vector<double> charge_density;         // [nodes, density_width]
        std::vector<double> spin_density;           // [nodes, density_width]
        std::vector<double> pre_loop_density;       // [nodes, charges]
        std::vector<double> local_electron_energy;  // [nodes]
        std::vector<double> features_mixed;         // [nodes, node width]
        std::vector<std::vector<double>> potential_features;  // [steps][nodes, potential]
        std::vector<std::vector<double>> fukui_sources;       // [steps + 1][nodes, 2]
        std::vector<std::vector<double>> density_history;     // [steps + 1][nodes, charges]
    };

    struct ComposedForwardResult {
        State state;
        std::vector<double> energy;                    // [graphs]
        std::vector<double> local_electron_energy;     // [graphs]
        std::vector<double> electrostatic_energy;      // [graphs]
        std::vector<double> correction_energy;         // [graphs]
        std::vector<double> external_field_energy;     // [graphs]
        std::vector<double> dipole;                    // [graphs, 3]
    };

    // The seeds of the reverse pass.  Every entry is optional: an absent
    // (empty) one seeds nothing.
    struct Seeds {
        std::vector<double> local_electron_energy;  // [nodes]
        std::vector<double> spin_charge_density;    // [nodes, charges]
        // Per unrolled step, in forward order.
        std::vector<std::vector<double>> potential_features;
        std::vector<std::vector<double>> fukui_sources;  // [steps + 1]
        // The caller's own seeds of the shared cache's cos/sin tables (the
        // long-range energy path's adjoint, which fires over the same cache).
        // The chain adds them to its accumulators *before* its single
        // ``adjoint_geometry`` firing, so the position gradient composes both
        // paths exactly once -- the design note's shared-step rule.  Empty
        // seeds contribute nothing.
        std::vector<double> table_seed_cos;  // [n_k, nodes]
        std::vector<double> table_seed_sin;  // [n_k, nodes]
    };

    // The reverse pass's result: the seeds of the chain's own inputs, plus the
    // shared cache's table seeds (the geometry/cell reverse composes them, and
    // it stays with the long-range kernels).
    struct Adjoints {
        std::vector<double> node_features;      // [nodes, layers * width]
        std::vector<double> node_attrs;         // [nodes, num_elements]
        std::vector<double> positions;          // [nodes, 3]
        std::vector<double> external_potential; // [graphs, 4]
        std::vector<double> total_charge;       // [graphs]
        std::vector<double> total_spin;         // [graphs]
        std::vector<double> grad_cos_table;     // [n_k, nodes]
        std::vector<double> grad_sin_table;     // [n_k, nodes]
        // The chain's own cell (strain) adjoint at fixed cartesian positions:
        // the accumulated table, k-vector and per-k volume seeds of every
        // stage it runs, through the cache's cell-dependent outputs, plus the
        // corrective node fields' volume terms.  The affine half of the stress
        // is the position adjoint's (the family adds it), exactly as the
        // long-range kernels split them.
        std::vector<double> strain;             // [graphs, 3, 3]
    };

    struct ComposedReverseResult {
        Adjoints adjoints;
        std::vector<double> stress;             // [graphs, 3, 3]
    };

    PolarDeltaChainKokkosT() = default;
    PolarDeltaChainKokkosT(const nlohmann::json& data, const Contract& contract);

    bool ready() const { return ready_; }
    const Contract& contract() const { return contract_; }
    void set_particle_mesh_workspace(
        std::shared_ptr<PolarParticleMeshWorkspaceT<Precision>> workspace)
    {
        particle_mesh_workspace_ = std::move(workspace);
    }
    PolarParticleMeshUsage particle_mesh_usage() const
    {
#if defined(SYMMETRIX_PARTICLE_MESH_AVAILABLE) && SYMMETRIX_PARTICLE_MESH_AVAILABLE
        return particle_mesh_workspace_ ? particle_mesh_workspace_->usage()
                                        : PolarParticleMeshUsage{};
#else
        return {};
#endif
    }
    int interaction_count() const { return contract_.interaction_count; }
    int steps() const { return contract_.long_range.recursion_step_count; }
    int charges_dimension() const { return charges_dimension_; }
    int density_width() const { return density_width_; }
    int potential_dimension() const { return potential_dimension_; }
    int field_dimension() const { return field_dimension_; }
    std::vector<const E3LinearKokkosT<Precision>*> e3linear_modules() const;
    E3LinearDiagnostics e3linear_diagnostics() const;
    PotentialAdjointDiagnostics potential_adjoint_diagnostics() const
    {
        return potential_adjoint_diagnostics_;
    }
    GraphOutputDiagnostics graph_output_diagnostics() const
    {
        return graph_output_diagnostics_;
    }
    static PotentialAdjointExtents validate_potential_adjoint_extents(
        std::size_t nodes, std::size_t graphs);
    void set_realspace_reverse_strategy(const std::string_view strategy)
    {
        ops_.set_realspace_reverse_strategy(strategy);
    }
    void set_realspace_materialized_budget_bytes(const std::size_t bytes)
    {
        ops_.set_realspace_materialized_budget_bytes(bytes);
    }
    void set_realspace_full_materialized_gradients_required(const bool required)
    {
        ops_.set_realspace_full_materialized_gradients_required(required);
    }
    execution::long_range::RealspaceReverseStrategy
    realspace_reverse_strategy() const
    {
        return ops_.realspace_reverse_strategy();
    }
    execution::long_range::RealspaceReverseStrategyRequest
    realspace_reverse_strategy_request() const
    {
        return ops_.realspace_reverse_strategy_request();
    }
    std::string_view realspace_reverse_strategy_resolved() const
    {
        return ops_.realspace_reverse_strategy_resolved();
    }
    std::string_view realspace_reverse_strategy_reason() const
    {
        return ops_.realspace_reverse_strategy_reason();
    }

    // Drive the chain.  ``inputs`` must be coherent with the contract's
    // dimensions (the call validates and throws naming the offender).
    State forward(
        const Inputs& inputs, bool materialize_public_state = true);
    ComposedForwardResult forward_composed_energy(
        const Inputs& inputs, bool materialize_public_state = true,
        bool materialize_outputs = true);
    // The reverse pass over the state the last forward produced; the caller
    // supplies the seeds of the outputs it holds.  Both passes write the
    // object's own buffers: the forward fills the step histories, and the
    // reverse replays each step's update (so it leaves the blocks' intermediate
    // buffers holding the *last* replayed step's values).  Call ``forward``
    // again, on the same inputs, before reading any block-level scratch or
    // calling ``forward`` for another graph.
    Adjoints reverse(const Inputs& inputs, const Seeds& seeds,
        bool materialize_table_adjoints = true,
        const execution::long_range::LongRangeKernelMatrix<Precision>*
            initial_grad_cos_table = nullptr,
        const execution::long_range::LongRangeKernelMatrix<Precision>*
            initial_grad_sin_table = nullptr,
        bool materialize_learned_adjoints = true,
        bool use_resident_output_seeds = false,
        bool use_reference_potential_adjoint = false,
        bool materialize_input_adjoints = true);

    // Production reverse of the complete polar energy contribution.  The
    // energy, correction, dipole, and chain adjoints are scheduled here so
    // their shared phase-table seeds never cross any frontend boundary.
    // The returned table vectors are intentionally empty; the compact input,
    // position, and strain adjoints are the composed evaluator's outputs.
    ComposedReverseResult reverse_composed_energy(
        const Inputs& inputs, const State& state,
        bool materialize_learned_adjoints = true,
        bool materialize_input_adjoints = true);

    // The production reverse keeps this flat per-layer seed in the kernel
    // space.  Its owner can adopt it directly on the backbone reverse; only
    // diagnostic entry points request the host vector in ``Adjoints``.
    const Kokkos::View<Precision**, Kokkos::LayoutRight>&
    node_feature_seed() const { return node_feature_seed_; }
    const Kokkos::View<double**, Kokkos::LayoutRight>&
    position_adjoint_device() const { return public_position_seed_device_; }
    const Kokkos::View<double**, Kokkos::LayoutRight>&
    composed_stress_device() const { return composed_stress_device_; }
    const Kokkos::View<double**, Kokkos::LayoutRight>&
    composed_graph_outputs_device() const
    {
        return composed_graph_outputs_device_;
    }

    // --- the individual blocks, for the block-level gates ----------------
    // The delta blocks are driven block by block in the adjoint tests, so the
    // chain exposes them with their own forward/reverse rather than only
    // through the composed pass.
    const std::vector<E3LinearKokkosT<Precision>>& spin_source_heads() const
    {
        return spin_source_heads_;
    }
    const std::vector<bool>& spin_source_zero_charges() const
    {
        return spin_source_zero_charges_;
    }
    const std::vector<E3LinearKokkosT<Precision>>& feature_mixer() const
    {
        return feature_mixer_;
    }
    PolarNonLinearBiasReadoutKokkosT<Precision>& fukui_source_map()
    {
        return fukui_source_map_;
    }
    // The unrolled chain's update blocks: one per recursion step (the shipped
    // checkpoints serialize one per step, with their own weights), so the
    // accessor names the step the caller means.
    PolarFieldUpdateKokkosT<Precision>& field_update(
        const int step = 0)
    {
        if (step < 0
            || step >= static_cast<int>(field_updates_.size()))
            throw std::out_of_range(
                "the polar chain's field-update step is out of range.");
        return field_updates_[static_cast<std::size_t>(step)];
    }
    PolarLocalElectronReadoutKokkosT<Precision>& local_electron_readout()
    {
        return local_electron_readout_;
    }
    // The long-range operator group the chain drives with, so a caller can
    // read the same cache and kernels the chain used (the conformance-style
    // gates compare against them rather than against a second copy).
    const execution::long_range::PolarLongRangeOpsT<Precision>& long_range() const
    {
        return ops_;
    }
    const execution::long_range::LongRangeGeometryCacheT<Precision>& geometry_cache() const
    {
        return cache_;
    }
    static constexpr int long_range_scalar_bits()
    {
        return static_cast<int>(
            sizeof(typename decltype(cache_.cos_table)::value_type) * 8);
    }
    std::size_t workspace_nodes() const { return workspace_nodes_; }
    std::size_t workspace_graphs() const { return workspace_graphs_; }
    std::size_t workspace_n_k() const { return workspace_n_k_; }
    std::size_t workspace_replacement_count() const
    {
        return workspace_replacement_count_;
    }
    std::size_t input_h2d_bytes() const
    {
        const std::size_t long_range_bytes=ops_.input_h2d_bytes();
        if(input_h2d_bytes_
            >std::numeric_limits<std::size_t>::max()-long_range_bytes)
            throw std::overflow_error(
                "polar chain cumulative input byte count overflows size_t.");
        return input_h2d_bytes_+long_range_bytes;
    }
    std::size_t workspace_reuse_count() const { return workspace_reuse_count_; }
    std::size_t arena_view_allocation_count() const
    {
        return ops_.arena_view_allocation_count();
    }
    const execution::long_range::LongRangeSolverDecision& solver_decision() const
    {
        return solver_decision_;
    }
    std::size_t automatic_nodes() const { return automatic_nodes_; }
    std::size_t automatic_reciprocal_modes() const
    {
        return automatic_reciprocal_modes_;
    }
    bool automatic_reciprocal_modes_are_lower_bound() const
    {
        return automatic_reciprocal_modes_are_lower_bound_;
    }
    const std::array<std::int64_t, 3>& effective_pme_mesh() const
    {
        return current_inputs_pme_mesh_;
    }
    execution::long_range::LongRangeAllocationUsage
    reciprocal_cache_allocation_usage() const;
    execution::long_range::LongRangeAllocationUsage
    particle_mesh_adapter_allocation_usage() const;
    execution::long_range::LongRangeAllocationUsage
    reciprocal_arena_allocation_usage() const
    {
        if (solver_decision_.resolved
            == execution::long_range::LongRangeSolver::realspace)
            return {};
        return ops_.workspace_allocation_usage();
    }
    execution::long_range::LongRangeAllocationUsage
    realspace_cache_allocation_usage() const
    {
        return cache_.realspace_allocation_usage();
    }
    execution::long_range::LongRangeAllocationUsage
    realspace_arena_allocation_usage() const
    {
        if (solver_decision_.resolved
            != execution::long_range::LongRangeSolver::realspace)
            return {};
        return ops_.workspace_allocation_usage();
    }
    execution::long_range::LongRangeAllocationUsage
    realspace_retained_allocation_usage() const
    {
        auto usage = realspace_cache_allocation_usage();
        usage += realspace_arena_allocation_usage();
        return usage;
    }
    static constexpr std::size_t realspace_pair_tile()
    {
        return execution::long_range::PolarLongRangeOpsT<Precision>::
            realspace_pair_tile();
    }
    const execution::long_range::RealspaceEnergyLaunchDiagnostics&
    realspace_energy_launch_diagnostics() const
    {
        return ops_.realspace_energy_launch_diagnostics();
    }
    const execution::long_range::RealspaceReverseLaunchDiagnostics&
    realspace_reverse_launch_diagnostics() const
    {
        return ops_.realspace_reverse_launch_diagnostics();
    }
    execution::long_range::LongRangeAllocationUsage
    reciprocal_adjoint_allocation_usage() const;
    execution::long_range::LongRangeAllocationUsage
    reciprocal_allocation_usage() const
    {
        auto usage = reciprocal_cache_allocation_usage();
        usage += reciprocal_arena_allocation_usage();
        usage += reciprocal_adjoint_allocation_usage();
        return usage;
    }
    std::size_t diagnostic_state_materialization_count() const
    {
        return diagnostic_state_materialization_count_;
    }
    std::size_t resident_reverse_count() const
    {
        return resident_reverse_count_;
    }
    std::size_t uvu_workspace_replacement_count() const
    {
        std::size_t count = local_electron_readout_
            .uvu_workspace_replacement_count();
        for (const auto& update : field_updates_)
            count += update.uvu_workspace_replacement_count();
        return count;
    }
    static std::string workspace_execution_space()
    {
        return Kokkos::DefaultExecutionSpace::name();
    }
    // nvcc requires members enclosing extended device lambdas to be public.
    // These are native composition helpers, not host materialization APIs.
    void normalize_fukui_device(
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& raw,
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& normalized);
    void equilibrate_device(
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& sources,
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& state);
    void potential_features_device(
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& density,
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& output);
    void adjoint_normalize_fukui_device(
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& raw,
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& seed);
    void adjoint_equilibrate_device(
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& sources,
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& unequilibrated,
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& state_seed,
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& source_seed,
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& target_seed);
    void adjoint_potential_features_device(
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
        bool use_reference = false);
private:
    // The shared cache and the per-step buffers the orchestration needs; the
    // chain keeps them so the reverse pass sees exactly what the forward
    // computed (the cache is one per evaluation, shared by both channels).
    void prepare(const int nodes, const int graphs);
    void validate_inputs(const Inputs& inputs, int nodes, int graphs) const;
    void resolve_solver(const Inputs& inputs, int nodes, int graphs);
    std::array<std::int64_t, 3> automatic_pme_mesh(const Inputs& inputs) const;
    void release_reciprocal_state();
    void build_geometry_cache(const Inputs& inputs);
    void prepare_long_range_workspace();
    void record_input_h2d(
        std::size_t elements,std::size_t element_size,const char* what);
    // The electrostatic feature path of one spin channel: assemble, Coulomb-convolve,
    // project, subtract the self-interaction terms, add the analytic
    // corrections, and divide by the feature norms. Kernel-space tables,
    // workspaces, and GEMMs use Precision; the public diagnostic result is
    // widened to double at the host boundary.
    std::vector<double> feature_path(
        const std::vector<double>& channel_density,
        const std::vector<double>& corrections) const;
    // The per-channel analytic correction fields of one density state.
    void correction_fields(
        const std::vector<double>& density,
        std::vector<double>& alpha_out,
        std::vector<double>& beta_out) const;
    // The oracle's zero-guarded per-graph normalization of the Fukui sources.
    void normalize_fukui(
        std::vector<double>& sources, const Inputs& inputs) const;
    // The normalization's own adjoint: the forward divides each node's source
    // by its per-graph (zero-guarded) sum, so a seed on the *normalized*
    // sources becomes ``dy/norm - sum_j(dy_j x_j)/norm^2`` on the raw ones --
    // the map output the source kinds actually produce.
    void adjoint_normalize_fukui(
        const std::vector<double>& raw_sources,
        const std::vector<double>& normalized_seed,
        const std::vector<int>& node_graph_index,
        std::vector<double>& raw_seed) const;
    std::vector<double> channel_slice(
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& density,
        const int channel) const;
    std::vector<double> to_double(
        const Kokkos::View<Precision**, Kokkos::LayoutRight>& values) const;
    // The closed-form equilibration of one density state, and the two staging
    // steps that move an fp64 state vector into the Precision buffers the
    // learned blocks read.
    void equilibrate(
        const std::vector<double>& sources, const std::vector<double>& state,
        std::vector<double>& corrections_alpha,
        std::vector<double>& corrections_beta) const;
    void store_density(const std::vector<double>& state);
    void store_pre_loop_density(const std::vector<double>& state);
    void store_potential_features(const std::vector<double>& features);

    Contract contract_;
    bool ready_ = false;
    int charges_dimension_ = 0, density_width_ = 0, field_dimension_ = 0,
        potential_dimension_ = 0;
    int nodes_ = 0, graphs_ = 0;

    std::vector<E3LinearKokkosT<Precision>> spin_source_heads_;
    std::vector<bool> spin_source_zero_charges_;
    std::vector<E3LinearKokkosT<Precision>> feature_mixer_;
    PolarNonLinearBiasReadoutKokkosT<Precision> fukui_source_map_;
    std::vector<PolarFieldUpdateKokkosT<Precision>> field_updates_;
    PolarLocalElectronReadoutKokkosT<Precision> local_electron_readout_;

    execution::long_range::PolarLongRangeOpsT<Precision> ops_;
    execution::long_range::LongRangeGeometryCacheT<Precision> cache_;
    execution::long_range::LongRangeVector<double> norms_;

    // The graph geometry the forward ran on, retained in fp64 so the reverse
    // pass composes the long-range adjoints against exactly the tables it
    // built (and so the correction kinds see the same pbc rows).
    std::vector<double> positions_, cell_, external_potential_;
    std::vector<int> node_graph_index_, graph_pbc_;
    std::vector<double> total_charge_, total_spin_;
    std::vector<int> staged_node_types_;
    std::array<std::int64_t, 3> current_inputs_pme_mesh_{64, 64, 64};
    std::int64_t current_inputs_pme_spline_order_ = 6;

    // The chain's own scratch and long-range tables/accumulators follow the
    // evaluator's Precision. Scalar reductions inside long-range kernels may
    // retain fp64 accumulation where the numerical gates require it.
    Kokkos::View<Precision**, Kokkos::LayoutRight> node_features_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> node_attrs_;
    Kokkos::View<int*, Kokkos::LayoutRight> node_types_;
    // The chain state, resident in the default execution space and keyed by
    // step: the density the field update reads as its local charges and the
    // potential features it consumes, one entry per step.  Index 0 is the
    // pre-loop equilibration -- which is also the local-electron readout's
    // ``charges_0``, so the pre-loop density is not a second buffer -- and
    // index ``steps`` is the equilibrated state the readout consumes as its
    // ``charges_induced``.  Keying the state by step is what lets the reverse
    // replay each update from the forward's own device state instead of
    // re-uploading an fp64 host mirror of it.
    std::vector<Kokkos::View<Precision**, Kokkos::LayoutRight>> density_steps_;
    std::vector<Kokkos::View<Precision**, Kokkos::LayoutRight>>
        fukui_raw_steps_, fukui_steps_, source_steps_;
    std::vector<Kokkos::View<Precision**, Kokkos::LayoutRight>>
        potential_features_steps_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> features_mixed_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> head_row_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> mixer_row_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> fukui_sources_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> fukui_input_;
    // The source map's pair as a read-only operand of its own reverse: the
    // chain stages the raw pair's seed here rather than in a temporary view.
    Kokkos::View<Precision**, Kokkos::LayoutRight> fukui_seed_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> update_output_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> local_energy_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> node_feature_seed_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> graph_pair_sums_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> channel_density_,
        graph_multipoles_, barycenter_device_, external_potential_device_;
    Kokkos::View<int*> graph_node_counts_device_;
    Kokkos::View<int**, Kokkos::LayoutRight> graph_pbc_device_;
    Kokkos::View<Precision*, Kokkos::LayoutRight> field_norms_device_,
        si_feature_constants_device_;
    Kokkos::View<std::size_t*, Kokkos::LayoutRight>
        si_feature_select_device_;
    Kokkos::View<Precision*, Kokkos::LayoutRight> total_charge_device_,
        total_spin_device_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> heads_density_device_;
    // Complete-energy composition storage. The spin-summed final density and
    // every seed derived from it remain in the owning execution space. Graph
    // scalar outputs use fp64 storage so an FP32 model does not quantize the
    // final reduction before the public host boundary.
    Kokkos::View<Precision**, Kokkos::LayoutRight> composed_density_device_;
    Kokkos::View<double**, Kokkos::LayoutRight>
        composed_graph_multipoles_device_, composed_graph_outputs_device_,
        composed_graph_seeds_device_, composed_stress_device_,
        realspace_position_seed_device_, public_position_seed_device_;
    // The real external gauge shares the cache's FP64 Cartesian boundary;
    // direct reciprocal keeps using the historical Precision barycenter above.
    Kokkos::View<double**, Kokkos::LayoutRight> realspace_barycenter_device_;
    Kokkos::View<double*, Kokkos::LayoutRight> reporting_volumes_device_;
    Kokkos::View<double***, Kokkos::LayoutRight>
        composed_energy_strain_device_;
    Kokkos::View<Precision*, Kokkos::LayoutRight>
        si_energy_constants_device_, realspace_graph_energy_seed_device_;
    Kokkos::View<std::size_t*, Kokkos::LayoutRight>
        si_energy_select_device_;
    std::vector<Kokkos::View<Precision**, Kokkos::LayoutRight>>
        density_seed_steps_, fukui_seed_steps_, potential_seed_steps_;
    Kokkos::View<Precision**, Kokkos::LayoutRight> graph_reverse_sums_,
        position_seed_device_, external_seed_device_, target_seed_device_,
        correction_strain_device_, channel_seed_, projected_seed_,
        multipole_seed_, node_field_seed_, external_node_adjoint_,
        external_graph_adjoint_, correction_node_adjoint_,
        correction_graph_adjoint_;

    // Reverse and composed-energy scratch. These views are evaluator-owned,
    // replaced as one extent-keyed workspace, and cleared between uses.
    Kokkos::View<Precision**, Kokkos::LayoutRight> attrs_seed_, mixed_seed_,
        row_seed_, potential_seed_, charge_seed_, pre_loop_seed_, update_seed_,
        local_feature_seed_, energy_seed_;
    execution::long_range::LongRangeKernelMatrix<Precision>
        grad_cos_workspace_, grad_sin_workspace_, grad_k_workspace_;
    execution::long_range::LongRangeKernelVector<Precision>
        grad_volume_workspace_;
    std::size_t workspace_nodes_ = 0, workspace_graphs_ = 0,
        workspace_n_k_ = 0;
    std::size_t workspace_replacement_count_ = 0, workspace_reuse_count_ = 0;
    std::size_t input_h2d_bytes_ = 0;
    std::size_t automatic_nodes_ = 0, automatic_reciprocal_modes_ = 0;
    bool automatic_reciprocal_modes_are_lower_bound_ = false;
    std::size_t diagnostic_state_materialization_count_ = 0;
    std::size_t resident_reverse_count_ = 0;
    execution::long_range::LongRangeSolverDecision solver_decision_;
    std::shared_ptr<PolarParticleMeshWorkspaceT<Precision>>
        particle_mesh_workspace_;
    Kokkos::View<Precision**, Kokkos::LayoutRight>
        pme_forward_features_, pme_feature_seeds_, pme_source_adjoints_,
        pme_energy_features_, pme_energy_feature_seeds_,
        pme_energy_source_adjoints_;
    Kokkos::View<double**, Kokkos::LayoutRight> pme_position_adjoints_;
    Kokkos::View<double*, Kokkos::LayoutRight>
        pme_channel_energies_, pme_energy_seeds_, pme_cell_adjoints_,
        pme_energy_channel_energy_, pme_energy_seed_;
    PotentialAdjointDiagnostics potential_adjoint_diagnostics_;
    GraphOutputDiagnostics graph_output_diagnostics_;

    // Explicit diagnostic snapshots. Production never populates or consumes
    // these; it owns the corresponding typed step views above.
    std::vector<std::vector<double>> density_history_;
    std::vector<std::vector<double>> fukui_history_;
    std::vector<std::vector<double>> potential_features_history_;
};

extern template class NodeLocalUvuTensorProductKokkosT<float>;
extern template class NodeLocalUvuTensorProductKokkosT<double>;
extern template class PolarGateKokkosT<float>;
extern template class PolarGateKokkosT<double>;
extern template class PolarGatedReadoutKokkosT<float>;
extern template class PolarGatedReadoutKokkosT<double>;
extern template class PolarNonLinearBiasReadoutKokkosT<float>;
extern template class PolarNonLinearBiasReadoutKokkosT<double>;
extern template class PolarPotentialEmbeddingKokkosT<float>;
extern template class PolarPotentialEmbeddingKokkosT<double>;
extern template class PolarFieldUpdateKokkosT<float>;
extern template class PolarFieldUpdateKokkosT<double>;
extern template class PolarLocalElectronReadoutKokkosT<float>;
extern template class PolarLocalElectronReadoutKokkosT<double>;
extern template class PolarDeltaChainKokkosT<float>;
extern template class PolarDeltaChainKokkosT<double>;

} // namespace symmetrix::mace_polar
