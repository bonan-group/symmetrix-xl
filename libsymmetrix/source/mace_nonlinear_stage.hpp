#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The MACE_Nonlinear (MH-1) execution stage description.
//
// The interaction-layer count, the profiling region name of every layer's R and
// M forward and reverse stage, and the ordered forward and reverse execution
// schedules are DATA.  A description is built from a layer count and names the
// canonical regions for that count, so an evaluator that admits a different
// topology profiles and dispatches it through the same value instead of a
// compiled-in two-layer table.  The description is family-owned state, not a
// shared-layer type: it names this family's regions and this family's contract
// vocabulary.
//
// The region spelling is the established one the ROCm profiler analyzer decodes
// (`symmetrix/mh1/<R|M><layer>/<forward|reverse>`), so a description built for
// any count keeps the profile contract readable.

namespace symmetrix::mace_nonlinear {

//: One ordered entry of a forward or reverse execution schedule.  The
//: vocabulary is the MH-1 v4 contract's: a generated node program names a
//: phase, and an edge tensor product names the layer it runs for.  A schedule
//: is data, so the orchestration it drives cannot drift from the contract that
//: declares it.
enum class StageKind {
    node_program,
    edge_tensor_product,
    edge_tensor_product_reverse,
};

//: A per-layer node-program phase, in the order a layer's forward descriptor
//: stores it: the phase that runs before the layer's edge tensor product, then
//: the phase that finishes the layer.  The forward schedule fixes that order,
//: so the descriptor ordinal is read from the stage vocabulary rather than
//: written as a literal index at every dispatch site.
enum class ForwardPhase {
    prepare = 0,
    finish = 1,
};

//: A per-layer reverse node-program phase, in descriptor order: the phase that
//: consumes the layer's tensor-product adjoint, then the phase that propagates
//: the layer's input adjoint.
enum class ReversePhase {
    post_reverse = 0,
    pre_reverse = 1,
};

struct Stage {
    StageKind kind = StageKind::node_program;
    //: The contract's phase name for a node-program entry; empty otherwise.
    std::string phase;
    //: The layer a tensor-product entry runs for; -1 for a node program.
    int layer = -1;
};

class StageDescription {
public:
    //: The canonical description of `layer_count` interaction layers: every
    //: region name follows the established spelling, and the schedules are the
    //: contract's own — the same entries, in the same order, with the same
    //: phase names the contract generates for this count.  The forward schedule
    //: gives every layer a node program before its edge tensor product and one
    //: after it, and the reverse schedule runs from the last layer back to the
    //: first, so a schedule-walking orchestration reads the program the
    //: contract declares rather than a second ordering of it.
    static StageDescription canonical(const std::size_t layer_count)
    {
        StageDescription description;
        description.layer_count_ = layer_count;
        description.r_forward_.reserve(layer_count);
        description.m_forward_.reserve(layer_count);
        description.r_reverse_.reserve(layer_count);
        description.m_reverse_.reserve(layer_count);
        description.forward_schedule_.reserve(2 * layer_count + 1);
        description.reverse_schedule_.reserve(2 * layer_count);
        for (std::size_t layer = 0; layer < layer_count; ++layer) {
            description.r_forward_.push_back(
                region(RegionOwner::r, layer, "forward"));
            description.m_forward_.push_back(
                region(RegionOwner::m, layer, "forward"));
            description.r_reverse_.push_back(
                region(RegionOwner::r, layer, "reverse"));
            description.m_reverse_.push_back(
                region(RegionOwner::m, layer, "reverse"));
        }
        if (layer_count == 0)
            return description;

        // Forward: layer 0 is prepared first, each layer's edge tensor product
        // follows, and the phase that finishes a layer is the phase that
        // prepares its successor — so the interior entries carry both names,
        // exactly as the contract spells them.  The last layer's finish phase
        // closes the schedule.
        description.forward_schedule_.push_back(
            node_program_stage("prepare_layer_0"));
        for (std::size_t layer = 0; layer < layer_count; ++layer) {
            description.forward_schedule_.push_back(tensor_product_stage(layer));
            if (layer + 1 < layer_count)
                description.forward_schedule_.push_back(node_program_stage(
                    "finish_layer_" + std::to_string(layer)
                    + "_prepare_layer_" + std::to_string(layer + 1)));
        }
        description.forward_schedule_.push_back(node_program_stage(
            "finish_layer_" + std::to_string(layer_count - 1)));

        // Reverse: the last layer is reversed first and the schedule walks back
        // to layer 0, which is the order the evaluator's reverse loop runs in.
        // The first entry names the last layer's reversal; every later entry
        // propagates a layer's adjoint into its predecessor.
        for (std::size_t position = 0; position < layer_count; ++position) {
            const std::size_t layer = layer_count - 1 - position;
            description.reverse_schedule_.push_back(node_program_stage(
                position == 0
                    ? "reverse_layer_" + std::to_string(layer)
                    : "propagate_layer_" + std::to_string(layer + 1)
                        + "_reverse_layer_" + std::to_string(layer)));
            description.reverse_schedule_.push_back(
                tensor_product_reverse_stage(layer));
        }
        return description;
    }

    std::size_t layer_count() const noexcept { return layer_count_; }

    //: Bounds-checked layer index.  A topology the description does not
    //: describe is a named error rather than an out-of-range table read.
    std::size_t layer_index(const std::size_t layer) const
    {
        if (layer >= layer_count_)
            throw std::out_of_range(
                "MACE_Nonlinear stage description describes "
                + std::to_string(layer_count_) + " interaction layers; layer "
                + std::to_string(layer) + " is out of range.");
        return layer;
    }

    std::string_view r_forward(const std::size_t layer) const
    {
        return r_forward_.at(layer_index(layer));
    }
    std::string_view m_forward(const std::size_t layer) const
    {
        return m_forward_.at(layer_index(layer));
    }
    std::string_view r_reverse(const std::size_t layer) const
    {
        return r_reverse_.at(layer_index(layer));
    }
    std::string_view m_reverse(const std::size_t layer) const
    {
        return m_reverse_.at(layer_index(layer));
    }

    const std::vector<Stage>& forward_schedule() const noexcept
    {
        return forward_schedule_;
    }
    const std::vector<Stage>& reverse_schedule() const noexcept
    {
        return reverse_schedule_;
    }

    //: Number of node-program phases a layer's forward descriptor carries,
    //: read from the phase vocabulary the forward schedule fixes.
    static constexpr std::size_t forward_phase_count() noexcept
    {
        return static_cast<std::size_t>(ForwardPhase::finish) + 1;
    }
    static constexpr std::size_t reverse_phase_count() noexcept
    {
        return static_cast<std::size_t>(ReversePhase::pre_reverse) + 1;
    }

    //: The descriptor phase ordinal `phase` occupies in a layer's forward
    //: phase table.
    static constexpr std::size_t forward_phase_index(
        const ForwardPhase phase) noexcept
    {
        return static_cast<std::size_t>(phase);
    }
    static constexpr std::size_t reverse_phase_index(
        const ReversePhase phase) noexcept
    {
        return static_cast<std::size_t>(phase);
    }

private:
    enum class RegionOwner { r, m };

    static std::string region(
        const RegionOwner owner, const std::size_t layer,
        const std::string_view direction)
    {
        std::string name = "symmetrix/mh1/";
        name.push_back(owner == RegionOwner::r ? 'R' : 'M');
        name += std::to_string(layer);
        name.push_back('/');
        name += direction;
        return name;
    }

    static Stage node_program_stage(std::string phase)
    {
        Stage stage;
        stage.kind = StageKind::node_program;
        stage.phase = std::move(phase);
        return stage;
    }

    static Stage tensor_product_stage(const std::size_t layer)
    {
        Stage stage;
        stage.kind = StageKind::edge_tensor_product;
        stage.layer = static_cast<int>(layer);
        return stage;
    }

    static Stage tensor_product_reverse_stage(const std::size_t layer)
    {
        Stage stage;
        stage.kind = StageKind::edge_tensor_product_reverse;
        stage.layer = static_cast<int>(layer);
        return stage;
    }

    std::size_t layer_count_ = 0;
    std::vector<std::string> r_forward_;
    std::vector<std::string> m_forward_;
    std::vector<std::string> r_reverse_;
    std::vector<std::string> m_reverse_;
    std::vector<Stage> forward_schedule_;
    std::vector<Stage> reverse_schedule_;
};

}  // namespace symmetrix::mace_nonlinear
