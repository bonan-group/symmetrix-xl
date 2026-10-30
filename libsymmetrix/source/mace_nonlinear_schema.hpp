#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

//: The backbone-rule selector of the shared nonlinear backbone schema.  The
//: MH-1 family's models and the PolarMACE family's models are the same
//: backbone; each family's evaluator validates its own files, and they differ
//: in exactly the spellings the polar schema module documents (the model type,
//: the ``scale_shift`` shape, and the ``atomic_energies`` shape).  The
//: selector carries no rules: it selects which family's rule set the shared
//: validator applies, so the structure both families serialize is validated in
//: one place and cannot drift.
enum class MaceBackboneSchema {
    mh1,
    polar,
};

//: Validate a backbone JSON against one family's rule set.
void validate_mace_backbone_schema(
    const nlohmann::json& data, MaceBackboneSchema schema);

//: The MH-1 family's own entry point: the shared validator under the MH-1
//: rules, unchanged from the single-family form.
void validate_mace_nonlinear_schema(const nlohmann::json& data);

struct MaceMH1IrrepBlockPattern {
    int multiplicity = 0;
    int l = 0;
    int parity = 1;
};

enum class MaceMH1FinalOutputPolicy {
    not_final,
    scalar,
    keep_last,
};

const char* mace_mh1_final_output_policy_name(
    MaceMH1FinalOutputPolicy policy);

//: One interaction's independently parsed MH-1 execution topology.  Keeping
//: the edge and product patterns here prevents later program construction from
//: rebuilding a layer's dimensions from another interaction.
struct MaceMH1LayerDescriptor {
    std::size_t interaction_index = 0;
    int uvu_channels = 0;
    std::vector<MaceMH1IrrepBlockPattern> edge_blocks;
    std::vector<MaceMH1IrrepBlockPattern> product_output_blocks;
    MaceMH1FinalOutputPolicy final_output_policy =
        MaceMH1FinalOutputPolicy::not_final;
};

//: The MH-1 node-program support matrix over a parsed backbone.  It describes
//: the *generated* node program's admissible topologies, not the family's
//: schema.  The matrix admits the explicit polar keep-last scalar-plus-vector
//: output policy while preserving the scalar-final rule for ordinary MH-1.
struct MaceMH1FamilyDescriptor {
    bool compatible = false;
    int l_max = 0;
    int radial_size = 0;
    int node_channels = 0;
    std::vector<MaceMH1LayerDescriptor> layers;
    std::string rejection_reason;

    std::size_t layer_count() const { return layers.size(); }
    const MaceMH1LayerDescriptor& layer(const std::size_t index) const {
        if (index >= layers.size())
            throw std::out_of_range("MH-1 layer index is out of range.");
        return layers[index];
    }
    // Compatibility query retained until all native program consumers migrate
    // to the indexed descriptor.  Its name makes the old first-layer meaning
    // explicit instead of presenting that value as a family-wide invariant.
    int first_layer_edge_channels() const {
        if (layers.empty() || layers.front().edge_blocks.empty()) return 0;
        return layers.front().edge_blocks.front().multiplicity;
    }
};

MaceMH1FamilyDescriptor analyze_mh1_family_architecture(
    const nlohmann::json& data, MaceBackboneSchema schema);
