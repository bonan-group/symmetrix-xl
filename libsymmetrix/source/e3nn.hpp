#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

// Minimal, runtime-independent implementation of the e3nn pieces used by
// MACE_Nonlinear.  The serialized model carries all Wigner tensors, so this
// code deliberately has no e3nn or Python dependency.
struct IrrepBlock {
    int multiplicity;
    int l;
    int parity;
    int offset;
    int dimension() const { return multiplicity * (2*l + 1); }
};
class Irreps {
public:
    explicit Irreps(const std::string& specification);
    int dimension() const { return dimension_; }
    std::vector<IrrepBlock> blocks;

private:
    int dimension_ = 0;
};

struct E3Instruction {
    struct WignerEntry {
        int a;
        int b;
        int c;
        double value;
    };
    int input_1 = -1;
    int input_2 = -1;
    int output = -1;
    std::string mode;
    bool has_weight = false;
    double path_weight = 1.0;
    std::vector<int> path_shape;
    std::vector<double> wigner_3j;
    std::vector<int> wigner_shape;
    std::vector<WignerEntry> nonzero_wigner;
    int weight_offset = 0;
};

struct E3LinearBatchWorkspace {
    std::vector<double> packed_input;
    std::vector<double> packed_output;
};

class E3Linear {
public:
    explicit E3Linear(const nlohmann::json& data);
    int input_dimension() const { return input.dimension(); }
    int output_dimension() const { return output.dimension(); }
    std::vector<double> evaluate(const std::vector<double>& x) const;
    void reverse(const std::vector<double>& output_adj, std::vector<double>& input_adj) const;
    void evaluate_batch(
        const std::vector<double>& input_values,
        int samples,
        std::vector<double>& output_values,
        E3LinearBatchWorkspace& workspace) const;
    void reverse_batch(
        const std::vector<double>& output_adjoint,
        int samples,
        std::vector<double>& input_adjoint,
        E3LinearBatchWorkspace& workspace) const;

    Irreps input;
    Irreps output;
    std::vector<E3Instruction> instructions;
    std::vector<double> weights;
    std::vector<double> bias;
    std::vector<double> output_mask;
};

class E3TensorProduct {
public:
    explicit E3TensorProduct(const nlohmann::json& data);
    int input_1_dimension() const { return input_1.dimension(); }
    int input_2_dimension() const { return input_2.dimension(); }
    int output_dimension() const { return output.dimension(); }

    std::vector<double> evaluate(
        const std::vector<double>& x1,
        const std::vector<double>& x2,
        const std::vector<double>& weights = {}) const;
    void reverse(
        const std::vector<double>& x1,
        const std::vector<double>& x2,
        const std::vector<double>& weights,
        const std::vector<double>& output_adj,
        std::vector<double>& input_1_adj,
        std::vector<double>& input_2_adj,
        std::vector<double>& weights_adj) const;

    Irreps input_1;
    Irreps input_2;
    Irreps output;
    std::vector<E3Instruction> instructions;
    std::vector<double> internal_weights;
    std::vector<double> output_mask;
    int weight_numel = 0;
};
