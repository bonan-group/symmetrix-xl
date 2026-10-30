#pragma once

// MH-1 direct-execution plan for the shared Kokkos tensor-product class.
//
// E3TensorProductKokkosT (e3nn_kokkos.hpp) keeps only an opaque handle to this
// type: `std::shared_ptr<DirectExecutionPlan> direct_execution_plan_`.  The
// plan layout, the admission rules that install it, and every kernel that
// consumes it live in the owning family layer, so the shared header and the
// OAM owners that include it carry no MH-1 state.
//
// This header is consumed only by mace_nonlinear_mh1_fast_path.cpp, which
// defines the corresponding E3TensorProductKokkosT members.

#include "e3nn_kokkos.hpp"

template<typename Precision>
struct E3TensorProductKokkosT<Precision>::DirectExecutionPlan {
    // Plan extents.  mh1_instruction_count is the number of admitted
    // instructions, mh1_multiplicity the shared channel multiplicity,
    // mh1_input_1_angular_dimension the number of input-1 angular components,
    // and mh1_max_output_width the widest output component block.
    int mh1_instruction_count=0;
    int mh1_multiplicity=0;
    int mh1_input_1_angular_dimension=0;
    int mh1_max_output_width=0;
    // CUDA team sizes derived from the channel multiplicity.
    int mh1_cuda_channel_team_size=0;
    int mh1_cuda_harmonic_team_size=0;
    // True when every instruction carries the same channel multiplicity and
    // the input-1 irreps expose one (offset, width) pair per angular
    // component, so the direct node kernels can address lanes directly.
    bool mh1_direct_node_layout=false;
    // Flattened plan: per instruction the input-1 offset, input-2 offset,
    // output offset, input width, output width, weight offset, and the base of
    // its component offsets.
    Kokkos::View<int**,Kokkos::LayoutRight> mh1_instruction_data,
        mh1_sparse_indices,mh1_harmonic_terms,mh1_input_component_data,
        mh1_input_terms,mh1_input_instruction_term_offsets;
    Kokkos::View<int*> mh1_component_offsets,mh1_harmonic_offsets,
        mh1_input_term_offsets,mh1_input_1_mul_to_ir,mh1_output_mul_to_ir;
    Kokkos::View<Precision*> mh1_path_weights,mh1_sparse_values,
        mh1_harmonic_values,mh1_input_values,mh1_output_mask_ir_mul;
};
