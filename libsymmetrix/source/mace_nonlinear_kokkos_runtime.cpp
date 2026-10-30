#define SYMMETRIX_MACE_NONLINEAR_KOKKOS_PART 2
#include "mace_nonlinear_kokkos_impl.tpp"

#include "mace_polar_kokkos.hpp"

// The family-adapter conformance gate: this translation unit instantiates the
// evaluator, so a missing or mistyped interface member fails the library build
// here rather than at a consumer.
static_assert(symmetrix::execution::FamilyEvaluator<MaceNonlinearKokkos>);
static_assert(symmetrix::execution::FamilyEvaluator<MaceNonlinearFloatKokkos>);
static_assert(symmetrix::execution::TieredFamilyEvaluator<MaceNonlinearKokkos>);
static_assert(
    symmetrix::execution::TieredFamilyEvaluator<MaceNonlinearFloatKokkos>);
// The PolarMACE family is a sibling evaluator over the internal backbone, and
// must satisfy the same neutral family-adapter contract without inheriting the
// public nonlinear family.
static_assert(!std::is_base_of_v<MaceNonlinearKokkos,MacePolarKokkos>);
static_assert(
    !std::is_base_of_v<MaceNonlinearFloatKokkos,MacePolarFloatKokkos>);
static_assert(symmetrix::execution::FamilyEvaluator<MacePolarKokkos>);
static_assert(symmetrix::execution::FamilyEvaluator<MacePolarFloatKokkos>);
static_assert(symmetrix::execution::TieredFamilyEvaluator<MacePolarKokkos>);
static_assert(symmetrix::execution::TieredFamilyEvaluator<MacePolarFloatKokkos>);
