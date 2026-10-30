#pragma once

template <typename Precision>
void MACEKokkos<Precision>::begin_model_evaluation(int)
{}

template <typename Precision>
void MACEKokkos<Precision>::prepare_model_readouts(int)
{}

template <typename Precision>
void MACEKokkos<Precision>::evaluate_model_after_readouts(int)
{}

template <typename Precision>
void MACEKokkos<Precision>::before_model_evaluation_fence()
{}

template <typename Precision>
void MACEKokkos<Precision>::after_model_evaluation_fence()
{}

template <typename Precision>
void MACEKokkos<Precision>::finish_model_evaluation(int)
{}

template <typename Precision>
void MACEKokkos<Precision>::compose_model_forces(int)
{}

template <typename Precision>
void MACEKokkos<Precision>::compose_model_stress(double)
{}

template <typename Precision>
void MACEKokkos<Precision>::validate_model_capability(MACEModelCapability) const
{}

template <typename Precision>
void MACEKokkos<Precision>::validate_model_execution_mode(ExecutionMode) const
{}

template <typename Precision>
void MACEKokkos<Precision>::on_model_execution_mode_change(
    ExecutionMode,
    ExecutionMode)
{}
