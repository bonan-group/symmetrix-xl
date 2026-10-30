#ifndef SYMMETRIX_D2_KOKKOS_HPP
#define SYMMETRIX_D2_KOKKOS_HPP

#include "dispersion_common.hpp"
#include "dispersion_graph_kokkos.hpp"
#include "dispersion_parameters.hpp"

#include <Kokkos_Core.hpp>

#include <span>
#include <string>

namespace symmetrix::dispersion {

template<class Precision>
class D2Kokkos {
public:
    D2Kokkos(std::string functional, double pair_cutoff);

    void prepare_periodic_graph(
        std::span<const int> atomic_numbers,
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const double> inverse_cell,
        double build_cutoff);
    void prepare_host_graph(
        std::span<const int> atomic_numbers,
        std::span<const int> receiver_offsets,
        std::span<const int> sources,
        std::span<const int> shifts,
        std::span<const double> reference_positions,
        std::span<const double> cell,
        double build_cutoff);
    void update_geometry(
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const double> inverse_cell);
    void compute(bool compute_forces, bool compute_stress);

    [[nodiscard]] const DispersionResultKokkos& result() const;
    [[nodiscard]] DispersionGraphDiagnostics graph_diagnostics() const
    {
        return graph_.diagnostics();
    }
    [[nodiscard]] bool forces_available() const
    {
        return computed_ && forces_available_;
    }
    [[nodiscard]] bool stress_available() const
    {
        return computed_ && stress_available_;
    }

private:
    using ParameterView = Kokkos::View<Precision*>;
    using AtomicEnergyView = Kokkos::View<double*>;
    using ForceView = Kokkos::View<double*[3]>;
    using VirialView = Kokkos::View<double*[6]>;

    void validate_build_cutoff(double build_cutoff) const;
    void validate_atomic_numbers(std::span<const int> atomic_numbers) const;
    void ensure_result_capacity(std::size_t nodes);

    DispersionGraphKokkos graph_;
    std::string functional_;
    double pair_cutoff_ = 0.0;
    DampingParameters parameters_{};
    ParameterView r0_;
    ParameterView c6_;
    AtomicEnergyView atomic_energies_storage_;
    ForceView forces_storage_;
    VirialView virial_storage_;
    DispersionResultKokkos result_;
    bool computed_ = false;
    bool forces_available_ = false;
    bool stress_available_ = false;
};

extern template class D2Kokkos<double>;
extern template class D2Kokkos<float>;

} // namespace symmetrix::dispersion

#endif
