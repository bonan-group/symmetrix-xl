#ifndef SYMMETRIX_D3_KOKKOS_HPP
#define SYMMETRIX_D3_KOKKOS_HPP

#include "dispersion_common.hpp"
#include "dispersion_graph_kokkos.hpp"
#include "dispersion_parameters.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <utility>

namespace symmetrix::dispersion {

enum class D3DampingForm { zero, bj, zerom, bjm };

template<class Precision>
class D3Kokkos {
public:
    D3Kokkos(
        std::string damping, std::string functional, double pair_cutoff,
        double coordination_cutoff);

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
    [[nodiscard]] std::array<double,6> interpolation_diagnostics(
        int receiver_atomic_number, int source_atomic_number,
        double receiver_coordination, double source_coordination) const;
    [[nodiscard]] double pair_energy_diagnostics(
        int receiver_atomic_number, int source_atomic_number, double distance,
        double receiver_coordination, double source_coordination) const;
    void compute_coordination_diagnostics(bool use_team);
    void compute_pair_diagnostics(
        bool use_team, bool compute_forces, bool compute_stress);
    void compute_adjoint_diagnostics(bool use_team);

    [[nodiscard]] const DispersionResultKokkos& result() const;
    [[nodiscard]] auto coordination_numbers() const
    {
        require_computed();
        return Kokkos::subview(
            coordination_numbers_storage_,
            std::make_pair(
                std::size_t(0),static_cast<std::size_t>(graph_.num_nodes())));
    }
    [[nodiscard]] auto energy_coordination_derivatives() const
    {
        require_computed();
        return Kokkos::subview(
            energy_coordination_derivatives_storage_,
            std::make_pair(
                std::size_t(0),static_cast<std::size_t>(graph_.num_nodes())));
    }
    [[nodiscard]] auto coordination_switch_values() const
    {
        require_computed();
        return Kokkos::subview(
            coordination_values_storage_,
            std::make_pair(std::size_t(0),graph_.num_edges()));
    }
    [[nodiscard]] auto coordination_radial_derivatives() const
    {
        require_computed();
        return Kokkos::subview(
            coordination_radial_derivatives_storage_,
            std::make_pair(std::size_t(0),graph_.num_edges()));
    }
    [[nodiscard]] std::size_t coordination_switch_capacity() const
    {
        require_computed();
        return coordination_values_storage_.extent(0);
    }
    [[nodiscard]] std::size_t reference_weight_active_atoms() const
    {
        return std::min(
            static_cast<std::size_t>(graph_.diagnostics().num_nodes),
            reference_weights_storage_.extent(0));
    }
    [[nodiscard]] std::size_t reference_weight_capacity_atoms() const
    {
        return reference_weights_storage_.extent(0);
    }
    [[nodiscard]] std::size_t reference_weight_workspace_bytes() const
    {
        return 2*reference_weights_storage_.span()*sizeof(Precision);
    }
    [[nodiscard]] std::size_t coordination_cache_active_edges() const
    {
        return std::min(
            static_cast<std::size_t>(graph_.diagnostics().num_edges),
            coordination_values_storage_.extent(0));
    }
    [[nodiscard]] std::size_t coordination_cache_capacity_edges() const
    {
        return coordination_values_storage_.extent(0);
    }
    [[nodiscard]] std::size_t coordination_cache_workspace_bytes() const
    {
        return 2*coordination_values_storage_.span()*sizeof(Precision);
    }
    [[nodiscard]] std::size_t d3_workspace_bytes() const
    {
        return reference_weight_workspace_bytes()
            +coordination_cache_workspace_bytes();
    }
    [[nodiscard]] std::size_t view_allocation_count() const
    {
        return graph_.diagnostics().view_allocation_count
            +workspace_allocation_count_;
    }
    [[nodiscard]] bool forces_available() const
    {
        return computed_ && forces_available_;
    }
    [[nodiscard]] bool stress_available() const
    {
        return computed_ && stress_available_;
    }
    [[nodiscard]] auto coordination_numbers_diagnostics() const
    {
        return Kokkos::subview(
            coordination_numbers_storage_,
            std::make_pair(
                std::size_t(0),static_cast<std::size_t>(graph_.num_nodes())));
    }
    [[nodiscard]] auto coordination_switch_values_diagnostics() const
    {
        return Kokkos::subview(
            coordination_values_storage_,
            std::make_pair(std::size_t(0),graph_.num_edges()));
    }
    [[nodiscard]] auto coordination_radial_derivatives_diagnostics() const
    {
        return Kokkos::subview(
            coordination_radial_derivatives_storage_,
            std::make_pair(std::size_t(0),graph_.num_edges()));
    }
    [[nodiscard]] auto atomic_energies_diagnostics() const
    {
        return Kokkos::subview(
            atomic_energies_storage_,
            std::make_pair(
                std::size_t(0),static_cast<std::size_t>(graph_.num_nodes())));
    }
    [[nodiscard]] auto energy_coordination_derivatives_diagnostics() const
    {
        return Kokkos::subview(
            energy_coordination_derivatives_storage_,
            std::make_pair(
                std::size_t(0),static_cast<std::size_t>(graph_.num_nodes())));
    }
    [[nodiscard]] auto forces_diagnostics() const
    {
        return Kokkos::subview(
            forces_storage_,
            std::make_pair(
                std::size_t(0),static_cast<std::size_t>(graph_.num_nodes())),
            Kokkos::ALL());
    }
    [[nodiscard]] auto virial_diagnostics() const
    {
        return Kokkos::subview(
            virial_storage_,
            std::make_pair(
                std::size_t(0),static_cast<std::size_t>(graph_.num_nodes())),
            Kokkos::ALL());
    }
    [[nodiscard]] std::size_t coordination_team_launch_count() const
    {
        return coordination_team_launch_count_;
    }
    [[nodiscard]] std::size_t coordination_serial_fallback_launch_count() const
    {
        return coordination_serial_fallback_launch_count_;
    }
    [[nodiscard]] std::size_t stress_reduction_launch_count() const
    {
        return stress_reduction_launch_count_;
    }
    [[nodiscard]] const char* coordination_production_path() const
    {
        switch (coordination_production_path_) {
        case CoordinationProductionPath::not_run:
            return "not_run";
        case CoordinationProductionPath::team:
            return "team";
        case CoordinationProductionPath::serial_fallback:
            return "serial_fallback";
        }
        return "unknown";
    }
    [[nodiscard]] DispersionGraphDiagnostics graph_diagnostics() const
    {
        return graph_.diagnostics();
    }

private:
    enum class CoordinationProductionPath { not_run, team, serial_fallback };

    using ParameterView = Kokkos::View<Precision*>;
    using CountView = Kokkos::View<int*>;
    using ReferenceWorkspaceView =
        Kokkos::View<
            Precision*[d3_max_coordination_references],Kokkos::LayoutRight>;
    using AtomicEnergyView = Kokkos::View<double*>;
    using ForceView = Kokkos::View<double*[3]>;
    using VirialView = Kokkos::View<double*[6]>;

    void require_computed() const;
    void validate_build_cutoff(double build_cutoff) const;
    void validate_atomic_numbers(std::span<const int> atomic_numbers) const;
    void ensure_workspace_capacity(std::size_t nodes, std::size_t edges);
    void compute_coordination_team();
    void compute_coordination_serial_oracle();
    void compute_reference_weights();
    void compute_pair_team(bool compute_forces, bool compute_stress);
    void compute_pair_serial_oracle(bool compute_forces, bool compute_stress);
    void compute_adjoint_team(bool compute_forces, bool compute_stress);
    void compute_adjoint_serial_oracle(
        bool compute_forces, bool compute_stress);

    DispersionGraphKokkos graph_;
    std::string damping_;
    std::string functional_;
    D3DampingForm damping_form_ = D3DampingForm::zero;
    double pair_cutoff_ = 0.0;
    double coordination_cutoff_ = 0.0;
    DampingParameters parameters_{};
    ParameterView c6_;
    ParameterView r0_;
    ParameterView rcov_;
    ParameterView r2r4_;
    CountView reference_counts_;
    ParameterView reference_coordination_;
    ParameterView coordination_numbers_storage_;
    ParameterView energy_coordination_derivatives_storage_;
    ParameterView coordination_values_storage_;
    ParameterView coordination_radial_derivatives_storage_;
    ReferenceWorkspaceView reference_weights_storage_;
    ReferenceWorkspaceView reference_derivatives_storage_;
    AtomicEnergyView atomic_energies_storage_;
    ForceView forces_storage_;
    VirialView virial_storage_;
    DispersionResultKokkos result_;
    std::size_t coordination_team_launch_count_ = 0;
    std::size_t coordination_serial_fallback_launch_count_ = 0;
    std::size_t stress_reduction_launch_count_ = 0;
    CoordinationProductionPath coordination_production_path_ =
        CoordinationProductionPath::not_run;
    std::size_t workspace_allocation_count_ = 0;
    bool computed_ = false;
    bool forces_available_ = false;
    bool stress_available_ = false;
};

extern template class D3Kokkos<double>;
extern template class D3Kokkos<float>;

} // namespace symmetrix::dispersion

#endif
