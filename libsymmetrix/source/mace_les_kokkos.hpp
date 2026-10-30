#pragma once

#include "mace_les_kokkos_state.hpp"

template <typename Precision>
class MACELESKokkos final : public MACEKokkos<Precision> {
public:
    MACELESKokkos(
        std::string filename,
        std::string requested_head = {});
    ~MACELESKokkos() override;

    bool is_mace_les() const { return true; }
    int validate_les_position_count(std::size_t position_count) const;
    void set_les_geometry(
        std::span<const double> positions,
        std::span<const double> cell,
        std::span<const int> pbc);
    void configure_les_particle_mesh_for_testing(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t spline_order);
    void use_explicit_les_periodic_for_testing();
    void fence_les_particle_mesh_owner(const char* label);
    static std::tuple<
        bool, bool, std::string, std::uint32_t, std::uint32_t, std::uint64_t>
    les_particle_mesh_stream_probe_for_testing();

    void initialize_les_readouts();
    void compute_les_charges(int num_nodes);
    void accumulate_les_charge_readout_adjoints(int num_nodes);
    void prepare_les_charge_readouts(int num_nodes);
    void evaluate_les_term(int num_nodes);
    void enqueue_les_energy_copy();
    void publish_les_energy();
    void add_les_explicit_forces(int num_nodes);
    void add_les_explicit_stress(double volume);
    std::size_t les_charge_workspace_active_bytes() const;
    std::size_t les_charge_workspace_capacity_bytes() const;

    MACELESState<Precision> les_;

protected:
    void begin_model_evaluation(int num_nodes) override;
    void prepare_model_readouts(int num_nodes) override;
    void evaluate_model_after_readouts(int num_nodes) override;
    void before_model_evaluation_fence() override;
    void after_model_evaluation_fence() override;
    void finish_model_evaluation(int num_nodes) override;
    void compose_model_forces(int num_nodes) override;
    void compose_model_stress(double volume) override;
    void validate_model_capability(
        MACEModelCapability capability) const override;
    void validate_model_execution_mode(ExecutionMode mode) const override;
    void on_model_execution_mode_change(
        ExecutionMode previous,
        ExecutionMode next) override;

private:
    explicit MACELESKokkos(const SelectedMACEModel& selected);
};
