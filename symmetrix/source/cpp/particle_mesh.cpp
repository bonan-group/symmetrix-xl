// Private qualification seam for the physics-neutral particle-mesh transport.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "particle_mesh_transport.hpp"
#include "particle_mesh_strain.hpp"
#include "mace_polar_particle_mesh.hpp"

namespace py = pybind11;

namespace {

constexpr std::string_view k_float32 = "float32";
constexpr std::string_view k_float64 = "float64";

template <typename Value>
using c_array = py::array_t<Value, py::array::c_style | py::array::forcecast>;

template <typename Value>
[[nodiscard]] py::array_t<Value> wrap(
    const std::vector<Value>& values, const std::vector<py::ssize_t>& shape)
{
    py::ssize_t expected = 1;
    for (const py::ssize_t extent : shape) {
        expected *= extent;
    }
    if (expected != static_cast<py::ssize_t>(values.size())) {
        throw std::logic_error("Particle-mesh result shape does not match its storage.");
    }
    py::array_t<Value> result(shape);
    if (!values.empty()) {
        std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(Value));
    }
    return result;
}

template <typename Float>
class TransportImpl {
public:
    using Workspace = symmetrix::particle_mesh::ParticleMeshWorkspace<Float>;
    using complex_type = Kokkos::complex<Float>;

    explicit TransportImpl(const std::int64_t channel_bound)
        : execution_space_(),
          workspace_(
              symmetrix::particle_mesh::MeshLayout::right, channel_bound,
              execution_space_)
    {
    }

    void reserve(
        const std::int64_t nx,
        const std::int64_t ny,
        const std::int64_t nz,
        const std::int64_t particles,
        const std::int64_t lanes,
        const std::int64_t spline_order)
    {
        workspace_.reserve(nx, ny, nz, particles, lanes, spline_order);
        const std::int64_t coordinates = symmetrix::particle_mesh::detail::checked_product(
            particles, 3, "particle-mesh probe position coordinates");
        const std::int64_t weight_elements =
            symmetrix::particle_mesh::detail::checked_product(
                particles, lanes, "particle-mesh probe weight elements");
        positions_ = Kokkos::View<double*>(
            "symmetrix_particle_mesh_probe_positions", coordinates);
        weights_ = Kokkos::View<Float*>(
            "symmetrix_particle_mesh_probe_weights", weight_elements);
        weight_adjoints_ = Kokkos::View<Float*>(
            "symmetrix_particle_mesh_probe_weight_adjoints", weight_elements);
        position_forces_ = Kokkos::View<Float*>(
            "symmetrix_particle_mesh_probe_position_forces", coordinates);
        nx_ = nx;
        ny_ = ny;
        nz_ = nz;
        particles_ = particles;
        lanes_ = lanes;
        reserved_ = true;
        cell_set_ = false;
        positions_set_ = false;
    }

    void set_cell(const c_array<double>& cell)
    {
        require_reserved();
        if (cell.size() != 9) {
            throw std::invalid_argument("Particle-mesh cell must hold nine entries.");
        }
        std::array<double, 9> values{};
        std::copy(cell.data(), cell.data() + 9, values.begin());
        for (const double value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Particle-mesh cell entries must be finite.");
            }
        }
        const double determinant =
            values[0] * (values[4] * values[8] - values[5] * values[7])
            - values[1] * (values[3] * values[8] - values[5] * values[6])
            + values[2] * (values[3] * values[7] - values[4] * values[6]);
        if (!std::isfinite(determinant) || !(std::abs(determinant) > 0.0)) {
            throw std::invalid_argument("Particle-mesh cell must be finite and nonsingular.");
        }
        std::array<double, 9> inverse{};
        inverse[0] = (values[4] * values[8] - values[5] * values[7]) / determinant;
        inverse[1] = (values[2] * values[7] - values[1] * values[8]) / determinant;
        inverse[2] = (values[1] * values[5] - values[2] * values[4]) / determinant;
        inverse[3] = (values[5] * values[6] - values[3] * values[8]) / determinant;
        inverse[4] = (values[0] * values[8] - values[2] * values[6]) / determinant;
        inverse[5] = (values[2] * values[3] - values[0] * values[5]) / determinant;
        inverse[6] = (values[3] * values[7] - values[4] * values[6]) / determinant;
        inverse[7] = (values[1] * values[6] - values[0] * values[7]) / determinant;
        inverse[8] = (values[0] * values[4] - values[1] * values[3]) / determinant;
        workspace_.set_cell(inverse, std::abs(determinant));
        workspace_.prepare_reciprocal_grid();
        cell_set_ = true;
        positions_set_ = false;
    }

    void set_positions(const c_array<double>& positions)
    {
        require_reserved();
        if (!cell_set_) {
            throw std::runtime_error("Particle-mesh positions require a cell.");
        }
        if (positions.size() != 3 * particles_) {
            throw std::invalid_argument(
                "Particle-mesh positions must have shape [particles, 3].");
        }
        host_positions_.assign(positions.data(), positions.data() + positions.size());
        copy_to_device(host_positions_.data(), host_positions_.size(), positions_);
        static_cast<void>(workspace_.prepare_stencil(
            std::span<const double>(host_positions_.data(), host_positions_.size()),
            positions_));
        positions_set_ = true;
    }

    void spread(const py::array& weights)
    {
        copy_weights(weights);
        workspace_.spread(workspace_.checked_weights_view(
            weights_, particles_, lanes_, "particle-mesh probe weights"));
    }

    void set_real_mesh(const py::array& values)
    {
        require_positions();
        const auto converted = c_array<Float>::ensure(values);
        if (!converted || converted.size() != lanes_ * nx_ * ny_ * nz_) {
            throw std::invalid_argument(
                "Particle-mesh real field must have shape [lanes, nx, ny, nz].");
        }
        auto host = Kokkos::create_mirror_view(workspace_.real_mesh());
        std::copy(converted.data(), converted.data() + converted.size(), host.data());
        Kokkos::deep_copy(execution_space_, workspace_.real_mesh(), host);
        execution_space_.fence("particle-mesh probe real-field copy");
    }

    void transform_forward() { workspace_.transform_forward(); }
    void deconvolve_amplitudes() { workspace_.deconvolve_amplitudes(); }

    void load_physical_cotangent(const py::array& values)
    {
        require_positions();
        const auto converted = c_array<std::complex<Float>>::ensure(values);
        const std::int64_t elements = lanes_ * nx_ * ny_ * (nz_ / 2 + 1);
        if (!converted || converted.size() != elements) {
            throw std::invalid_argument(
                "Particle-mesh deconvolved amplitude cotangent must have shape "
                "[lanes, nx, ny, nz // 2 + 1].");
        }
        Kokkos::View<complex_type*> device(
            "symmetrix_particle_mesh_probe_cotangent", elements);
        Kokkos::View<complex_type*, Kokkos::HostSpace> host(
            "symmetrix_particle_mesh_probe_cotangent_host", elements);
        for (std::int64_t index = 0; index < elements; ++index) {
            const auto value = converted.data()[index];
            host(index) = complex_type(value.real(), value.imag());
        }
        Kokkos::deep_copy(execution_space_, device, host);
        execution_space_.fence("particle-mesh probe cotangent copy");
        workspace_.load_physical_cotangent(typename Workspace::complex_input_view(
            device.data(), elements));
        execution_space_.fence("particle-mesh probe cotangent load");
    }

    void deconvolve_cotangent() { workspace_.deconvolve_cotangent(); }
    void transform_inverse() { workspace_.transform_inverse(); }

    [[nodiscard]] py::dict gather(const py::array& weights)
    {
        copy_weights(weights);
        const typename Workspace::weight_adjoints_view adjoints(
            weight_adjoints_.data(), particles_, lanes_);
        const typename Workspace::position_forces_view forces(
            position_forces_.data(), particles_, 3);
        workspace_.gather(
            workspace_.checked_weights_view(
                weights_, particles_, lanes_, "particle-mesh probe weights"),
            typename Workspace::groups_view(), 0, adjoints, forces);
        py::dict result;
        result["weight_adjoint"] = read_real(
            weight_adjoints_, {particles_, lanes_}, particles_ * lanes_);
        result["position_forces"] = read_real(
            position_forces_, {particles_, 3}, particles_ * 3);
        return result;
    }

    [[nodiscard]] py::array_t<Float> real_mesh() const
    {
        const std::int64_t elements = lanes_ * nx_ * ny_ * nz_;
        return read_real(
            workspace_.real_mesh(), {lanes_, nx_, ny_, nz_}, elements);
    }

    [[nodiscard]] py::array_t<std::complex<Float>> spectrum() const
    {
        const std::int64_t nzh = nz_ / 2 + 1;
        const std::int64_t elements = lanes_ * nx_ * ny_ * nzh;
        auto mirror = Kokkos::create_mirror_view(workspace_.spectrum());
        Kokkos::deep_copy(execution_space_, mirror, workspace_.spectrum());
        execution_space_.fence("particle-mesh probe spectrum readback");
        std::vector<std::complex<Float>> values(static_cast<std::size_t>(elements));
        for (std::int64_t index = 0; index < elements; ++index) {
            values[static_cast<std::size_t>(index)] = {
                mirror.data()[index].real(), mirror.data()[index].imag()};
        }
        return wrap(values, {lanes_, nx_, ny_, nzh});
    }

    [[nodiscard]] py::array_t<Float> deconvolution() const
    {
        const std::int64_t bins = nx_ * ny_ * (nz_ / 2 + 1);
        return read_real(
            workspace_.reciprocal_grid().deconvolution, {nx_, ny_, nz_ / 2 + 1}, bins);
    }

    [[nodiscard]] std::int64_t radial_strain_work_items() const
    {
        require_reserved();
        return symmetrix::particle_mesh::radial_strain_work_items(
            workspace_.reciprocal_grid());
    }

private:
    template <typename HostValue, typename DeviceView>
    void copy_to_device(
        const HostValue* values, const std::size_t count, const DeviceView& destination)
    {
        auto host = Kokkos::create_mirror_view(destination);
        std::copy(values, values + count, host.data());
        Kokkos::deep_copy(execution_space_, destination, host);
        execution_space_.fence("particle-mesh probe input copy");
    }

    void copy_weights(const py::array& values)
    {
        require_positions();
        const auto converted = c_array<Float>::ensure(values);
        if (!converted || converted.size() != particles_ * lanes_) {
            throw std::invalid_argument(
                "Particle-mesh weights must have shape [particles, lanes].");
        }
        copy_to_device(converted.data(), converted.size(), weights_);
    }

    template <typename View>
    [[nodiscard]] py::array_t<Float> read_real(
        const View& source,
        const std::vector<py::ssize_t>& shape,
        const std::int64_t elements) const
    {
        auto mirror = Kokkos::create_mirror_view(source);
        Kokkos::deep_copy(execution_space_, mirror, source);
        execution_space_.fence("particle-mesh probe real readback");
        std::vector<Float> values(
            mirror.data(), mirror.data() + static_cast<std::size_t>(elements));
        return wrap(values, shape);
    }

    void require_reserved() const
    {
        if (!reserved_) {
            throw std::runtime_error("Particle-mesh transport must be reserved first.");
        }
    }

    void require_positions() const
    {
        require_reserved();
        if (!positions_set_) {
            throw std::runtime_error("Particle-mesh transport requires prepared positions.");
        }
    }

    Kokkos::DefaultExecutionSpace execution_space_;
    Workspace workspace_;
    Kokkos::View<double*> positions_;
    Kokkos::View<Float*> weights_;
    Kokkos::View<Float*> weight_adjoints_;
    Kokkos::View<Float*> position_forces_;
    std::vector<double> host_positions_;
    std::int64_t nx_ = 0;
    std::int64_t ny_ = 0;
    std::int64_t nz_ = 0;
    std::int64_t particles_ = 0;
    std::int64_t lanes_ = 0;
    bool reserved_ = false;
    bool cell_set_ = false;
    bool positions_set_ = false;
};

class TransportSeam {
public:
    TransportSeam(const std::string& precision, const std::int64_t channel_bound)
        : precision_(precision)
    {
        if (precision == k_float32) {
            engine_ = std::make_shared<TransportImpl<float>>(channel_bound);
        } else if (precision == k_float64) {
            engine_ = std::make_shared<TransportImpl<double>>(channel_bound);
        } else {
            throw std::invalid_argument(
                "Particle-mesh precision must be 'float32' or 'float64'.");
        }
    }

    template <typename Function>
    decltype(auto) dispatch(Function&& function)
    {
        if (precision_ == k_float32) {
            return function(*std::static_pointer_cast<TransportImpl<float>>(engine_));
        }
        return function(*std::static_pointer_cast<TransportImpl<double>>(engine_));
    }

    template <typename Function>
    decltype(auto) dispatch_const(Function&& function) const
    {
        if (precision_ == k_float32) {
            return function(*std::static_pointer_cast<TransportImpl<float>>(engine_));
        }
        return function(*std::static_pointer_cast<TransportImpl<double>>(engine_));
    }

    void reserve(
        std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz,
        std::int64_t particles,
        std::int64_t lanes,
        std::int64_t order)
    {
        dispatch([&](auto& value) { value.reserve(nx, ny, nz, particles, lanes, order); });
    }
    void set_cell(const c_array<double>& cell)
    {
        dispatch([&](auto& value) { value.set_cell(cell); });
    }
    void set_positions(const c_array<double>& positions)
    {
        dispatch([&](auto& value) { value.set_positions(positions); });
    }
    void spread(const py::array& weights)
    {
        dispatch([&](auto& value) { value.spread(weights); });
    }
    void set_real_mesh(const py::array& values)
    {
        dispatch([&](auto& value) { value.set_real_mesh(values); });
    }
    void transform_forward()
    {
        dispatch([](auto& value) { value.transform_forward(); });
    }
    void deconvolve_amplitudes()
    {
        dispatch([](auto& value) { value.deconvolve_amplitudes(); });
    }
    void load_physical_cotangent(const py::array& values)
    {
        dispatch([&](auto& value) { value.load_physical_cotangent(values); });
    }
    void deconvolve_cotangent()
    {
        dispatch([](auto& value) { value.deconvolve_cotangent(); });
    }
    void transform_inverse()
    {
        dispatch([](auto& value) { value.transform_inverse(); });
    }
    [[nodiscard]] py::dict gather(const py::array& weights)
    {
        return dispatch([&](auto& value) { return value.gather(weights); });
    }
    [[nodiscard]] py::object real_mesh() const
    {
        return dispatch_const([](const auto& value) -> py::object { return value.real_mesh(); });
    }
    [[nodiscard]] py::object spectrum() const
    {
        return dispatch_const([](const auto& value) -> py::object { return value.spectrum(); });
    }
    [[nodiscard]] py::object deconvolution() const
    {
        return dispatch_const(
            [](const auto& value) -> py::object { return value.deconvolution(); });
    }
    [[nodiscard]] std::int64_t radial_strain_work_items() const
    {
        return dispatch_const(
            [](const auto& value) { return value.radial_strain_work_items(); });
    }

private:
    std::string precision_;
    std::shared_ptr<void> engine_;
};

class PolarWorkspaceSeam {
public:
    PolarWorkspaceSeam(
        const std::string& precision,
        const int density_max_l,
        const int feature_max_l,
        const int spin_channels,
        const std::int64_t receiver_widths)
        : precision_(precision)
    {
        if (precision == k_float32) {
            engine_ = std::make_shared<
                symmetrix::mace_polar::PolarParticleMeshWorkspaceT<float>>(
                    Kokkos::DefaultExecutionSpace{}, density_max_l, feature_max_l,
                    spin_channels, receiver_widths);
        } else if (precision == k_float64) {
            engine_ = std::make_shared<
                symmetrix::mace_polar::PolarParticleMeshWorkspaceT<double>>(
                    Kokkos::DefaultExecutionSpace{}, density_max_l, feature_max_l,
                    spin_channels, receiver_widths);
        } else {
            throw std::invalid_argument(
                "POLAR particle-mesh precision must be 'float32' or 'float64'.");
        }
    }

    template <typename Function>
    decltype(auto) dispatch(Function&& function)
    {
        if (precision_ == k_float32) {
            return function(*std::static_pointer_cast<
                symmetrix::mace_polar::PolarParticleMeshWorkspaceT<float>>(engine_));
        }
        return function(*std::static_pointer_cast<
            symmetrix::mace_polar::PolarParticleMeshWorkspaceT<double>>(engine_));
    }

    template <typename Function>
    decltype(auto) dispatch_const(Function&& function) const
    {
        if (precision_ == k_float32) {
            return function(*std::static_pointer_cast<
                symmetrix::mace_polar::PolarParticleMeshWorkspaceT<float>>(engine_));
        }
        return function(*std::static_pointer_cast<
            symmetrix::mace_polar::PolarParticleMeshWorkspaceT<double>>(engine_));
    }

    [[nodiscard]] py::dict layout() const
    {
        return dispatch_const([](const auto& value) {
            const auto& layout = value.layout();
            py::dict result;
            result["density_max_l"] = layout.density_max_l;
            result["feature_max_l"] = layout.feature_max_l;
            result["spin_channels"] = layout.spin_channels;
            result["receiver_widths"] = layout.receiver_widths;
            result["source_components"] = layout.source_components;
            result["receiver_components"] = layout.receiver_components;
            result["source_lanes"] = layout.source_lanes;
            result["receiver_lanes"] = layout.receiver_lanes;
            result["transform_lanes"] = layout.transform_lanes;
            return result;
        });
    }

    [[nodiscard]] py::dict reserve(
        const std::int64_t nx,
        const std::int64_t ny,
        const std::int64_t nz,
        const std::int64_t nodes,
        const std::int64_t spline_order)
    {
        return dispatch([&](auto& value) {
            const auto outcome = value.reserve(nx, ny, nz, nodes, spline_order);
            py::dict result;
            result["rebuilt"] = outcome.rebuilt;
            result["generation"] = outcome.generation;
            return result;
        });
    }

    [[nodiscard]] py::dict usage() const
    {
        return dispatch_const([](const auto& value) {
            const auto usage = value.usage();
            py::dict result;
            result["active_bytes"] = usage.active_bytes;
            result["capacity_bytes"] = usage.capacity_bytes;
            result["reservations"] = usage.reservations;
            result["rebuilds"] = usage.rebuilds;
            result["reuses"] = usage.reuses;
            result["generation"] = usage.generation;
            result["nodes"] = usage.nodes;
            py::list mesh;
            for (const auto extent : usage.mesh) {
                mesh.append(extent);
            }
            result["mesh"] = std::move(mesh);
            result["spline_order"] = usage.spline_order;
            result["forward_plan_builds"] = usage.forward_plan_builds;
            result["forward_plan_reuses"] = usage.forward_plan_reuses;
            result["inverse_plan_builds"] = usage.inverse_plan_builds;
            result["inverse_plan_reuses"] = usage.inverse_plan_reuses;
            result["resident_scalar_validation_transfers"] =
                usage.resident_scalar_validation_transfers;
            result["resident_completion_fences"] =
                usage.resident_completion_fences;
            result["resident_stencil_path"] = usage.resident_stencil_path;
            result["source_lanes"] = usage.layout.source_lanes;
            result["receiver_lanes"] = usage.layout.receiver_lanes;
            result["transform_lanes"] = usage.layout.transform_lanes;
            py::list buffers;
            for (const auto& item : usage.buffers) {
                py::dict entry;
                entry["name"] = item.name;
                entry["active_elements"] = item.active_elements;
                entry["capacity_elements"] = item.capacity_elements;
                entry["active_bytes"] = item.active_bytes;
                entry["capacity_bytes"] = item.capacity_bytes;
                entry["allocation_id"] = item.allocation_id;
                buffers.append(std::move(entry));
            }
            result["buffers"] = std::move(buffers);
            return result;
        });
    }

    [[nodiscard]] py::dict scalar_forward(
        const c_array<double>& positions,
        const c_array<double>& sources,
        const c_array<double>& cell,
        const std::array<int, 3>& pbc,
        const double source_sigma,
        const std::vector<double>& receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        const std::int64_t spline_order,
        const bool include_feature_self_interaction,
        const bool include_energy_self_interaction)
    {
        if (positions.ndim() != 2 || positions.shape(1) != 3) {
            throw std::invalid_argument(
                "POLAR scalar particle-mesh positions must have shape [nodes, 3].");
        }
        if (sources.ndim() != 1 || sources.shape(0) != positions.shape(0)) {
            throw std::invalid_argument(
                "POLAR scalar particle-mesh sources must have shape [nodes].");
        }
        if (cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3) {
            throw std::invalid_argument(
                "POLAR scalar particle-mesh cell must have shape [3, 3].");
        }
        return dispatch([&](auto& value) {
            std::array<double, 9> cell_values{};
            std::copy_n(cell.data(), cell_values.size(), cell_values.begin());
            const auto result = value.scalar_forward(
                std::span<const double>(positions.data(), positions.size()),
                std::span<const double>(sources.data(), sources.size()),
                cell_values, pbc, source_sigma,
                std::span<const double>(receiver_sigmas.data(),
                    receiver_sigmas.size()),
                mesh, spline_order, include_feature_self_interaction,
                include_energy_self_interaction);
            py::dict output;
            output["features"] = wrap(result.features,
                {static_cast<py::ssize_t>(result.nodes),
                    static_cast<py::ssize_t>(result.receiver_widths)});
            output["energy"] = result.energy;
            return output;
        });
    }

    [[nodiscard]] py::dict forward(
        const c_array<double>& positions,
        const c_array<double>& source_multipoles,
        const c_array<double>& cell,
        const std::array<int, 3>& pbc,
        const double source_sigma,
        const std::vector<double>& receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        const std::int64_t spline_order,
        const bool include_feature_self_interaction,
        const bool include_energy_self_interaction)
    {
        if (positions.ndim() != 2 || positions.shape(1) != 3) {
            throw std::invalid_argument(
                "POLAR particle-mesh positions must have shape [nodes, 3].");
        }
        if (source_multipoles.ndim() != 2
            || source_multipoles.shape(0) != positions.shape(0)) {
            throw std::invalid_argument(
                "POLAR particle-mesh source multipoles must have shape "
                "[nodes, source_components].");
        }
        if (cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3) {
            throw std::invalid_argument(
                "POLAR particle-mesh cell must have shape [3, 3].");
        }
        return dispatch([&](auto& value) {
            if (source_multipoles.shape(1)
                != static_cast<py::ssize_t>(value.layout().source_components)) {
                throw std::invalid_argument(
                    "POLAR particle-mesh source multipoles must have shape "
                    "[nodes, source_components].");
            }
            std::array<double, 9> cell_values{};
            std::copy_n(cell.data(), cell_values.size(), cell_values.begin());
            const auto result = value.forward(
                std::span<const double>(positions.data(), positions.size()),
                std::span<const double>(
                    source_multipoles.data(), source_multipoles.size()),
                cell_values, pbc, source_sigma,
                std::span<const double>(receiver_sigmas.data(),
                    receiver_sigmas.size()),
                mesh, spline_order, include_feature_self_interaction,
                include_energy_self_interaction);
            py::dict output;
            output["features"] = wrap(result.features,
                {static_cast<py::ssize_t>(result.nodes),
                    static_cast<py::ssize_t>(result.feature_dimension)});
            output["energy"] = result.energy;
            return output;
        });
    }

    [[nodiscard]] py::dict forward_channels(
        const c_array<double>& positions,
        const c_array<double>& source_multipoles,
        const c_array<double>& cell,
        const std::array<int, 3>& pbc,
        const double source_sigma,
        const std::vector<double>& receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        const std::int64_t spline_order,
        const bool include_feature_self_interaction,
        const bool include_energy_self_interaction,
        const std::int64_t requested_active_channels)
    {
        if (positions.ndim() != 2 || positions.shape(1) != 3) {
            throw std::invalid_argument(
                "POLAR particle-mesh positions must have shape [nodes, 3].");
        }
        if (source_multipoles.ndim() != 2
            || source_multipoles.shape(0) != positions.shape(0)) {
            throw std::invalid_argument(
                "POLAR particle-mesh source multipoles must have shape "
                "[nodes, source_lanes].");
        }
        if (cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3) {
            throw std::invalid_argument(
                "POLAR particle-mesh cell must have shape [3, 3].");
        }
        return dispatch([&](auto& value) {
            const std::int64_t active_channels = requested_active_channels == 0
                ? value.layout().spin_channels : requested_active_channels;
            if (active_channels <= 0
                || active_channels > value.layout().spin_channels) {
                throw std::invalid_argument(
                    "POLAR particle-mesh active channels exceed the workspace "
                    "layout.");
            }
            const std::int64_t source_columns = active_channels
                * value.layout().source_components;
            if (source_multipoles.shape(1)
                != static_cast<py::ssize_t>(source_columns)) {
                throw std::invalid_argument(
                    "POLAR particle-mesh source multipoles must have shape "
                    "[nodes, source_lanes].");
            }
            std::array<double, 9> cell_values{};
            std::copy_n(cell.data(), cell_values.size(), cell_values.begin());
            const auto result = value.forward_impl(
                std::span<const double>(positions.data(), positions.size()),
                std::span<const double>(
                    source_multipoles.data(), source_multipoles.size()),
                active_channels, cell_values, pbc, source_sigma,
                std::span<const double>(receiver_sigmas.data(),
                    receiver_sigmas.size()),
                mesh, spline_order, include_feature_self_interaction,
                include_energy_self_interaction);
            py::dict output;
            output["features"] = wrap(result.features,
                {static_cast<py::ssize_t>(result.nodes),
                    static_cast<py::ssize_t>(
                        result.channels * result.feature_dimension)});
            output["energies"] = wrap(result.energies,
                {static_cast<py::ssize_t>(result.channels)});
            return output;
        });
    }

    [[nodiscard]] py::dict forward_channels_device(
        const c_array<double>& positions,
        const c_array<double>& source_multipoles,
        const c_array<double>& cell,
        const std::array<int, 3>& pbc,
        const double source_sigma,
        const std::vector<double>& receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        const std::int64_t spline_order,
        const bool include_feature_self_interaction,
        const bool include_energy_self_interaction,
        const std::int64_t requested_active_channels)
    {
        if (positions.ndim() != 2 || positions.shape(1) != 3
            || source_multipoles.ndim() != 2
            || cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3) {
            throw std::invalid_argument(
                "POLAR resident particle-mesh inputs have inconsistent shapes.");
        }
        if (source_multipoles.shape(0) != positions.shape(0)) {
            throw std::invalid_argument(
                "POLAR resident particle-mesh source rows must match positions.");
        }
        return dispatch([&](auto& value) {
            using Workspace = std::decay_t<decltype(value)>;
            using Float = typename Workspace::precision_type;
            const auto nodes = static_cast<std::int64_t>(positions.shape(0));
            const std::int64_t active_channels = requested_active_channels == 0
                ? value.layout().spin_channels : requested_active_channels;
            if (active_channels <= 0
                || active_channels > value.layout().spin_channels) {
                throw std::invalid_argument(
                    "POLAR resident particle-mesh active channels exceed the "
                    "workspace layout.");
            }
            const std::int64_t source_columns = active_channels
                * value.layout().source_components;
            if (source_multipoles.shape(1)
                != static_cast<py::ssize_t>(source_columns)) {
                throw std::invalid_argument(
                    "POLAR resident particle-mesh sources must have shape "
                    "[nodes, source_lanes].");
            }
            const std::int64_t feature_dimension =
                value.layout().receiver_widths
                * value.layout().receiver_components;
            const std::int64_t feature_columns =
                active_channels * feature_dimension;
            Kokkos::View<Float**, Kokkos::LayoutRight> sources(
                "POLAR resident diagnostic sources", nodes,
                source_columns);
            Kokkos::View<Float**, Kokkos::LayoutRight> features(
                "POLAR resident diagnostic features", nodes, feature_columns);
            Kokkos::View<double*> energies(
                "POLAR resident diagnostic energies", active_channels);
            auto host_sources = Kokkos::create_mirror_view(sources);
            for (std::int64_t node = 0; node < nodes; ++node) {
                for (std::int64_t lane = 0;
                    lane < source_columns; ++lane) {
                    host_sources(node, lane) = static_cast<Float>(
                        source_multipoles.data()[
                            node * source_columns + lane]);
                }
            }
            Kokkos::deep_copy(sources, host_sources);
            std::array<double, 9> cell_values{};
            std::copy_n(cell.data(), cell_values.size(), cell_values.begin());
            value.forward_channels_device(
                std::span<const double>(positions.data(), positions.size()),
                typename Workspace::const_device_matrix(sources), features,
                energies, cell_values, pbc, source_sigma,
                std::span<const double>(receiver_sigmas.data(),
                    receiver_sigmas.size()),
                mesh, spline_order, include_feature_self_interaction,
                include_energy_self_interaction, active_channels);
            auto host_features = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), features);
            auto host_energies = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), energies);
            std::vector<double> feature_values(
                static_cast<std::size_t>(nodes * feature_columns));
            std::vector<double> energy_values(
                static_cast<std::size_t>(active_channels));
            for (std::int64_t node = 0; node < nodes; ++node) {
                for (std::int64_t lane = 0; lane < feature_columns; ++lane) {
                    feature_values[static_cast<std::size_t>(
                        node * feature_columns + lane)] =
                        static_cast<double>(host_features(node, lane));
                }
            }
            for (int channel = 0; channel < active_channels;
                ++channel) {
                energy_values[static_cast<std::size_t>(channel)] =
                    host_energies(channel);
            }
            py::dict output;
            output["features"] = wrap(feature_values,
                {static_cast<py::ssize_t>(nodes),
                    static_cast<py::ssize_t>(feature_columns)});
            output["energies"] = wrap(energy_values,
                {static_cast<py::ssize_t>(active_channels)});
            return output;
        });
    }

    [[nodiscard]] py::dict reverse_channels(
        const c_array<double>& positions,
        const c_array<double>& source_multipoles,
        const c_array<double>& feature_seeds,
        const c_array<double>& energy_seeds,
        const c_array<double>& cell,
        const std::array<int, 3>& pbc,
        const double source_sigma,
        const std::vector<double>& receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        const std::int64_t spline_order,
        const bool include_feature_self_interaction,
        const bool include_energy_self_interaction,
        const std::int64_t requested_active_channels)
    {
        if (positions.ndim() != 2 || positions.shape(1) != 3
            || source_multipoles.ndim() != 2
            || feature_seeds.ndim() != 2 || energy_seeds.ndim() != 1
            || cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3) {
            throw std::invalid_argument(
                "POLAR particle-mesh reverse inputs have inconsistent shapes.");
        }
        return dispatch([&](auto& value) {
            const std::int64_t active_channels = requested_active_channels == 0
                ? value.layout().spin_channels : requested_active_channels;
            if (active_channels <= 0
                || active_channels > value.layout().spin_channels) {
                throw std::invalid_argument(
                    "POLAR particle-mesh reverse active channels exceed the "
                    "workspace layout.");
            }
            const std::int64_t source_lanes =
                active_channels * value.layout().source_components;
            const std::int64_t feature_lanes = active_channels
                * value.layout().receiver_widths
                * value.layout().receiver_components;
            if (source_multipoles.shape(0) != positions.shape(0)
                || feature_seeds.shape(0) != positions.shape(0)
                || source_multipoles.shape(1) != source_lanes
                || feature_seeds.shape(1) != feature_lanes
                || energy_seeds.shape(0) != active_channels) {
                throw std::invalid_argument(
                    "POLAR particle-mesh reverse source/seed shapes do not match "
                    "the workspace layout.");
            }
            std::array<double, 9> cell_values{};
            std::copy_n(cell.data(), cell_values.size(), cell_values.begin());
            const auto result = value.reverse_impl(
                std::span<const double>(positions.data(), positions.size()),
                std::span<const double>(source_multipoles.data(),
                    source_multipoles.size()),
                std::span<const double>(feature_seeds.data(), feature_seeds.size()),
                std::span<const double>(energy_seeds.data(), energy_seeds.size()),
                cell_values, pbc, source_sigma,
                std::span<const double>(receiver_sigmas.data(),
                    receiver_sigmas.size()),
                mesh, spline_order, include_feature_self_interaction,
                include_energy_self_interaction, active_channels);
            py::dict output;
            output["source_multipole_adjoints"] = wrap(
                result.source_multipole_adjoints,
                {static_cast<py::ssize_t>(result.nodes),
                    static_cast<py::ssize_t>(result.source_lanes)});
            output["position_adjoints"] = wrap(result.position_adjoints,
                {static_cast<py::ssize_t>(result.nodes), 3});
            output["cell_adjoints"] = wrap(result.cell_adjoints, {3, 3});
            return output;
        });
    }

    [[nodiscard]] py::dict reverse_channels_device(
        const c_array<double>& positions,
        const c_array<double>& source_multipoles,
        const c_array<double>& feature_seeds,
        const c_array<double>& energy_seeds,
        const c_array<double>& cell,
        const std::array<int, 3>& pbc,
        const double source_sigma,
        const std::vector<double>& receiver_sigmas,
        const std::array<std::int64_t, 3>& mesh,
        const std::int64_t spline_order,
        const bool include_feature_self_interaction,
        const bool include_energy_self_interaction,
        const std::int64_t requested_active_channels)
    {
        if (positions.ndim() != 2 || positions.shape(1) != 3
            || source_multipoles.ndim() != 2
            || feature_seeds.ndim() != 2 || energy_seeds.ndim() != 1
            || cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3) {
            throw std::invalid_argument(
                "POLAR resident particle-mesh reverse inputs have "
                "inconsistent shapes.");
        }
        return dispatch([&](auto& value) {
            using Workspace = std::decay_t<decltype(value)>;
            using Float = typename Workspace::precision_type;
            const auto nodes = static_cast<std::int64_t>(positions.shape(0));
            const std::int64_t channels = requested_active_channels == 0
                ? value.layout().spin_channels : requested_active_channels;
            if (channels <= 0 || channels > value.layout().spin_channels) {
                throw std::invalid_argument(
                    "POLAR resident particle-mesh reverse active channels "
                    "exceed the workspace layout.");
            }
            const std::int64_t source_lanes =
                channels * value.layout().source_components;
            const std::int64_t feature_lanes = channels
                * value.layout().receiver_widths
                * value.layout().receiver_components;
            if (source_multipoles.shape(0) != positions.shape(0)
                || feature_seeds.shape(0) != positions.shape(0)
                || source_multipoles.shape(1) != source_lanes
                || feature_seeds.shape(1) != feature_lanes
                || energy_seeds.shape(0) != channels) {
                throw std::invalid_argument(
                    "POLAR resident particle-mesh reverse source/seed shapes "
                    "do not match the workspace layout.");
            }

            Kokkos::View<Float**, Kokkos::LayoutRight> sources(
                "POLAR resident reverse diagnostic sources", nodes,
                source_lanes);
            Kokkos::View<Float**, Kokkos::LayoutRight> feature_seed_view(
                "POLAR resident reverse diagnostic feature seeds", nodes,
                feature_lanes);
            Kokkos::View<double*> energy_seed_view(
                "POLAR resident reverse diagnostic energy seeds", channels);
            Kokkos::View<Float**, Kokkos::LayoutRight> forward_features(
                "POLAR resident reverse diagnostic forward features", nodes,
                feature_lanes);
            Kokkos::View<double*> forward_energies(
                "POLAR resident reverse diagnostic forward energies", channels);
            Kokkos::View<Float**, Kokkos::LayoutRight> source_adjoints(
                "POLAR resident reverse diagnostic source adjoints", nodes,
                source_lanes);
            Kokkos::View<double**, Kokkos::LayoutRight> position_adjoints(
                "POLAR resident reverse diagnostic position adjoints", nodes, 3);
            Kokkos::View<double*> cell_adjoints(
                "POLAR resident reverse diagnostic cell adjoints", 9);

            auto host_sources = Kokkos::create_mirror_view(sources);
            auto host_feature_seeds = Kokkos::create_mirror_view(feature_seed_view);
            auto host_energy_seeds = Kokkos::create_mirror_view(energy_seed_view);
            for (std::int64_t node = 0; node < nodes; ++node) {
                for (std::int64_t lane = 0; lane < source_lanes; ++lane) {
                    host_sources(node, lane) = static_cast<Float>(
                        source_multipoles.data()[node * source_lanes + lane]);
                }
                for (std::int64_t lane = 0; lane < feature_lanes; ++lane) {
                    host_feature_seeds(node, lane) = static_cast<Float>(
                        feature_seeds.data()[node * feature_lanes + lane]);
                }
            }
            for (std::int64_t channel = 0; channel < channels; ++channel) {
                host_energy_seeds(channel) = energy_seeds.data()[channel];
            }
            Kokkos::deep_copy(sources, host_sources);
            Kokkos::deep_copy(feature_seed_view, host_feature_seeds);
            Kokkos::deep_copy(energy_seed_view, host_energy_seeds);

            std::array<double, 9> cell_values{};
            std::copy_n(cell.data(), cell_values.size(), cell_values.begin());
            value.reverse_channels_device(
                std::span<const double>(positions.data(), positions.size()),
                typename Workspace::const_device_matrix(sources),
                typename Workspace::const_device_matrix(feature_seed_view),
                typename Workspace::const_device_vector(energy_seed_view),
                forward_features, forward_energies, source_adjoints,
                position_adjoints, cell_adjoints, cell_values, pbc,
                source_sigma,
                std::span<const double>(receiver_sigmas.data(),
                    receiver_sigmas.size()),
                mesh, spline_order, include_feature_self_interaction,
                include_energy_self_interaction, channels);

            auto host_source_adjoints = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), source_adjoints);
            auto host_position_adjoints = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), position_adjoints);
            auto host_cell_adjoints = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), cell_adjoints);
            std::vector<double> source_values(
                static_cast<std::size_t>(nodes * source_lanes));
            std::vector<double> position_values(
                static_cast<std::size_t>(nodes * 3));
            std::vector<double> cell_values_output(9);
            for (std::int64_t node = 0; node < nodes; ++node) {
                for (std::int64_t lane = 0; lane < source_lanes; ++lane) {
                    source_values[static_cast<std::size_t>(
                        node * source_lanes + lane)] =
                        static_cast<double>(host_source_adjoints(node, lane));
                }
                for (int component = 0; component < 3; ++component) {
                    position_values[static_cast<std::size_t>(
                        node * 3 + component)] =
                        host_position_adjoints(node, component);
                }
            }
            for (int component = 0; component < 9; ++component) {
                cell_values_output[static_cast<std::size_t>(component)] =
                    host_cell_adjoints(component);
            }
            py::dict output;
            output["source_multipole_adjoints"] = wrap(source_values,
                {static_cast<py::ssize_t>(nodes),
                    static_cast<py::ssize_t>(source_lanes)});
            output["position_adjoints"] = wrap(position_values,
                {static_cast<py::ssize_t>(nodes), 3});
            output["cell_adjoints"] = wrap(cell_values_output, {3, 3});
            auto host_forward_features = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), forward_features);
            auto host_forward_energies = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), forward_energies);
            std::vector<double> forward_feature_values(
                static_cast<std::size_t>(nodes * feature_lanes));
            std::vector<double> forward_energy_values(
                static_cast<std::size_t>(channels));
            for (std::int64_t node = 0; node < nodes; ++node) {
                for (std::int64_t lane = 0; lane < feature_lanes; ++lane) {
                    forward_feature_values[static_cast<std::size_t>(
                        node * feature_lanes + lane)] =
                        static_cast<double>(host_forward_features(node, lane));
                }
            }
            for (std::int64_t channel = 0; channel < channels; ++channel) {
                forward_energy_values[static_cast<std::size_t>(channel)] =
                    host_forward_energies(channel);
            }
            output["forward_features"] = wrap(forward_feature_values,
                {static_cast<py::ssize_t>(nodes),
                    static_cast<py::ssize_t>(feature_lanes)});
            output["forward_energies"] = wrap(forward_energy_values,
                {static_cast<py::ssize_t>(channels)});
            return output;
        });
    }

    void arm_allocation_failure(const std::int64_t fail_at)
    {
        symmetrix::particle_mesh::detail::arm_allocation_fault(fail_at);
    }

    void clear_allocation_failure()
    {
        symmetrix::particle_mesh::detail::clear_allocation_fault();
    }

private:
    std::string precision_;
    std::shared_ptr<void> engine_;
};

}  // namespace

void bind_particle_mesh(py::module_& m)
{
    py::class_<TransportSeam>(m, "_ParticleMeshTransport")
        .def(py::init<std::string, std::int64_t>())
        .def("reserve", &TransportSeam::reserve)
        .def("set_cell", &TransportSeam::set_cell)
        .def("set_positions", &TransportSeam::set_positions)
        .def("spread", &TransportSeam::spread)
        .def("set_real_mesh", &TransportSeam::set_real_mesh)
        .def("transform_forward", &TransportSeam::transform_forward)
        .def("deconvolve_amplitudes", &TransportSeam::deconvolve_amplitudes)
        .def("load_physical_cotangent", &TransportSeam::load_physical_cotangent)
        .def("deconvolve_cotangent", &TransportSeam::deconvolve_cotangent)
        .def("transform_inverse", &TransportSeam::transform_inverse)
        .def("gather", &TransportSeam::gather)
        .def("real_mesh", &TransportSeam::real_mesh)
        .def("spectrum", &TransportSeam::spectrum)
        .def("deconvolution", &TransportSeam::deconvolution)
        .def("radial_strain_work_items", &TransportSeam::radial_strain_work_items);

    py::class_<PolarWorkspaceSeam>(m, "_PolarParticleMeshWorkspace")
        .def(py::init<std::string, int, int, int, std::int64_t>())
        .def("layout", &PolarWorkspaceSeam::layout)
        .def("reserve", &PolarWorkspaceSeam::reserve)
        .def("usage", &PolarWorkspaceSeam::usage)
        .def("scalar_forward", &PolarWorkspaceSeam::scalar_forward,
            py::arg("positions"), py::arg("sources"), py::arg("cell"),
            py::arg("pbc"), py::arg("source_sigma"),
            py::arg("receiver_sigmas"), py::arg("mesh"),
            py::arg("spline_order"),
            py::arg("include_feature_self_interaction") = false,
            py::arg("include_energy_self_interaction") = false)
        .def("forward", &PolarWorkspaceSeam::forward,
            py::arg("positions"), py::arg("source_multipoles"),
            py::arg("cell"), py::arg("pbc"), py::arg("source_sigma"),
            py::arg("receiver_sigmas"), py::arg("mesh"),
            py::arg("spline_order"),
            py::arg("include_feature_self_interaction") = false,
            py::arg("include_energy_self_interaction") = false)
        .def("forward_channels", &PolarWorkspaceSeam::forward_channels,
            py::arg("positions"), py::arg("source_multipoles"),
            py::arg("cell"), py::arg("pbc"), py::arg("source_sigma"),
            py::arg("receiver_sigmas"), py::arg("mesh"),
            py::arg("spline_order"),
            py::arg("include_feature_self_interaction") = false,
            py::arg("include_energy_self_interaction") = false,
            py::arg("active_channels") = 0)
        .def("forward_channels_device", &PolarWorkspaceSeam::forward_channels_device,
            py::arg("positions"), py::arg("source_multipoles"),
            py::arg("cell"), py::arg("pbc"), py::arg("source_sigma"),
            py::arg("receiver_sigmas"), py::arg("mesh"),
            py::arg("spline_order"),
            py::arg("include_feature_self_interaction") = true,
            py::arg("include_energy_self_interaction") = true,
            py::arg("active_channels") = 0)
        .def("reverse_channels", &PolarWorkspaceSeam::reverse_channels,
            py::arg("positions"), py::arg("source_multipoles"),
            py::arg("feature_seeds"), py::arg("energy_seeds"),
            py::arg("cell"), py::arg("pbc"), py::arg("source_sigma"),
            py::arg("receiver_sigmas"), py::arg("mesh"),
            py::arg("spline_order"),
            py::arg("include_feature_self_interaction") = false,
            py::arg("include_energy_self_interaction") = false,
            py::arg("active_channels") = 0)
        .def("reverse_channels_device",
            &PolarWorkspaceSeam::reverse_channels_device,
            py::arg("positions"), py::arg("source_multipoles"),
            py::arg("feature_seeds"), py::arg("energy_seeds"),
            py::arg("cell"), py::arg("pbc"), py::arg("source_sigma"),
            py::arg("receiver_sigmas"), py::arg("mesh"),
            py::arg("spline_order"),
            py::arg("include_feature_self_interaction") = false,
            py::arg("include_energy_self_interaction") = false,
            py::arg("active_channels") = 0)
        .def("arm_allocation_failure", &PolarWorkspaceSeam::arm_allocation_failure)
        .def("clear_allocation_failure", &PolarWorkspaceSeam::clear_allocation_failure);
}
