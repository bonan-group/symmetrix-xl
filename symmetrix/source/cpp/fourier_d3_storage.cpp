// Private native seam over the FourierD3 storage, cache keys, and workspaces.
//
// These bindings are internal infrastructure: they exist so the storage
// contract (checked 64-bit geometry, grow-only allocations, active/capacity
// byte accounting, cache-key separation, and destruction before Kokkos
// finalization) is testable in the qualified build.  They are not package
// exports, they evaluate nothing, and they publish no evaluator type. The
// public FP32/FP64 evaluators and diagnostic surface are bound separately.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fourier_d3_diagnostics_report.hpp"
#include "fourier_d3_storage.hpp"

namespace py = pybind11;

namespace binding = symmetrix::fourier_d3::binding;

namespace {

using symmetrix::fourier_d3::BufferUsage;
using symmetrix::fourier_d3::DampingCoefficients;
using symmetrix::fourier_d3::FftPlanDomain;
using symmetrix::fourier_d3::FftPlanKey;
using symmetrix::fourier_d3::FftPlanRole;
using symmetrix::fourier_d3::FftPlanUsage;
using symmetrix::fourier_d3::KernelDomain;
using symmetrix::fourier_d3::KernelDomainUsage;
using symmetrix::fourier_d3::KernelKey;
using symmetrix::fourier_d3::MeshShape;
using symmetrix::fourier_d3::MeshStorage;
using symmetrix::fourier_d3::MeshStorageUsage;
using symmetrix::fourier_d3::RetainedAllocationReport;
using symmetrix::fourier_d3::PlanIdentity;

constexpr std::string_view k_float32 = "float32";
constexpr std::string_view k_float64 = "float64";

void require_known_precision(const std::string& precision)
{
    if (precision != k_float32 && precision != k_float64) {
        throw std::invalid_argument(
            "FourierD3 precision must be 'float32' or 'float64', got '" + precision + "'.");
    }
}


// Owns one precision-specialized mesh storage behind a stable Python name.
class MeshStorageSeam {
public:
    MeshStorageSeam(
        std::string precision, const std::string& layout, std::int64_t channel_bound)
        : precision_(std::move(precision))
    {
        require_known_precision(precision_);
        const auto mesh_layout = symmetrix::fourier_d3::layout_from_token(layout);
        if (precision_ == k_float32) {
            assign<float>(mesh_layout, channel_bound);
        } else {
            assign<double>(mesh_layout, channel_bound);
        }
    }

    [[nodiscard]] const std::string& precision() const noexcept { return precision_; }

    template <typename Float>
    [[nodiscard]] MeshStorage<Float>& typed()
    {
        // `precision_` selects the concrete instantiation, so this cast is
        // sound and the seam has exactly one owner of the storage.
        if (precision_ != MeshStorage<Float>::precision_token()) {
            throw std::logic_error(
                "FourierD3 storage seam precision mismatch; this is a binding defect.");
        }
        return *static_cast<MeshStorage<Float>*>(storage_.get());
    }

    // The dispatch helpers are defined before their call sites: a member
    // function with a deduced return type cannot be called before its
    // definition is seen.
    template <typename Function>
    auto dispatch(Function&& function)
    {
        if (precision_ == k_float32) {
            return function(typed<float>());
        }
        return function(typed<double>());
    }

    template <typename Function>
    auto dispatch(Function&& function) const
    {
        if (precision_ == k_float32) {
            return function(typed_const<float>());
        }
        return function(typed_const<double>());
    }

    py::dict reserve(
        std::int64_t nx, std::int64_t ny, std::int64_t nz, std::int64_t channels)
    {
        return dispatch([&](auto& storage) {
            const auto outcome = storage.reserve(nx, ny, nz, channels);
            py::dict report;
            report["grown"] = outcome.grown;
            report["allocation_generation"] = outcome.allocation_generation;
            report["active_channels"] = outcome.active_channels;
            return report;
        });
    }

    [[nodiscard]] py::dict usage() const
    {
        return dispatch([&](const auto& storage) {
            const MeshStorageUsage usage = storage.usage();
            py::dict report;
            report["buffers"] = binding::buffer_reports(usage.buffers);
            report["active_bytes"] = usage.active_bytes;
            report["capacity_bytes"] = usage.capacity_bytes;
            report["reservations"] = usage.reservations;
            report["allocations"] = usage.allocations;
            report["allocation_generation"] = usage.allocation_generation;
            report["channel_bound"] = usage.channel_bound;
            report["active_channels"] = usage.active_channels;
            report["peak_channels"] = usage.peak_channels;
            return report;
        });
    }

    [[nodiscard]] std::optional<MeshShape> shape() const
    {
        return dispatch([&](const auto& storage) { return storage.shape(); });
    }

private:
    template <typename Float>
    void assign(symmetrix::fourier_d3::MeshLayout layout, std::int64_t channel_bound)
    {
        storage_ = std::static_pointer_cast<void>(
            std::make_shared<MeshStorage<Float>>(layout, channel_bound));
    }

    template <typename Float>
    [[nodiscard]] const MeshStorage<Float>& typed_const() const
    {
        return *static_cast<const MeshStorage<Float>*>(storage_.get());
    }

    std::string precision_;
    std::shared_ptr<void> storage_;
};

[[nodiscard]] FftPlanRole role_from_argument(const std::optional<std::string>& role)
{
    if (!role) {
        return FftPlanRole::forward_r2c;
    }
    try {
        return symmetrix::fourier_d3::plan_role_from_token(*role);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(
            "FourierD3 FFT plan role must be 'forward_r2c' or 'inverse_c2r', got '"
            + *role + "'.");
    }
}

// Owns one precision-specialized FFT-plan domain.
class FftPlanDomainSeam {
public:
    explicit FftPlanDomainSeam(std::string precision) : precision_(std::move(precision))
    {
        require_known_precision(precision_);
        if (precision_ == k_float32) {
            assign<float>();
        } else {
            assign<double>();
        }
    }

    py::dict acquire(MeshStorageSeam& storage, const std::optional<std::string>& role)
    {
        if (storage.precision() != precision_) {
            throw std::invalid_argument(
                "FourierD3 FFT plan domain precision '" + precision_
                + "' does not match the storage precision '" + storage.precision() + "'.");
        }
        const FftPlanRole resolved = role_from_argument(role);
        if (precision_ == k_float32) {
            return acquire_typed<float>(storage, resolved);
        }
        return acquire_typed<double>(storage, resolved);
    }

    py::dict execute(MeshStorageSeam& storage, const std::optional<std::string>& role)
    {
        if (storage.precision() != precision_) {
            throw std::invalid_argument(
                "FourierD3 FFT plan domain precision '" + precision_
                + "' does not match the storage precision '" + storage.precision() + "'.");
        }
        const FftPlanRole resolved = role_from_argument(role);
        if (precision_ == k_float32) {
            static_cast<FftPlanDomain<float>*>(domain_.get())
                ->execute(storage.typed<float>(), resolved);
        } else {
            static_cast<FftPlanDomain<double>*>(domain_.get())
                ->execute(storage.typed<double>(), resolved);
        }
        return precision_ == k_float32
            ? binding::plan_usage_report(
                  static_cast<FftPlanDomain<float>*>(domain_.get())->usage(resolved))
            : binding::plan_usage_report(
                  static_cast<FftPlanDomain<double>*>(domain_.get())->usage(resolved));
    }

    template <typename Function>
    auto dispatch(Function&& function)
    {
        if (precision_ == k_float32) {
            return function(*static_cast<FftPlanDomain<float>*>(domain_.get()));
        }
        return function(*static_cast<FftPlanDomain<double>*>(domain_.get()));
    }

    template <typename Function>
    auto dispatch(Function&& function) const
    {
        if (precision_ == k_float32) {
            return function(*static_cast<const FftPlanDomain<float>*>(domain_.get()));
        }
        return function(*static_cast<const FftPlanDomain<double>*>(domain_.get()));
    }

    py::dict release(const std::optional<std::string>& role)
    {
        const std::optional<FftPlanRole> resolved =
            role ? std::optional<FftPlanRole>(role_from_argument(role)) : std::nullopt;
        return dispatch([&](auto& domain) {
            py::dict report;
            report["released"] =
                resolved ? domain.release(*resolved) : domain.release();
            return report;
        });
    }

    // The flat fields describe the forward slot, which is what a single-plan
    // caller means; `plans` carries both roles.
    [[nodiscard]] py::dict usage() const
    {
        return dispatch([&](const auto& domain) {
            const FftPlanUsage forward = domain.usage(FftPlanRole::forward_r2c);
            py::dict report = binding::plan_usage_report(forward);
            report["execution_instance_id"] = domain.execution_instance_id();
            py::list plans;
            for (const FftPlanUsage& usage : domain.plan_usage()) {
                py::dict plan = binding::plan_usage_report(usage);
                plan["execution_instance_id"] = usage.execution_instance_id;
                plans.append(std::move(plan));
            }
            report["plans"] = plans;
            return report;
        });
    }

private:
    template <typename Float>
    void assign()
    {
        domain_ = std::static_pointer_cast<void>(
            std::make_shared<FftPlanDomain<Float>>(Kokkos::DefaultExecutionSpace()));
    }

    template <typename Float>
    py::dict acquire_typed(MeshStorageSeam& storage, FftPlanRole role)
    {
        auto& domain = *static_cast<FftPlanDomain<Float>*>(domain_.get());
        const auto outcome = domain.acquire(storage.typed<Float>(), role);
        const FftPlanUsage usage = domain.usage(role);
        py::dict report;
        report["role"] = std::string(symmetrix::fourier_d3::plan_role_token(role));
        report["built"] = outcome.built;
        report["reused"] = outcome.reused;
        report["allocation_generation"] = outcome.allocation_generation;
        report["key"] = usage.key;
        report["identity"] = binding::identity_report(usage.identity);
        report["allocation"] = binding::retained_allocation_report(usage.allocation);
        return report;
    }

    std::string precision_;
    std::shared_ptr<void> domain_;
};

// Owns one precision-specialized reciprocal-kernel domain.
class KernelDomainSeam {
public:
    explicit KernelDomainSeam(std::string precision) : precision_(std::move(precision))
    {
        require_known_precision(precision_);
        if (precision_ == k_float32) {
            assign<float>();
        } else {
            assign<double>();
        }
    }

    py::dict reserve(
        const std::array<double, 9>& cell,
        const std::array<double, 4>& damping,
        const std::vector<int>& species,
        int spline_order,
        const MeshShape& mesh)
    {
        if (precision_ == k_float32) {
            return reserve_typed<float>(cell, damping, species, spline_order, mesh);
        }
        return reserve_typed<double>(cell, damping, species, spline_order, mesh);
    }

    template <typename Function>
    auto dispatch(Function&& function) const
    {
        if (precision_ == k_float32) {
            return function(*static_cast<const KernelDomain<float>*>(domain_.get()));
        }
        return function(*static_cast<const KernelDomain<double>*>(domain_.get()));
    }

    [[nodiscard]] py::dict usage() const
    {
        return dispatch([&](const auto& domain) {
            const KernelDomainUsage usage = domain.usage();
            py::dict report;
            report["has_key"] = usage.has_key;
            report["key"] = usage.key;
            report["key_hits"] = usage.key_hits;
            report["key_misses"] = usage.key_misses;
            report["key_rebuilds"] = usage.key_rebuilds;
            report["reservations"] = usage.reservations;
            report["allocations"] = usage.allocations;
            report["buffers"] = binding::buffer_reports(usage.buffers);
            report["active_bytes"] = usage.active_bytes;
            report["capacity_bytes"] = usage.capacity_bytes;
            report["identity"] = binding::identity_report(usage.identity);
            return report;
        });
    }

private:
    template <typename Float>
    void assign()
    {
        domain_ = std::static_pointer_cast<void>(std::make_shared<KernelDomain<Float>>());
    }

    template <typename Float>
    py::dict reserve_typed(
        const std::array<double, 9>& cell,
        const std::array<double, 4>& damping,
        const std::vector<int>& species,
        int spline_order,
        const MeshShape& mesh)
    {
        auto& domain = *static_cast<KernelDomain<Float>*>(domain_.get());
        const PlanIdentity identity = symmetrix::fourier_d3::compiled_plan_identity<Float>();
        const auto outcome = domain.reserve(
            KernelKey(
                cell, DampingCoefficients{damping[0], damping[1], damping[2], damping[3]},
                species, spline_order, identity.backend, identity.precision),
            mesh);
        py::dict report;
        report["key_reused"] = outcome.key_reused;
        report["buffers_grown"] = outcome.buffers_grown;
        return report;
    }

    std::string precision_;
    std::shared_ptr<void> domain_;
};

[[nodiscard]] py::tuple extents_tuple(const MeshShape& shape)
{
    return py::make_tuple(shape.extent(0), shape.extent(1), shape.extent(2));
}

[[nodiscard]] py::tuple strides_tuple(const MeshShape& shape)
{
    return py::make_tuple(shape.stride(0), shape.stride(1), shape.stride(2));
}

[[nodiscard]] py::tuple spectrum_strides_tuple(const MeshShape& shape)
{
    return py::make_tuple(
        shape.spectrum_stride(0), shape.spectrum_stride(1), shape.spectrum_stride(2));
}

}  // namespace

void bind_fourier_d3_storage(py::module_& m)
{
    py::class_<MeshShape>(m, "_FourierD3MeshShape")
        .def(
            py::init([](std::int64_t nx, std::int64_t ny, std::int64_t nz,
                        std::int64_t batch, const std::string& layout) {
                return MeshShape(
                    nx, ny, nz, batch, symmetrix::fourier_d3::layout_from_token(layout));
            }),
            py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("batch"),
            py::arg("layout") = "layout_right",
            "Checked 64-bit mesh geometry; overflow and impossible extents are rejected.")
        .def_property_readonly(
            "extents", [](const MeshShape& shape) { return extents_tuple(shape); })
        .def_property_readonly("batch", &MeshShape::batch)
        .def_property_readonly(
            "layout", [](const MeshShape& shape) {
                return std::string(symmetrix::fourier_d3::layout_token(shape.layout()));
            })
        .def_property_readonly("point_count", &MeshShape::point_count)
        .def_property_readonly("real_element_count", &MeshShape::real_element_count)
        .def_property_readonly("spectrum_extent", &MeshShape::spectrum_extent)
        .def_property_readonly("spectrum_point_count", &MeshShape::spectrum_point_count)
        .def_property_readonly("spectrum_element_count", &MeshShape::spectrum_element_count)
        .def_property_readonly(
            "strides", [](const MeshShape& shape) { return strides_tuple(shape); })
        .def_property_readonly(
            "spectrum_strides",
            [](const MeshShape& shape) { return spectrum_strides_tuple(shape); })
        .def_property_readonly("description", &MeshShape::description)
        .def("offset", &MeshShape::offset, py::arg("i"), py::arg("j"), py::arg("k"))
        .def(
            "spectrum_offset", &MeshShape::spectrum_offset, py::arg("i"), py::arg("j"),
            py::arg("k"))
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<MeshStorageSeam>(m, "_FourierD3MeshStorage")
        .def(
            py::init<std::string, std::string, std::int64_t>(), py::arg("precision"),
            py::arg("layout"), py::arg("channel_bound"),
            "Grow-only mesh workspace bounded by the configured rank chunk.")
        .def_property_readonly("precision", &MeshStorageSeam::precision)
        .def(
            "reserve", &MeshStorageSeam::reserve, py::arg("nx"), py::arg("ny"), py::arg("nz"),
            py::arg("channels"))
        .def("usage", &MeshStorageSeam::usage)
        .def_property_readonly("shape", &MeshStorageSeam::shape);

    py::class_<FftPlanKey>(m, "_FourierD3FftPlanKey")
        .def(
            py::init([](const MeshShape& mesh, std::string backend, std::string precision,
                        std::string provider, const std::string& role) {
                return FftPlanKey(
                    mesh,
                    PlanIdentity{
                        std::move(backend), std::move(precision), std::move(provider)},
                    role_from_argument(role));
            }),
            py::arg("mesh"), py::arg("backend"), py::arg("precision"), py::arg("provider"),
            py::arg("role") = "forward_r2c")
        .def_property_readonly("description", &FftPlanKey::description)
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<FftPlanDomainSeam>(m, "_FourierD3FftPlanDomain")
        .def(py::init<std::string>(), py::arg("precision"))
        .def(
            "acquire", &FftPlanDomainSeam::acquire, py::arg("storage"),
            py::arg("role") = py::none(),
            "Acquires the plan of one role for the storage's retained allocation; "
            "the two roles are cached independently.")
        .def(
            "execute", &FftPlanDomainSeam::execute, py::arg("storage"),
            py::arg("role") = py::none(),
            "Runs the held plan of one role on the storage's active views, in that "
            "role's direction and unnormalized.")
        .def(
            "release", &FftPlanDomainSeam::release, py::arg("role") = py::none(),
            "Releases every role, or one named role.")
        .def("usage", &FftPlanDomainSeam::usage);

    py::class_<KernelKey>(m, "_FourierD3KernelKey")
        .def(
            py::init([](const std::array<double, 9>& cell,
                        const std::array<double, 4>& damping,
                        std::vector<int> species, int spline_order, std::string backend,
                        std::string precision) {
                return KernelKey(
                    cell,
                    DampingCoefficients{damping[0], damping[1], damping[2], damping[3]},
                    std::move(species), spline_order, std::move(backend),
                    std::move(precision));
            }),
            py::arg("cell"), py::arg("damping"), py::arg("species"), py::arg("spline_order"),
            py::arg("backend"), py::arg("precision"))
        .def_property_readonly("description", &KernelKey::description)
        .def_property_readonly("pair_count", &KernelKey::pair_count)
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<KernelDomainSeam>(m, "_FourierD3KernelDomain")
        .def(py::init<std::string>(), py::arg("precision"))
        .def(
            "reserve", &KernelDomainSeam::reserve, py::arg("cell"), py::arg("damping"),
            py::arg("species"), py::arg("spline_order"), py::arg("mesh"))
        .def("usage", &KernelDomainSeam::usage);

    m.def(
        "_fourier_d3_plan_identity",
        [](const std::string& precision) {
            require_known_precision(precision);
            const PlanIdentity identity = precision == k_float32
                ? symmetrix::fourier_d3::compiled_plan_identity<float>()
                : symmetrix::fourier_d3::compiled_plan_identity<double>();
            return binding::identity_report(identity);
        },
        py::arg("precision"),
        "Identity of the compiled FourierD3 FFT boundary (backend, precision, provider).");

    m.def(
        "_fourier_d3_arm_allocation_fault",
        [](std::int64_t fail_at) { symmetrix::fourier_d3::detail::arm_allocation_fault(fail_at); },
        py::arg("fail_at"),
        "Fail the n-th storage allocation after this call (0 = the next one), once.");
    m.def(
        "_fourier_d3_clear_allocation_fault",
        []() { symmetrix::fourier_d3::detail::clear_allocation_fault(); },
        "Disarm the allocation fault injection.");
    m.def(
        "_fourier_d3_allocation_fault_armed",
        []() { return symmetrix::fourier_d3::detail::allocation_fault_armed(); },
        "Whether an allocation fault is still armed.");
}
