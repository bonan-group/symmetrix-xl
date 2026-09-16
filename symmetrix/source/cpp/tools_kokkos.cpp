#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <barrier>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#if defined(__linux__)
#include <link.h>
#include <sched.h>
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
#include <omp.h>
#endif

#include "tools_kokkos.hpp"
#include "device_backend.hpp"
#include "device_environment.hpp"
#include "host_dense_kernels.hpp"
#include "host_worker_blas.hpp"
#include "neighbor_graph_kokkos.hpp"
#include "openmp_build_metadata.hpp"
#include "cblas.hpp"

namespace py = pybind11;

namespace {

std::string canonical_path(const std::string& path)
{
    if (path.empty())
        return {};
    std::error_code error;
    const auto canonical = std::filesystem::canonical(path, error);
    return error ? path : canonical.string();
}

bool files_are_byte_identical(
    const std::string& first_path, const std::string& second_path)
{
    if (first_path.empty() || second_path.empty())
        return false;
    if (first_path == second_path)
        return true;
    std::error_code error;
    const auto first_size = std::filesystem::file_size(first_path, error);
    if (error)
        return false;
    const auto second_size = std::filesystem::file_size(second_path, error);
    if (error || first_size != second_size)
        return false;
    std::ifstream first(first_path, std::ios::binary);
    std::ifstream second(second_path, std::ios::binary);
    if (!first || !second)
        return false;
    constexpr std::size_t buffer_size = 64*1024;
    std::vector<char> first_buffer(buffer_size);
    std::vector<char> second_buffer(buffer_size);
    while (first && second) {
        first.read(first_buffer.data(), first_buffer.size());
        second.read(second_buffer.data(), second_buffer.size());
        const auto count = first.gcount();
        if (count != second.gcount()
                || !std::equal(
                    first_buffer.begin(), first_buffer.begin()+count,
                    second_buffer.begin()))
            return false;
    }
    return first.eof() && second.eof();
}

struct OpenMPWorkerRecord {
    int thread_id = -1;
    int current_cpu = -1;
    std::vector<int> affinity;
};

std::vector<int> current_affinity()
{
    std::vector<int> result;
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
        for (int cpu=0; cpu<CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &mask))
                result.push_back(cpu);
#endif
    return result;
}

py::dict openmp_worker_sentinel_dict()
{
    py::dict result;
    result["schema"] = "symmetrix.openmp-worker-sentinel";
    result["version"] = 1;
    result["enabled"] = symmetrix::build::openmp_enabled;
#if defined(__linux__)
    result["cpu_observation_supported"] = true;
#else
    result["cpu_observation_supported"] = false;
#endif
    py::list workers;
#if defined(KOKKOS_ENABLE_OPENMP)
    const int capacity = omp_get_max_threads();
    std::vector<OpenMPWorkerRecord> records(capacity);
    int actual_team_size = 0;
#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
#pragma omp single
        actual_team_size = omp_get_num_threads();
        records[thread_id].thread_id = thread_id;
#if defined(__linux__)
        records[thread_id].current_cpu = sched_getcpu();
#endif
        records[thread_id].affinity = current_affinity();
    }
    std::set<int> current_cpus;
    for (int index=0; index<actual_team_size; ++index) {
        const auto& record = records[index];
        py::dict worker;
        worker["thread_id"] = record.thread_id;
        worker["current_cpu"] = record.current_cpu;
        worker["affinity"] = record.affinity;
        workers.append(std::move(worker));
        if (record.current_cpu >= 0)
            current_cpus.insert(record.current_cpu);
    }
    result["requested_capacity"] = capacity;
    result["actual_team_size"] = actual_team_size;
    result["unique_current_cpus"] = std::vector<int>(
        current_cpus.begin(), current_cpus.end());
#else
    result["requested_capacity"] = py::none();
    result["actual_team_size"] = 0;
    result["unique_current_cpus"] = py::list();
#endif
    result["workers"] = std::move(workers);
    return result;
}

py::dict host_blas_runtime_info_dict()
{
    py::dict result;
    result["environment_policy"] = std::string(
        symmetrix::host_worker_blas_requested_policy());
    result["selected_backend"] = std::string(
        symmetrix::host_worker_blas_selected_backend());
    result["dense_environment_policy"] = std::string(
        symmetrix::host_dense_backend_requested_policy());
    result["dense_selected_backend"] = std::string(
        symmetrix::host_dense_backend_selected_backend());
#if defined(__unix__) || defined(__APPLE__)
    const auto library_path = symmetrix_blas_cblas_library_path();
    if (library_path.empty())
        result["library_path"] = py::none();
    else
        result["library_path"] = canonical_path(library_path);
    result["provider"] = symmetrix_blas_provider();
    result["threading"] = symmetrix_blas_threading_layer();
    const auto threading_library = symmetrix_blas_threading_library_path();
    if (threading_library.empty())
        result["threading_library_path"] = py::none();
    else
        result["threading_library_path"] = canonical_path(threading_library);
    const auto runtime_path = symmetrix_process_openmp_runtime_path();
    if (runtime_path.empty())
        result["openmp_runtime_path"] = py::none();
    else
        result["openmp_runtime_path"] = canonical_path(runtime_path);
    result["openmp_runtime_compatible"] =
        symmetrix_blas_openmp_runtime_compatible();
    const char* config = symmetrix_blas_openblas_config();
    if (config != nullptr)
        result["openblas_config"] = config;
    else
        result["openblas_config"] = py::none();
    const int parallel = symmetrix_blas_parallel_mode();
    if (parallel >= 0) {
        result["openblas_parallel"] = parallel;
        result["openblas_openmp_enabled"] = parallel == 2;
    } else {
        result["openblas_parallel"] = py::none();
        result["openblas_openmp_enabled"] = py::none();
    }
#if defined(KOKKOS_ENABLE_OPENMP)
    if (parallel == 2)
        result["openblas_openmp_runtime_compatible"] =
            symmetrix_blas_openmp_runtime_compatible();
    else
        result["openblas_openmp_runtime_compatible"] = py::none();
#else
    result["openblas_openmp_runtime_compatible"] = py::none();
#endif
    const int openblas_threads = symmetrix_blas_get_num_threads();
    if (openblas_threads > 0)
        result["openblas_threads"] = openblas_threads;
    else
        result["openblas_threads"] = py::none();
#else
    result["library_path"] = py::none();
    result["provider"] = "unknown";
    result["threading"] = "unknown";
    result["threading_library_path"] = py::none();
    result["openmp_runtime_path"] = py::none();
    result["openmp_runtime_compatible"] = py::none();
    result["openblas_config"] = py::none();
    result["openblas_parallel"] = py::none();
    result["openblas_threads"] = py::none();
    result["openblas_openmp_enabled"] = py::none();
    result["openblas_openmp_runtime_compatible"] = py::none();
#endif
    return result;
}

void warn_non_openmp_openblas()
{
#if defined(__unix__) || defined(__APPLE__)
    static bool warned = false;
    if (warned)
        return;
    const char* config = symmetrix_blas_openblas_config();
    const int parallel = symmetrix_blas_parallel_mode();
    if (config == nullptr || std::string(config).find("OpenBLAS") == std::string::npos)
        return;
    if (parallel == 2)
        return;
    int kokkos_threads = 1;
#if defined(KOKKOS_ENABLE_OPENMP)
    if (Kokkos::is_initialized())
        kokkos_threads = Kokkos::DefaultExecutionSpace().concurrency();
    else
        kokkos_threads = omp_get_max_threads();
#endif
    std::cerr << "WARNING: Symmetrix loaded a non-OpenMP OpenBLAS build";
    if (parallel == 1 && kokkos_threads > 1)
        std::cerr << " for a multithreaded CPU Kokkos job; host GEMMs are "
            "forced to one BLAS thread, which is inefficient";
    else if (parallel == 1)
        std::cerr << "; host GEMMs are forced to one BLAS thread";
    else if (kokkos_threads > 1)
        std::cerr << " for a multithreaded CPU Kokkos job; threading mode "
            "could not be identified";
    else
        std::cerr << "; OpenBLAS threading mode could not be identified";
    std::cerr << ". Prefer an OpenMP-enabled OpenBLAS build." << std::endl;
    warned = true;
#endif
}

py::dict concurrent_host_blas_stress(
    const int worker_count, const int iterations, const int size)
{
    if (worker_count <= 0 || iterations <= 0 || size <= 0)
        throw std::invalid_argument(
            "worker_count, iterations, and size must be positive");
    const std::size_t matrix_size = static_cast<std::size_t>(size)*size;
    struct WorkerData {
        std::vector<double> a;
        std::vector<double> b;
        std::vector<double> actual;
        std::vector<double> expected;
    };
    std::vector<WorkerData> data(worker_count);
    for (int worker=0; worker<worker_count; ++worker) {
        auto& item = data[worker];
        item.a.resize(matrix_size);
        item.b.resize(matrix_size);
        item.actual.resize(matrix_size);
        item.expected.resize(matrix_size);
        for (std::size_t index=0; index<matrix_size; ++index) {
            item.a[index] = std::sin(
                static_cast<double>((worker+1)*(index+1))*0.013);
            item.b[index] = std::cos(
                static_cast<double>((worker+3)*(index+1))*0.017);
        }
        for (int row=0; row<size; ++row)
            for (int column=0; column<size; ++column) {
                double value = 0.0;
                for (int inner=0; inner<size; ++inner)
                    value += item.a[static_cast<std::size_t>(row)*size+inner]
                        *item.b[static_cast<std::size_t>(inner)*size+column];
                item.expected[static_cast<std::size_t>(row)*size+column] = value;
            }
    }
    std::barrier start(worker_count);
    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    for (int worker=0; worker<worker_count; ++worker)
        threads.emplace_back([&,worker] {
            start.arrive_and_wait();
            auto& item = data[worker];
            for (int iteration=0; iteration<iterations; ++iteration)
                symmetrix_blas_gemm<double>(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    size, size, size, 1.0,
                    item.a.data(), size, item.b.data(), size,
                    0.0, item.actual.data(), size);
        });
    for (auto& thread : threads)
        thread.join();
    double max_abs_error = 0.0;
    bool finite = true;
    for (const auto& item : data)
        for (std::size_t index=0; index<matrix_size; ++index) {
            finite = finite && std::isfinite(item.actual[index]);
            max_abs_error = std::max(
                max_abs_error,
                std::abs(item.actual[index]-item.expected[index]));
        }
    py::dict result;
    result["workers"] = worker_count;
    result["iterations"] = iterations;
    result["matrix_size"] = size;
    result["finite"] = finite;
    result["max_abs_error"] = max_abs_error;
    result["blas"] = host_blas_runtime_info_dict();
    return result;
}

bool is_openmp_runtime(const std::string& path)
{
    auto filename = std::filesystem::path(path).filename().string();
    std::transform(filename.begin(), filename.end(), filename.begin(),
        [] (const unsigned char value) { return std::tolower(value); });
    const auto starts_with = [&filename] (const std::string& prefix) {
        return filename.rfind(prefix, 0) == 0;
    };
    return starts_with("libgomp")
        || (starts_with("libomp") && !starts_with("libomptarget"))
        || starts_with("libiomp")
        || starts_with("vcomp");
}

bool is_auditwheel_copy_of_expected_runtime(const std::string& path)
{
    auto filename = std::filesystem::path(path).filename().string();
    const auto so = filename.find(".so");
    const auto dash = filename.rfind('-', so);
    if (so == std::string::npos || dash == std::string::npos
        || so-dash-1 != 8)
        return false;
    auto prefix = filename.substr(dash+1, 8);
    std::transform(prefix.begin(), prefix.end(), prefix.begin(),
        [] (const unsigned char value) { return std::tolower(value); });
    if (!std::all_of(prefix.begin(), prefix.end(), [] (const unsigned char value) {
            return std::isxdigit(value);
        }))
        return false;
    auto expected_filename = std::filesystem::path(
        symmetrix::build::expected_openmp_runtime_path).filename().string();
    const auto expected_so = expected_filename.find(".so");
    if (expected_so == std::string::npos)
        return false;
    auto loaded_stem = filename.substr(0, dash);
    auto expected_stem = expected_filename.substr(0, expected_so);
    std::transform(loaded_stem.begin(), loaded_stem.end(), loaded_stem.begin(),
        [] (const unsigned char value) { return std::tolower(value); });
    std::transform(expected_stem.begin(), expected_stem.end(), expected_stem.begin(),
        [] (const unsigned char value) { return std::tolower(value); });
    if (loaded_stem != expected_stem)
        return false;
    std::string expected_hash =
        symmetrix::build::expected_openmp_runtime_sha256;
    std::transform(expected_hash.begin(), expected_hash.end(), expected_hash.begin(),
        [] (const unsigned char value) { return std::tolower(value); });
    return expected_hash.rfind(prefix, 0) == 0;
}

struct OpenMPRuntimeSnapshot {
    std::string loaded_runtime_path;
    std::vector<std::string> loaded_openmp_libraries;
};

#if defined(__linux__)
int collect_openmp_library(
    dl_phdr_info* info, std::size_t, void* raw_libraries)
{
    if (info == nullptr || info->dlpi_name == nullptr || info->dlpi_name[0] == '\0')
        return 0;
    const std::string path = canonical_path(info->dlpi_name);
    if (is_openmp_runtime(path))
        static_cast<std::set<std::string>*>(raw_libraries)->insert(path);
    return 0;
}
#endif

OpenMPRuntimeSnapshot openmp_runtime_snapshot()
{
    OpenMPRuntimeSnapshot result;
#if defined(KOKKOS_ENABLE_OPENMP) && (defined(__unix__) || defined(__APPLE__))
    Dl_info symbol_info{};
    if (dladdr(reinterpret_cast<const void*>(&omp_get_max_threads),
            &symbol_info) != 0
        && symbol_info.dli_fname != nullptr)
        result.loaded_runtime_path = canonical_path(symbol_info.dli_fname);
#endif
#if defined(__linux__)
    std::set<std::string> libraries;
    dl_iterate_phdr(collect_openmp_library, &libraries);
    result.loaded_openmp_libraries.assign(libraries.begin(), libraries.end());
#elif defined(KOKKOS_ENABLE_OPENMP)
    if (!result.loaded_runtime_path.empty())
        result.loaded_openmp_libraries.push_back(result.loaded_runtime_path);
#endif
    return result;
}

std::vector<std::string> native_openmp_runtime_issues(
    const OpenMPRuntimeSnapshot& snapshot)
{
    std::vector<std::string> issues;
    if (!symmetrix::build::openmp_enabled)
        return issues;
    const std::string expected = canonical_path(
        symmetrix::build::expected_openmp_runtime_path);
    if (snapshot.loaded_runtime_path.empty())
        issues.emplace_back("the loaded OpenMP runtime symbol owner is unavailable");
    else if (!expected.empty()
            && !files_are_byte_identical(snapshot.loaded_runtime_path, expected)
            && !is_auditwheel_copy_of_expected_runtime(
                snapshot.loaded_runtime_path))
        issues.emplace_back(
            "loaded OpenMP runtime '"+snapshot.loaded_runtime_path
            +"' is not byte-identical to build runtime '"+expected+"'");
    if (snapshot.loaded_openmp_libraries.size() > 1)
        issues.emplace_back(
            "multiple OpenMP runtimes are loaded in the process");
    return issues;
}

void preflight_openmp_runtime()
{
    const char* raw_policy = std::getenv("SYMMETRIX_OPENMP_RUNTIME_CHECK");
    if (raw_policy == nullptr || raw_policy[0] == '\0')
        return;
    std::string policy(raw_policy);
    std::transform(policy.begin(), policy.end(), policy.begin(),
        [] (const unsigned char value) { return std::tolower(value); });
    if (policy == "0" || policy == "off" || policy == "false")
        return;
    if (policy != "warn" && policy != "strict")
        throw std::invalid_argument(
            "SYMMETRIX_OPENMP_RUNTIME_CHECK must be 'off', 'warn', or 'strict'.");
    const auto issues = native_openmp_runtime_issues(openmp_runtime_snapshot());
    if (issues.empty())
        return;
    std::string message = "Symmetrix OpenMP runtime preflight failed:";
    for (const auto& issue : issues)
        message += "\n- "+issue;
    message += "\nRun 'symmetrix doctor --json' before evaluation.";
    if (policy == "strict")
        throw std::runtime_error(message);
    std::cerr << "WARNING: " << message << std::endl;
}

py::dict openmp_runtime_info_dict()
{
    const auto snapshot = openmp_runtime_snapshot();
    py::dict result;
    result["schema"] = "symmetrix.openmp-runtime-info";
    result["version"] = 1;
    result["enabled"] = symmetrix::build::openmp_enabled;
#if defined(_OPENMP)
    result["compiled_openmp"] = _OPENMP;
#else
    result["compiled_openmp"] = py::none();
#endif
    result["cmake_openmp_version"] = symmetrix::build::openmp_version;
    result["compiler_id"] = symmetrix::build::cxx_compiler_id;
    result["compiler_version"] = symmetrix::build::cxx_compiler_version;
    result["compiler_path"] = symmetrix::build::cxx_compiler_path;
    result["host_arch"] = symmetrix::build::host_arch;
    result["expected_runtime_path"] =
        symmetrix::build::expected_openmp_runtime_path;
    result["expected_runtime_sha256"] =
        symmetrix::build::expected_openmp_runtime_sha256;
    result["loaded_runtime_path"] = snapshot.loaded_runtime_path;
    result["loaded_openmp_libraries"] = snapshot.loaded_openmp_libraries;
    result["kokkos_execution_space"] = Kokkos::DefaultExecutionSpace::name();
    result["kokkos_initialized"] = Kokkos::is_initialized();
#if defined(KOKKOS_ENABLE_OPENMP)
    result["omp_max_threads"] = omp_get_max_threads();
    result["omp_max_active_levels"] = omp_get_max_active_levels();
#else
    result["omp_max_threads"] = py::none();
    result["omp_max_active_levels"] = py::none();
#endif
    if (Kokkos::is_initialized())
        result["kokkos_concurrency"] =
            Kokkos::DefaultExecutionSpace().concurrency();
    else
        result["kokkos_concurrency"] = py::none();
    result["native_preflight_issues"] =
        native_openmp_runtime_issues(snapshot);
    result["worker_sentinel"] = openmp_worker_sentinel_dict();
    result["host_blas"] = host_blas_runtime_info_dict();
    return result;
}

} // namespace

void bind_tools_kokkos(py::module_ &m)
{
    m.def("_init_kokkos", [] {
        preflight_openmp_runtime();
        _init_kokkos();
        warn_non_openmp_openblas();
    }, "Initialize Kokkos after the optional OpenMP runtime preflight.");
    m.def("_finalize_kokkos", &_finalize_kokkos, "TODO");
    m.def("_kokkos_is_initialized", &_kokkos_is_initialized, "TODO");
    m.def("_kokkos_live_object_count", &_kokkos_live_object_count,
        "Return the number of live Symmetrix Kokkos evaluator objects.");
    m.def(
        "_kokkos_default_execution_space",
        &_kokkos_default_execution_space,
        "Return the Kokkos default execution-space name.");
    m.def(
        "_openmp_runtime_info",
        &openmp_runtime_info_dict,
        "Report compiled and actually loaded OpenMP runtime provenance.");
    m.def(
        "_concurrent_host_blas_stress",
        &concurrent_host_blas_stress,
        py::arg("workers") = 8,
        py::arg("iterations") = 100,
        py::arg("size") = 32,
        "Stress concurrent host CBLAS calls using independent buffers.");
    m.def(
        "_execution_device_execution_environment",
        &execution_device_environment_dict,
        "Return normalized host/CUDA/HIP execution metadata.");
    m.def(
        "_kokkos_device_sentinel", &kokkos_device_sentinel,
        "Launch and validate a sentinel kernel on the default execution space.");
    m.def(
        "_team_scratch_admission_for_testing",
        [] (const std::size_t value_count, const std::size_t value_size) {
            return admitted_team_scratch_bytes<>(
                "team-scratch-test", value_count, value_size);
        },
        "Exercise the native per-team scratch admission contract.");
    m.def(
        "_normalize_hip_agent_target",
        [] (const std::string& target) {
            const auto normalized = normalize_hip_agent_target(target);
            py::dict result;
            result["raw_agent_target"] = normalized.raw;
            result["architecture"] = normalized.base_isa;
            result["target_features"] = normalized.features;
            return result;
        });
    m.def(
        "_kokkos_spherical_harmonics_reference",
        &kokkos_spherical_harmonics_reference,
        "Run the portable spherical-harmonic kernel for qualification tests.");
    m.def(
        "_kokkos_periodic_neighbor_graph",
        [] (
            py::array_t<double,py::array::c_style|py::array::forcecast> positions,
            py::array_t<double,py::array::c_style|py::array::forcecast> cell,
            py::array_t<double,py::array::c_style|py::array::forcecast> inverse_cell,
            const double cutoff) {
            const auto graph =
                symmetrix::execution::build_periodic_neighbor_graph_kokkos(
                    std::span<const double>(positions.data(), positions.size()),
                    std::span<const double>(cell.data(), cell.size()),
                    std::span<const double>(
                        inverse_cell.data(), inverse_cell.size()),
                    cutoff);
            const auto host_num_neigh = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), graph.num_neigh);
            const auto host_offsets = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), graph.receiver_offsets);
            const auto host_sources = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), graph.sources);
            const auto host_shifts = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), graph.shifts);
            const auto host_fractional_xyz = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), graph.fractional_xyz);
            py::array_t<int> num_neigh(graph.num_nodes);
            py::array_t<int> offsets(graph.num_nodes+1);
            py::array_t<int> sources(graph.num_edges);
            py::array_t<int> shifts(
                std::vector<py::ssize_t>{
                    static_cast<py::ssize_t>(graph.num_edges), 3});
            py::array_t<double> fractional_xyz(
                std::vector<py::ssize_t>{
                    static_cast<py::ssize_t>(graph.num_edges), 3});
            std::copy(
                host_num_neigh.data(),
                host_num_neigh.data()+graph.num_nodes,
                num_neigh.mutable_data());
            std::copy(
                host_offsets.data(),
                host_offsets.data()+graph.num_nodes+1,
                offsets.mutable_data());
            std::copy(
                host_sources.data(),
                host_sources.data()+graph.num_edges,
                sources.mutable_data());
            const std::size_t coordinate_count =
                std::size_t(3)*graph.num_edges;
            std::copy(
                host_shifts.data(), host_shifts.data()+coordinate_count,
                shifts.mutable_data());
            std::copy(
                host_fractional_xyz.data(),
                host_fractional_xyz.data()+coordinate_count,
                fractional_xyz.mutable_data());
            py::dict result;
            result["num_neigh"] = std::move(num_neigh);
            result["receiver_offsets"] = std::move(offsets);
            result["sources"] = std::move(sources);
            result["shifts"] = std::move(shifts);
            result["fractional_xyz"] = std::move(fractional_xyz);
            return result;
        },
        py::arg("positions"), py::arg("cell"), py::arg("inverse_cell"),
        py::arg("cutoff"),
        "Build and export a fully periodic Kokkos neighbor graph for qualification.");
}
