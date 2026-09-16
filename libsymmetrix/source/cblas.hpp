#pragma once

#include <string>
#include <type_traits>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#if defined(__linux__)
#include <link.h>
#endif

#include "../external/cblas-prototypes/cblas.h"

inline int symmetrix_blas_parallel_mode();
inline bool symmetrix_blas_provider_links_process_openmp_runtime();

inline std::string symmetrix_blas_cblas_library_path()
{
#if defined(__unix__) || defined(__APPLE__)
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&cblas_dgemm), &info) != 0
            && info.dli_fname != nullptr)
        return info.dli_fname;
#endif
    return {};
}

inline std::vector<std::string> symmetrix_blas_loaded_library_paths()
{
    std::vector<std::string> paths;
#if defined(__linux__)
    dl_iterate_phdr([] (struct dl_phdr_info* info, std::size_t, void* data) {
        auto* paths = static_cast<std::vector<std::string>*>(data);
        if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0')
            paths->emplace_back(info->dlpi_name);
        return 0;
    }, &paths);
#endif
    return paths;
}

inline bool symmetrix_blas_path_contains(
    const std::vector<std::string>& paths, const char* token)
{
    for (const auto& path : paths)
        if (path.find(token) != std::string::npos)
            return true;
    return false;
}

inline bool symmetrix_blas_path_is_openmp_runtime(const std::string& path)
{
    // Matches system runtime names (libgomp.so.1) as well as auditwheel-hashed
    // vendored names (libgomp-d46f9225.so.1.0.0) produced by `auditwheel repair`.
    const bool is_llvm_openmp = path.find("libomp.so") != std::string::npos
        && path.find("libomptarget") == std::string::npos;
    return path.find("libgomp.so") != std::string::npos
        || path.find("libgomp-") != std::string::npos
        || path.find("libiomp") != std::string::npos
        || is_llvm_openmp;
}

inline std::size_t symmetrix_blas_loaded_openmp_runtime_count()
{
    std::size_t count = 0;
    for (const auto& path : symmetrix_blas_loaded_library_paths()) {
        if (symmetrix_blas_path_is_openmp_runtime(path))
            ++count;
    }
    return count;
}

inline std::string symmetrix_blas_provider()
{
    const auto path = symmetrix_blas_cblas_library_path();
    if (path.find("openblas") != std::string::npos)
        return "openblas";
    if (path.find("mkl_") != std::string::npos
            || path.find("libmkl_rt") != std::string::npos)
        return "mkl";
    return "unknown";
}

inline std::string symmetrix_blas_threading_layer()
{
    const auto paths = symmetrix_blas_loaded_library_paths();
    if (symmetrix_blas_provider() == "mkl") {
        if (symmetrix_blas_path_contains(paths, "libmkl_sequential"))
            return "sequential";
        if (symmetrix_blas_path_contains(paths, "libmkl_gnu_thread"))
            return "gnu_openmp";
        if (symmetrix_blas_path_contains(paths, "libmkl_intel_thread")
                || symmetrix_blas_path_contains(paths, "libiomp5"))
            return "intel_openmp";
        if (symmetrix_blas_path_contains(paths, "libmkl_tbb_thread"))
            return "tbb";
        return "unknown";
    }
    if (symmetrix_blas_provider() == "openblas") {
        const int parallel = symmetrix_blas_parallel_mode();
        if (parallel == 2)
            return "openmp";
        if (parallel >= 0)
            return "pthread";
    }
    return "unknown";
}

inline std::string symmetrix_blas_threading_library_path()
{
    const auto paths = symmetrix_blas_loaded_library_paths();
    const auto layer = symmetrix_blas_threading_layer();
    const char* token = nullptr;
    if (layer == "gnu_openmp")
        token = "libmkl_gnu_thread";
    else if (layer == "intel_openmp")
        token = "libmkl_intel_thread";
    else if (layer == "sequential")
        token = "libmkl_sequential";
    else if (layer == "tbb")
        token = "libmkl_tbb_thread";
    else if (layer == "openmp" || layer == "pthread")
        return symmetrix_blas_cblas_library_path();
    if (token != nullptr)
        for (const auto& path : paths)
            if (path.find(token) != std::string::npos)
                return path;
    return {};
}

inline std::string symmetrix_process_openmp_runtime_path()
{
#if defined(__unix__) || defined(__APPLE__)
    Dl_info info{};
    void* symbol = nullptr;
#if defined(__GNUC__) && !defined(__clang__)
    symbol = dlsym(RTLD_DEFAULT, "GOMP_parallel");
#endif
    if (symbol == nullptr)
        symbol = dlsym(RTLD_DEFAULT, "__kmpc_fork_call");
    if (symbol != nullptr && dladdr(symbol, &info) != 0
            && info.dli_fname != nullptr)
        return info.dli_fname;
#endif
    return {};
}

inline bool symmetrix_blas_openmp_runtime_compatible()
{
    const auto layer = symmetrix_blas_threading_layer();
    if (layer == "sequential")
        return true;
    const auto runtime = symmetrix_process_openmp_runtime_path();
    if (runtime.empty())
        return false;
    if (layer == "gnu_openmp")
        return runtime.find("libgomp") != std::string::npos
            && symmetrix_blas_loaded_openmp_runtime_count() == 1;
    if (layer == "intel_openmp")
        return (runtime.find("libiomp") != std::string::npos
                || runtime.find("libomp") != std::string::npos)
            && symmetrix_blas_loaded_openmp_runtime_count() == 1;
    if (layer == "openmp")
        return symmetrix_blas_provider_links_process_openmp_runtime()
            && symmetrix_blas_loaded_openmp_runtime_count() == 1;
    return false;
}

inline bool symmetrix_blas_allows_concurrent_cblas()
{
    static const auto provider = symmetrix_blas_provider();
    static const auto layer = symmetrix_blas_threading_layer();
    if (provider == "mkl" && layer == "sequential")
        return true;
    if ((provider == "mkl" && layer == "gnu_openmp")
            || (provider == "openblas" && layer == "openmp"))
        return symmetrix_blas_openmp_runtime_compatible();
    return false;
}

inline bool symmetrix_blas_symbol_matches_cblas_provider(const void* symbol)
{
#if defined(__unix__) || defined(__APPLE__)
    if (symbol == nullptr)
        return false;
    Dl_info cblas_info{};
    Dl_info symbol_info{};
    return dladdr(reinterpret_cast<const void*>(&cblas_dgemm), &cblas_info) != 0
        && dladdr(symbol, &symbol_info) != 0
        && cblas_info.dli_fbase != nullptr
        && cblas_info.dli_fbase == symbol_info.dli_fbase;
#else
    (void)symbol;
    return false;
#endif
}

inline void* symmetrix_blas_provider_symbol(
    const char* name, const char* alternate_name = nullptr)
{
#if defined(__unix__) || defined(__APPLE__)
    void* symbol = dlsym(RTLD_DEFAULT, name);
    if (!symmetrix_blas_symbol_matches_cblas_provider(symbol)
            && alternate_name != nullptr)
        symbol = dlsym(RTLD_DEFAULT, alternate_name);
    return symmetrix_blas_symbol_matches_cblas_provider(symbol)
        ? symbol : nullptr;
#else
    (void)name;
    (void)alternate_name;
    return nullptr;
#endif
}

inline bool symmetrix_blas_provider_links_process_openmp_runtime()
{
#if defined(__linux__)
    const char* anchor_name = nullptr;
#if defined(__clang__)
    anchor_name = "__kmpc_fork_call";
#elif defined(__GNUC__)
    anchor_name = "GOMP_parallel";
#endif
    if (anchor_name == nullptr)
        return false;
    void* runtime_anchor = dlsym(RTLD_DEFAULT, anchor_name);
    Dl_info cblas_info{};
    Dl_info runtime_info{};
    if (dladdr(reinterpret_cast<const void*>(&cblas_dgemm), &cblas_info) == 0
            || cblas_info.dli_fname == nullptr
            || runtime_anchor == nullptr
            || dladdr(runtime_anchor, &runtime_info) == 0
            || runtime_info.dli_fname == nullptr)
        return false;
    int flags = RTLD_LAZY;
#if defined(RTLD_NOLOAD)
    flags |= RTLD_NOLOAD;
#endif
    void* provider = dlopen(cblas_info.dli_fname, flags);
    void* runtime = dlopen(runtime_info.dli_fname, flags);
    if (provider == nullptr || runtime == nullptr) {
        if (provider != nullptr)
            dlclose(provider);
        if (runtime != nullptr)
            dlclose(runtime);
        return false;
    }
    link_map* provider_map = nullptr;
    link_map* runtime_map = nullptr;
    if (dlinfo(provider, RTLD_DI_LINKMAP, &provider_map) != 0
            || dlinfo(runtime, RTLD_DI_LINKMAP, &runtime_map) != 0
            || provider_map == nullptr || runtime_map == nullptr) {
        dlclose(runtime);
        dlclose(provider);
        return false;
    }
    const auto string_table = [] (const link_map* map) {
        for (const auto* item=map->l_ld; item->d_tag != DT_NULL; ++item)
            if (item->d_tag == DT_STRTAB)
                return reinterpret_cast<const char*>(item->d_un.d_ptr);
        return static_cast<const char*>(nullptr);
    };
    const char* runtime_strings = string_table(runtime_map);
    const char* runtime_soname = nullptr;
    if (runtime_strings != nullptr)
        for (const auto* item=runtime_map->l_ld;
                item->d_tag != DT_NULL; ++item)
            if (item->d_tag == DT_SONAME) {
                runtime_soname = runtime_strings+item->d_un.d_val;
                break;
            }
    const std::string expected = runtime_soname == nullptr
        ? std::string{} : std::string(runtime_soname);
    const char* provider_strings = string_table(provider_map);
    bool matched = false;
    if (!expected.empty() && provider_strings != nullptr)
        for (const auto* item=provider_map->l_ld;
                item->d_tag != DT_NULL; ++item)
            if (item->d_tag == DT_NEEDED
                    && expected == provider_strings+item->d_un.d_val) {
                matched = true;
                break;
            }
    dlclose(runtime);
    dlclose(provider);
    return matched;
#else
    return false;
#endif
}

inline const char* symmetrix_blas_openblas_config()
{
    using ConfigFunction = const char* (*)();
    void* symbol = symmetrix_blas_provider_symbol(
        "openblas_get_config", "openblas_get_config64_");
    return symbol == nullptr
        ? nullptr : reinterpret_cast<ConfigFunction>(symbol)();
}

inline int symmetrix_blas_parallel_mode()
{
    using ParallelFunction = int (*)();
    void* symbol = symmetrix_blas_provider_symbol(
        "openblas_get_parallel", "openblas_get_parallel64_");
    return symbol == nullptr
        ? -1 : reinterpret_cast<ParallelFunction>(symbol)();
}

inline bool symmetrix_blas_openmp_enabled()
{
    return symmetrix_blas_parallel_mode() == 2;
}

inline int symmetrix_blas_get_num_threads()
{
    using ThreadCountFunction = int (*)();
    void* symbol = symmetrix_blas_provider_symbol(
        "openblas_get_num_threads", "openblas_get_num_threads64_");
    return symbol == nullptr
        ? 0 : reinterpret_cast<ThreadCountFunction>(symbol)();
}

inline int symmetrix_blas_set_num_threads(int threads);

inline int symmetrix_blas_configure_host_threads()
{
    static const int configured = [] {
        // OpenMP OpenBLAS shares the Kokkos OpenMP runtime; changing its team
        // size here would interfere with the outer execution space. Leave
        // that build alone.
        if (symmetrix_blas_openmp_enabled())
            return symmetrix_blas_get_num_threads();
        // A pthread/serial OpenBLAS team must not nest inside Kokkos workers.
        return symmetrix_blas_set_num_threads(1);
    }();
    return configured;
}

inline int symmetrix_blas_set_num_threads(const int threads)
{
    using SetThreadCountFunction = void (*)(int);
    void* symbol = symmetrix_blas_provider_symbol(
        "openblas_set_num_threads", "openblas_set_num_threads64_");
    if (symbol == nullptr)
        return 0;
    reinterpret_cast<SetThreadCountFunction>(symbol)(threads);
    const int configured = symmetrix_blas_get_num_threads();
    return configured == 0 ? threads : configured;
}

template <typename Precision>
inline void symmetrix_blas_gemm(
    const CBLAS_ORDER order,
    const CBLAS_TRANSPOSE trans_a,
    const CBLAS_TRANSPOSE trans_b,
    const int m,
    const int n,
    const int k,
    const Precision alpha,
    const Precision* a,
    const int lda,
    const Precision* b,
    const int ldb,
    const Precision beta,
    Precision* c,
    const int ldc)
{
    static_assert(std::is_same_v<Precision, float>
                  || std::is_same_v<Precision, double>);
    symmetrix_blas_configure_host_threads();
    if constexpr (std::is_same_v<Precision, float>)
        cblas_sgemm(
            order, trans_a, trans_b, m, n, k,
            alpha, a, lda, b, ldb, beta, c, ldc);
    else
        cblas_dgemm(
            order, trans_a, trans_b, m, n, k,
            alpha, a, lda, b, ldb, beta, c, ldc);
}

template <typename Precision>
inline void symmetrix_blas_gemv(
    const CBLAS_ORDER order,
    const CBLAS_TRANSPOSE trans,
    const int m,
    const int n,
    const Precision alpha,
    const Precision* a,
    const int lda,
    const Precision* x,
    const int incx,
    const Precision beta,
    Precision* y,
    const int incy)
{
    static_assert(std::is_same_v<Precision, float>
                  || std::is_same_v<Precision, double>);
    symmetrix_blas_configure_host_threads();
    if constexpr (std::is_same_v<Precision, float>)
        cblas_sgemv(
            order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    else
        cblas_dgemv(
            order, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <typename Precision>
inline void symmetrix_blas_copy(
    const int n,
    const Precision* x,
    const int incx,
    Precision* y,
    const int incy)
{
    static_assert(std::is_same_v<Precision, float>
                  || std::is_same_v<Precision, double>);
    if constexpr (std::is_same_v<Precision, float>)
        cblas_scopy(n, x, incx, y, incy);
    else
        cblas_dcopy(n, x, incx, y, incy);
}

template <typename Precision>
inline Precision symmetrix_blas_dot(
    const int n,
    const Precision* x,
    const int incx,
    const Precision* y,
    const int incy)
{
    static_assert(std::is_same_v<Precision, float>
                  || std::is_same_v<Precision, double>);
    if constexpr (std::is_same_v<Precision, float>)
        return cblas_sdot(n, x, incx, y, incy);
    else
        return cblas_ddot(n, x, incx, y, incy);
}
