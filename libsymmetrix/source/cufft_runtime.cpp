#include "cufft_runtime.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>

#ifndef SYMMETRIX_CUFFT_ABI_MAJOR
#error "The cuFFT runtime ABI must be selected by the CUDA build"
#endif

namespace symmetrix::cufft_runtime {
namespace {

struct State {
    std::mutex mutex;
    Api functions;
    Information report;
    std::string configured;
    void* handle = nullptr;
};

State& state()
{
    // Plans and pending CUDA work can survive other static destructors.
    // Keep the vendor handle and its dispatch table alive for process lifetime.
    static State* instance = new State;
    return *instance;
}

std::string install_hint()
{
    return SYMMETRIX_CUFFT_ABI_MAJOR == 12
        ? "Install the matching NVIDIA runtime with 'pip install nvidia-cufft'."
        : "Install the matching NVIDIA runtime with 'pip install nvidia-cufft-cu12'.";
}

template<class Function>
Function resolve(void* handle, const char* name)
{
    dlerror();
    void* address = dlsym(handle, name);
    const char* error = dlerror();
    if (error != nullptr || address == nullptr)
        throw std::runtime_error(std::string("missing cuFFT symbol ") + name);
    return reinterpret_cast<Function>(address);
}

const Api& load()
{
    auto& value = state();
    std::lock_guard lock(value.mutex);
    if (value.handle != nullptr)
        return value.functions;
    std::string candidate = value.configured;
    if (const char* override_path = std::getenv("SYMMETRIX_CUFFT_LIBRARY");
            override_path != nullptr && override_path[0] != '\0') {
        candidate = override_path;
        if (candidate.front() != '/') {
            value.report.state = "probe-failed";
            value.report.reason = "SYMMETRIX_CUFFT_LIBRARY must be an absolute path. "
                + install_hint();
            throw std::runtime_error(value.report.reason);
        }
    }
    if (candidate.empty())
        candidate = "libcufft.so." + std::to_string(SYMMETRIX_CUFFT_ABI_MAJOR);
    void* handle = nullptr;
    try {
        handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error(error == nullptr ? "dlopen failed" : error);
        }
        Api functions;
#define SYMMETRIX_CUFFT_RESOLVE(name) \
        functions.name = resolve<decltype(functions.name)>(handle, #name);
        SYMMETRIX_CUFFT_FUNCTIONS(SYMMETRIX_CUFFT_RESOLVE)
#undef SYMMETRIX_CUFFT_RESOLVE
        int version = 0;
        if (functions.cufftGetVersion(&version) != CUFFT_SUCCESS
                || version / 1000 != SYMMETRIX_CUFFT_ABI_MAJOR)
            throw std::runtime_error("incompatible cuFFT version "
                + std::to_string(version) + "; expected ABI "
                + std::to_string(SYMMETRIX_CUFFT_ABI_MAJOR));
        Dl_info identity{};
        const bool located = dladdr(
            reinterpret_cast<void*>(functions.cufftGetVersion), &identity) != 0;
        value.report.library = located && identity.dli_fname != nullptr
            ? identity.dli_fname : candidate;
        value.report.version = version;
        value.report.available = true;
        value.report.state = "loaded";
        value.report.reason.clear();
        value.functions = functions;
        value.handle = handle;
        return value.functions;
    } catch (const std::exception& error) {
        if (handle != nullptr)
            dlclose(handle);
        value.report.state = "probe-failed";
        value.report.available = false;
        value.report.library = candidate;
        value.report.reason = "Could not load cuFFT from " + candidate + ": "
            + error.what() + ". " + install_hint();
        throw std::runtime_error(value.report.reason);
    }
}

}  // namespace

void configure_library(const std::string& path)
{
    if (!path.empty() && path.front() != '/')
        throw std::invalid_argument("cuFFT library path must be absolute");
    auto& value = state();
    std::lock_guard lock(value.mutex);
    if (value.handle != nullptr && path != value.configured)
        throw std::runtime_error("cuFFT is already loaded; its provider cannot change");
    value.configured = path;
}

const Api& api()
{
    // C++ retries initialization after a failed load and serializes callers.
    static const Api& functions = load();
    return functions;
}

Information information(bool probe)
{
    if (probe) {
        try { (void)api(); }
        catch (const std::exception&) { /* The report retains the failure. */ }
    }
    auto& value = state();
    std::lock_guard lock(value.mutex);
    return value.report;
}

}  // namespace symmetrix::cufft_runtime
