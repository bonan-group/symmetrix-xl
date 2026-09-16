#include "jit_nvrtc.hpp"

#if __has_include(<Kokkos_Macros.hpp>)
#include <Kokkos_Macros.hpp>
#endif

#if defined(KOKKOS_ENABLE_CUDA) && (defined(__unix__) || defined(__APPLE__))
#include <dlfcn.h>
#include <nvrtc.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace symmetrix::execution {
namespace {

class NvrtcApi {
public:
    NvrtcApi()
    {
        if (const char* explicit_library =
                std::getenv("SYMMETRIX_JIT_NVRTC_LIBRARY");
            explicit_library != nullptr && explicit_library[0] != '\0') {
            handle_ = dlopen(explicit_library, RTLD_NOW | RTLD_LOCAL);
            if (handle_ != nullptr)
                library_ = explicit_library;
        }
        constexpr std::array candidates {
            "libnvrtc.so",
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so.11.2",
        };
        for (const char* candidate : candidates) {
            if (handle_ != nullptr)
                break;
            handle_ = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
            if (handle_ != nullptr) {
                library_ = candidate;
                break;
            }
        }
        if (handle_ == nullptr) {
            const char* error = dlerror();
            reason_ = error == nullptr ? "NVRTC shared library was not found" : error;
            return;
        }
        try {
            version = symbol<decltype(&nvrtcVersion)>("nvrtcVersion");
            get_num_supported_archs = symbol<decltype(&nvrtcGetNumSupportedArchs)>(
                "nvrtcGetNumSupportedArchs");
            get_supported_archs = symbol<decltype(&nvrtcGetSupportedArchs)>(
                "nvrtcGetSupportedArchs");
            create_program = symbol<decltype(&nvrtcCreateProgram)>("nvrtcCreateProgram");
            destroy_program = symbol<decltype(&nvrtcDestroyProgram)>("nvrtcDestroyProgram");
            compile_program = symbol<decltype(&nvrtcCompileProgram)>("nvrtcCompileProgram");
            get_program_log_size = symbol<decltype(&nvrtcGetProgramLogSize)>(
                "nvrtcGetProgramLogSize");
            get_program_log = symbol<decltype(&nvrtcGetProgramLog)>("nvrtcGetProgramLog");
            get_cubin_size = symbol<decltype(&nvrtcGetCUBINSize)>("nvrtcGetCUBINSize");
            get_cubin = symbol<decltype(&nvrtcGetCUBIN)>("nvrtcGetCUBIN");
            get_error_string = symbol<decltype(&nvrtcGetErrorString)>("nvrtcGetErrorString");
        } catch (const std::exception& error) {
            reason_ = error.what();
            dlclose(handle_);
            handle_ = nullptr;
        }
    }

    ~NvrtcApi()
    {
        if (handle_ != nullptr)
            dlclose(handle_);
    }

    NvrtcApi(const NvrtcApi&) = delete;
    NvrtcApi& operator=(const NvrtcApi&) = delete;

    bool available() const noexcept { return handle_ != nullptr; }
    const std::string& library() const noexcept { return library_; }
    const std::string& reason() const noexcept { return reason_; }

    decltype(&nvrtcVersion) version = nullptr;
    decltype(&nvrtcGetNumSupportedArchs) get_num_supported_archs = nullptr;
    decltype(&nvrtcGetSupportedArchs) get_supported_archs = nullptr;
    decltype(&nvrtcCreateProgram) create_program = nullptr;
    decltype(&nvrtcDestroyProgram) destroy_program = nullptr;
    decltype(&nvrtcCompileProgram) compile_program = nullptr;
    decltype(&nvrtcGetProgramLogSize) get_program_log_size = nullptr;
    decltype(&nvrtcGetProgramLog) get_program_log = nullptr;
    decltype(&nvrtcGetCUBINSize) get_cubin_size = nullptr;
    decltype(&nvrtcGetCUBIN) get_cubin = nullptr;
    decltype(&nvrtcGetErrorString) get_error_string = nullptr;

private:
    template<typename Function>
    Function symbol(const char* name)
    {
        dlerror();
        void* address = dlsym(handle_, name);
        const char* error = dlerror();
        if (error != nullptr || address == nullptr)
            throw std::runtime_error(
                std::string("Could not resolve NVRTC symbol ") + name + ": "
                + (error == nullptr ? "symbol is null" : error));
        return reinterpret_cast<Function>(address);
    }

    void* handle_ = nullptr;
    std::string library_;
    std::string reason_;
};

NvrtcApi& api()
{
    static NvrtcApi instance;
    return instance;
}

std::runtime_error nvrtc_error(
    const NvrtcApi& value, nvrtcResult status, std::string_view operation)
{
    const char* message = value.get_error_string == nullptr
        ? nullptr : value.get_error_string(status);
    return std::runtime_error(
        std::string(operation) + " failed: "
        + (message == nullptr ? "unknown NVRTC error" : message));
}

class Program {
public:
    Program(NvrtcApi& api, std::string_view source) : api_(api)
    {
        const std::string owned_source(source);
        const auto status = api_.create_program(
            &program_, owned_source.c_str(), "execution_module.cu", 0, nullptr, nullptr);
        if (status != NVRTC_SUCCESS)
            throw nvrtc_error(api_, status, "nvrtcCreateProgram");
    }

    ~Program()
    {
        if (program_ != nullptr)
            api_.destroy_program(&program_);
    }

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    nvrtcProgram get() const noexcept { return program_; }

private:
    NvrtcApi& api_;
    nvrtcProgram program_ = nullptr;
};

std::string program_log(NvrtcApi& value, nvrtcProgram program)
{
    std::size_t size = 0;
    auto status = value.get_program_log_size(program, &size);
    if (status != NVRTC_SUCCESS)
        throw nvrtc_error(value, status, "nvrtcGetProgramLogSize");
    if (size <= 1)
        return {};
    std::string log(size, '\0');
    status = value.get_program_log(program, log.data());
    if (status != NVRTC_SUCCESS)
        throw nvrtc_error(value, status, "nvrtcGetProgramLog");
    while (!log.empty() && log.back() == '\0')
        log.pop_back();
    return log;
}

}  // namespace

NvrtcInformation nvrtc_information()
{
    auto& value = api();
    NvrtcInformation result;
    result.available = value.available();
    result.library = value.library();
    result.reason = value.reason();
    if (!value.available())
        return result;

    auto status = value.version(&result.major, &result.minor);
    if (status != NVRTC_SUCCESS)
        throw nvrtc_error(value, status, "nvrtcVersion");
    int count = 0;
    status = value.get_num_supported_archs(&count);
    if (status != NVRTC_SUCCESS)
        throw nvrtc_error(value, status, "nvrtcGetNumSupportedArchs");
    if (count < 0)
        throw std::runtime_error("NVRTC returned a negative architecture count");
    result.supported_architectures.resize(static_cast<std::size_t>(count));
    if (count != 0) {
        status = value.get_supported_archs(result.supported_architectures.data());
        if (status != NVRTC_SUCCESS)
            throw nvrtc_error(value, status, "nvrtcGetSupportedArchs");
    }
    return result;
}

NvrtcCompilation compile_cuda_with_nvrtc(
    std::string_view source,
    const std::vector<std::string>& options)
{
    auto& value = api();
    if (!value.available())
        throw std::runtime_error(
            "NVRTC is unavailable: "
            + (value.reason().empty() ? std::string("library not found") : value.reason()));
    if (source.empty())
        throw std::invalid_argument("NVRTC source must not be empty");

    int major = 0;
    int minor = 0;
    auto status = value.version(&major, &minor);
    if (status != NVRTC_SUCCESS)
        throw nvrtc_error(value, status, "nvrtcVersion");

    Program program(value, source);
    std::vector<const char*> option_pointers;
    option_pointers.reserve(options.size());
    for (const auto& option : options) {
        if (option.empty())
            throw std::invalid_argument("NVRTC options must not contain empty strings");
        option_pointers.push_back(option.c_str());
    }
    status = value.compile_program(
        program.get(), static_cast<int>(option_pointers.size()), option_pointers.data());
    const std::string log = program_log(value, program.get());
    if (status != NVRTC_SUCCESS)
        throw std::runtime_error(
            std::string("nvrtcCompileProgram failed: ")
            + value.get_error_string(status)
            + (log.empty() ? std::string() : std::string("\n") + log));

    std::size_t cubin_size = 0;
    status = value.get_cubin_size(program.get(), &cubin_size);
    if (status != NVRTC_SUCCESS)
        throw nvrtc_error(value, status, "nvrtcGetCUBINSize");
    if (cubin_size == 0)
        throw std::runtime_error("NVRTC produced an empty cubin");
    NvrtcCompilation result;
    result.cubin.resize(cubin_size);
    status = value.get_cubin(program.get(), reinterpret_cast<char*>(result.cubin.data()));
    if (status != NVRTC_SUCCESS)
        throw nvrtc_error(value, status, "nvrtcGetCUBIN");
    result.log = log;
    result.major = major;
    result.minor = minor;
    return result;
}

}  // namespace symmetrix::execution

#else

#include <stdexcept>

namespace symmetrix::execution {

NvrtcInformation nvrtc_information()
{
    return {false, {}, "this Symmetrix build does not enable Kokkos CUDA", 0, 0, {}};
}

NvrtcCompilation compile_cuda_with_nvrtc(
    std::string_view,
    const std::vector<std::string>&)
{
    throw std::runtime_error(
        "NVRTC compilation requires a Kokkos-CUDA Symmetrix build");
}

}  // namespace symmetrix::execution

#endif
