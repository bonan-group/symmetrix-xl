#include "jit_hiprtc.hpp"

#if __has_include(<Kokkos_Macros.hpp>)
#include <Kokkos_Macros.hpp>
#endif

#if defined(KOKKOS_ENABLE_HIP) && (defined(__unix__) || defined(__APPLE__))
#include <dlfcn.h>
#include <hip/hiprtc.h>

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace symmetrix::execution {
namespace {

class HiprtcApi {
public:
    HiprtcApi()
    {
        if (const char* explicit_library =
                std::getenv("SYMMETRIX_JIT_HIPRTC_LIBRARY");
            explicit_library != nullptr && explicit_library[0] != '\0') {
            handle_ = dlopen(explicit_library, RTLD_NOW | RTLD_LOCAL);
            if (handle_ != nullptr)
                library_ = explicit_library;
        }
        constexpr std::array candidates {
            "libhiprtc.so",
            "libhiprtc.so.7",
            "/opt/rocm/lib/libhiprtc.so",
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
            reason_ = error == nullptr
                ? "hipRTC shared library was not found" : error;
            return;
        }
        try {
            version = symbol<decltype(&hiprtcVersion)>("hiprtcVersion");
            create_program = symbol<decltype(&hiprtcCreateProgram)>(
                "hiprtcCreateProgram");
            destroy_program = symbol<decltype(&hiprtcDestroyProgram)>(
                "hiprtcDestroyProgram");
            compile_program = symbol<decltype(&hiprtcCompileProgram)>(
                "hiprtcCompileProgram");
            get_program_log_size = symbol<decltype(&hiprtcGetProgramLogSize)>(
                "hiprtcGetProgramLogSize");
            get_program_log = symbol<decltype(&hiprtcGetProgramLog)>(
                "hiprtcGetProgramLog");
            get_code_size = symbol<decltype(&hiprtcGetCodeSize)>(
                "hiprtcGetCodeSize");
            get_code = symbol<decltype(&hiprtcGetCode)>("hiprtcGetCode");
            get_error_string = symbol<decltype(&hiprtcGetErrorString)>(
                "hiprtcGetErrorString");
        } catch (const std::exception& error) {
            reason_ = error.what();
            dlclose(handle_);
            handle_ = nullptr;
        }
    }

    ~HiprtcApi()
    {
        if (handle_ != nullptr)
            dlclose(handle_);
    }

    HiprtcApi(const HiprtcApi&) = delete;
    HiprtcApi& operator=(const HiprtcApi&) = delete;

    bool available() const noexcept { return handle_ != nullptr; }
    const std::string& library() const noexcept { return library_; }
    const std::string& reason() const noexcept { return reason_; }

    decltype(&hiprtcVersion) version = nullptr;
    decltype(&hiprtcCreateProgram) create_program = nullptr;
    decltype(&hiprtcDestroyProgram) destroy_program = nullptr;
    decltype(&hiprtcCompileProgram) compile_program = nullptr;
    decltype(&hiprtcGetProgramLogSize) get_program_log_size = nullptr;
    decltype(&hiprtcGetProgramLog) get_program_log = nullptr;
    decltype(&hiprtcGetCodeSize) get_code_size = nullptr;
    decltype(&hiprtcGetCode) get_code = nullptr;
    decltype(&hiprtcGetErrorString) get_error_string = nullptr;

private:
    template<typename Function>
    Function symbol(const char* name)
    {
        dlerror();
        void* address = dlsym(handle_, name);
        const char* error = dlerror();
        if (error != nullptr || address == nullptr)
            throw std::runtime_error(
                std::string("Could not resolve hipRTC symbol ") + name + ": "
                + (error == nullptr ? "symbol is null" : error));
        return reinterpret_cast<Function>(address);
    }

    void* handle_ = nullptr;
    std::string library_;
    std::string reason_;
};

HiprtcApi& api()
{
    static HiprtcApi instance;
    return instance;
}

std::runtime_error hiprtc_error(
    const HiprtcApi& value, hiprtcResult status, std::string_view operation)
{
    const char* message = value.get_error_string == nullptr
        ? nullptr : value.get_error_string(status);
    return std::runtime_error(
        std::string(operation) + " failed: "
        + (message == nullptr ? "unknown hipRTC error" : message));
}

class Program {
public:
    Program(HiprtcApi& api, std::string_view source) : api_(api)
    {
        const std::string owned_source(source);
        const auto status = api_.create_program(
            &program_, owned_source.c_str(), "execution_module.hip", 0, nullptr, nullptr);
        if (status != HIPRTC_SUCCESS)
            throw hiprtc_error(api_, status, "hiprtcCreateProgram");
    }

    ~Program()
    {
        if (program_ != nullptr)
            api_.destroy_program(&program_);
    }

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    hiprtcProgram get() const noexcept { return program_; }

private:
    HiprtcApi& api_;
    hiprtcProgram program_ = nullptr;
};

std::string program_log(HiprtcApi& value, hiprtcProgram program)
{
    std::size_t size = 0;
    auto status = value.get_program_log_size(program, &size);
    if (status != HIPRTC_SUCCESS)
        throw hiprtc_error(value, status, "hiprtcGetProgramLogSize");
    if (size <= 1)
        return {};
    std::string log(size, '\0');
    status = value.get_program_log(program, log.data());
    if (status != HIPRTC_SUCCESS)
        throw hiprtc_error(value, status, "hiprtcGetProgramLog");
    while (!log.empty() && log.back() == '\0')
        log.pop_back();
    return log;
}

}  // namespace

HiprtcInformation hiprtc_information()
{
    auto& value = api();
    HiprtcInformation result;
    result.available = value.available();
    result.library = value.library();
    result.reason = value.reason();
    if (!value.available())
        return result;
    const auto status = value.version(&result.major, &result.minor);
    if (status != HIPRTC_SUCCESS)
        throw hiprtc_error(value, status, "hiprtcVersion");
    return result;
}

HiprtcCompilation compile_hip_with_hiprtc(
    std::string_view source,
    const std::vector<std::string>& options)
{
    auto& value = api();
    if (!value.available())
        throw std::runtime_error(
            "hipRTC is unavailable: "
            + (value.reason().empty()
                ? std::string("library not found") : value.reason()));
    if (source.empty())
        throw std::invalid_argument("hipRTC source must not be empty");

    HiprtcCompilation result;
    auto status = value.version(&result.major, &result.minor);
    if (status != HIPRTC_SUCCESS)
        throw hiprtc_error(value, status, "hiprtcVersion");
    Program program(value, source);
    std::vector<const char*> option_pointers;
    option_pointers.reserve(options.size());
    for (const auto& option : options) {
        if (option.empty())
            throw std::invalid_argument(
                "hipRTC options must not contain empty strings");
        option_pointers.push_back(option.c_str());
    }
    status = value.compile_program(
        program.get(), static_cast<int>(option_pointers.size()),
        option_pointers.data());
    result.log = program_log(value, program.get());
    if (status != HIPRTC_SUCCESS)
        throw std::runtime_error(
            std::string("hiprtcCompileProgram failed: ")
            + value.get_error_string(status)
            + (result.log.empty()
                ? std::string() : std::string("\n") + result.log));

    std::size_t code_size = 0;
    status = value.get_code_size(program.get(), &code_size);
    if (status != HIPRTC_SUCCESS)
        throw hiprtc_error(value, status, "hiprtcGetCodeSize");
    if (code_size == 0)
        throw std::runtime_error("hipRTC produced an empty code object");
    result.code.resize(code_size);
    status = value.get_code(
        program.get(), reinterpret_cast<char*>(result.code.data()));
    if (status != HIPRTC_SUCCESS)
        throw hiprtc_error(value, status, "hiprtcGetCode");
    return result;
}

}  // namespace symmetrix::execution

#else

#include <stdexcept>

namespace symmetrix::execution {

HiprtcInformation hiprtc_information()
{
    return {
        false, {}, "this Symmetrix build does not enable Kokkos HIP", 0, 0};
}

HiprtcCompilation compile_hip_with_hiprtc(
    std::string_view,
    const std::vector<std::string>&)
{
    throw std::runtime_error(
        "hipRTC compilation requires a Kokkos-HIP Symmetrix build");
}

}  // namespace symmetrix::execution

#endif
