#pragma once

// Native Apple Metal runtime layer. The interface is plain C++20 so that
// Kokkos translation units can include it; Objective-C objects live only in
// metal_runtime.mm behind shared implementation handles.
//
// Metal shading language has no double type, so kernels compiled through this
// layer operate in FP32. Buffers use shared storage on unified-memory devices:
// the CPU may read or write contents() only while no submitted batch that
// references the buffer is still executing.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace symmetrix::execution::metal {

struct MetalInformation {
    bool available = false;
    std::string reason;
    std::string device_name;
    std::string gpu_family;  // Highest supported Apple family, e.g. "apple7".
    bool supports_metal3 = false;
    bool unified_memory = false;
    bool low_power = false;
    bool removable = false;
    std::uint64_t registry_id = 0;
    std::uint64_t recommended_working_set_bytes = 0;
    std::uint64_t max_buffer_bytes = 0;
    std::uint32_t max_threads_per_threadgroup = 0;
    std::uint32_t max_threadgroup_memory_bytes = 0;
    std::size_t page_bytes = 0;
};

// Queries the system default device without throwing.
MetalInformation metal_information();

// Floating-point contract for compiled kernels. Safe keeps strict IEEE
// semantics and forbids fused multiply-add contraction; relaxed allows
// contraction and reassociation but keeps NaN and infinity handling; fast
// additionally assumes finite values.
enum class MathMode { safe, relaxed, fast };

struct CompileOptions {
    // Encoded as major*100 + minor; 300 selects MSL 3.0.
    int language_version = 300;
    MathMode math_mode = MathMode::safe;
    std::map<std::string, std::string> macros;
};

struct Size3 {
    std::uint64_t x = 1;
    std::uint64_t y = 1;
    std::uint64_t z = 1;

    std::uint64_t product() const { return x*y*z; }
};

struct BatchTiming {
    // GPU timestamps reported by the command buffer. A positive gpu_seconds is
    // the evidence that the work was scheduled on the GPU.
    double gpu_seconds = 0.0;
    double kernel_seconds = 0.0;
    std::uint64_t dispatch_count = 0;
};

class Device;
class CommandBatch;
class Buffer;

// A row-major FP32 matrix inside a buffer, addressed with an explicit row
// stride so strided sub-blocks of evaluator tensors need no packing.
struct MatrixView {
    const Buffer* buffer = nullptr;
    std::size_t offset_bytes = 0;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::size_t row_bytes = 0;
};

class Buffer {
public:
    Buffer() = default;

    explicit operator bool() const { return static_cast<bool>(impl_); }
    std::size_t size() const;
    void* contents() const;
    bool owns_storage() const;
    // GPU virtual address for device pointers embedded in argument structs.
    // Buffers referenced only through such pointers must be declared to the
    // batch with CommandBatch::use_buffer.
    std::uint64_t gpu_address() const;

    template <class T>
    T* data() const { return static_cast<T*>(contents()); }

private:
    friend class Device;
    friend class CommandBatch;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

class Library {
public:
    Library() = default;

    explicit operator bool() const { return static_cast<bool>(impl_); }
    std::string compile_log() const;
    bool has_function(std::string_view name) const;

private:
    friend class Device;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

class Pipeline {
public:
    Pipeline() = default;

    explicit operator bool() const { return static_cast<bool>(impl_); }
    const std::string& function_name() const;
    std::uint32_t thread_execution_width() const;
    std::uint32_t max_total_threads_per_threadgroup() const;
    std::uint32_t static_threadgroup_memory_bytes() const;

private:
    friend class Device;
    friend class CommandBatch;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// Records compute dispatches into one serial compute encoder. Serial dispatch
// guarantees that each dispatch observes the writes of the previous one.
// Argument bindings persist across dispatches within a batch, as in Metal.
class CommandBatch {
public:
    CommandBatch() = default;
    CommandBatch(CommandBatch&&) noexcept;
    CommandBatch& operator=(CommandBatch&&) noexcept;
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;
    ~CommandBatch();

    explicit operator bool() const { return static_cast<bool>(impl_); }

    CommandBatch& set_buffer(
        std::uint32_t index, const Buffer& buffer, std::size_t offset = 0);
    // Copies at most 4096 bytes of small constant arguments into the batch.
    CommandBatch& set_bytes(
        std::uint32_t index, const void* bytes, std::size_t size);
    template <class T>
    CommandBatch& set_value(std::uint32_t index, const T& value)
    {
        return set_bytes(index, &value, sizeof(T));
    }
    CommandBatch& set_threadgroup_memory(
        std::uint32_t index, std::size_t bytes);
    // Makes a buffer reached through an embedded GPU address resident for
    // the dispatches that follow.
    CommandBatch& use_buffer(const Buffer& buffer, bool written);

    // result = alpha*op(left)*op(right) + beta*result with the runtime's FP32
    // GEMM kernel, encoded in the current compute encoder. It replaces the
    // pipeline and buffer(0) bindings, so set arguments again afterwards.
    CommandBatch& gemm(
        const MatrixView& left, const MatrixView& right, const MatrixView& result,
        bool transpose_left = false, bool transpose_right = false,
        double alpha = 1.0, double beta = 0.0);

    // Grid in threads; edge threadgroups may be partial.
    CommandBatch& dispatch_threads(
        const Pipeline& pipeline, Size3 threads, Size3 threadgroup);
    CommandBatch& dispatch_threadgroups(
        const Pipeline& pipeline, Size3 threadgroups, Size3 threadgroup);

    // Ends encoding and submits without waiting.
    void commit();
    // Waits for a committed batch and rethrows GPU execution errors.
    BatchTiming wait();
    BatchTiming submit_and_wait()
    {
        commit();
        return wait();
    }

private:
    friend class Device;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Device {
public:
    Device() = default;

    // Throws when no Metal device is present.
    static Device system_default();

    explicit operator bool() const { return static_cast<bool>(impl_); }
    const MetalInformation& information() const;

    // Zero-initialized shared-storage allocation visible to CPU and GPU.
    Buffer allocate(std::size_t bytes) const;
    // Wraps caller-owned memory without copying. Metal requires the pointer
    // and the length to be multiples of information().page_bytes; the memory
    // must outlive every batch that references the returned buffer.
    Buffer wrap(void* host, std::size_t bytes) const;

    Library compile(
        std::string_view source, const CompileOptions& options = {}) const;
    Library load_library(std::span<const std::uint8_t> metallib) const;
    Pipeline pipeline(
        const Library& library, std::string_view function_name) const;

    CommandBatch begin() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace symmetrix::execution::metal
