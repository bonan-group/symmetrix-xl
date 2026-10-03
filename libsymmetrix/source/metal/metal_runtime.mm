#include "metal_runtime.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <unistd.h>

#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#if !__has_feature(objc_arc)
#error "metal_runtime.mm must be compiled with -fobjc-arc"
#endif

namespace symmetrix::execution::metal {

namespace {

// Metal's argument tables hold 31 buffer slots; setBytes is limited to 4 KiB.
constexpr std::uint32_t argument_table_entries = 31;
constexpr std::size_t inline_bytes_limit = 4096;

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error("Metal: "+message);
}

std::string describe(NSError* error)
{
    if (error == nil)
        return "unknown error";
    const char* text = error.localizedDescription.UTF8String;
    return text != nullptr ? text : "unknown error";
}

NSString* to_ns(std::string_view text)
{
    return [[NSString alloc] initWithBytes:text.data()
                                    length:text.size()
                                  encoding:NSUTF8StringEncoding];
}

std::string highest_apple_family(id<MTLDevice> device)
{
    // MTLGPUFamilyApple<N> is encoded as 1000 + N. Probing beyond the SDK's
    // newest named family is harmless: unknown families report unsupported.
    for (int family = 1016; family >= 1001; --family) {
        if ([device supportsFamily:static_cast<MTLGPUFamily>(family)])
            return "apple"+std::to_string(family-1000);
    }
    return "";
}

MetalInformation describe_device(id<MTLDevice> device)
{
    MetalInformation info;
    info.available = true;
    info.device_name = device.name.UTF8String;
    info.gpu_family = highest_apple_family(device);
    info.supports_metal3 = [device supportsFamily:MTLGPUFamilyMetal3];
    info.unified_memory = device.hasUnifiedMemory;
    info.low_power = device.isLowPower;
    info.removable = device.isRemovable;
    info.registry_id = device.registryID;
    info.recommended_working_set_bytes =
        device.recommendedMaxWorkingSetSize;
    info.max_buffer_bytes = device.maxBufferLength;
    const MTLSize threads = device.maxThreadsPerThreadgroup;
    info.max_threads_per_threadgroup =
        static_cast<std::uint32_t>(threads.width);
    info.max_threadgroup_memory_bytes =
        static_cast<std::uint32_t>(device.maxThreadgroupMemoryLength);
    info.page_bytes = static_cast<std::size_t>(getpagesize());
    return info;
}

MTLSize to_mtl(const Size3& size)
{
    return MTLSizeMake(size.x, size.y, size.z);
}

}  // namespace

struct Buffer::Impl {
    id<MTLBuffer> buffer = nil;
    bool owns_storage = true;
};

struct Library::Impl {
    id<MTLLibrary> library = nil;
    std::string log;
};

struct Pipeline::Impl {
    id<MTLComputePipelineState> state = nil;
    std::string function_name;
};

struct CommandBatch::Impl {
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    std::uint32_t max_threadgroup_memory_bytes = 0;
    std::array<std::size_t, argument_table_entries> threadgroup_memory{};
    std::uint64_t dispatch_count = 0;
    bool committed = false;
    bool waited = false;
    // Residency declared with use_buffer; replayed on encoders opened after
    // the encoder is closed so later dispatches still see indirectly used buffers.
    std::vector<std::pair<id<MTLBuffer>, MTLResourceUsage>> resident;

    // Opens a compute encoder on demand, e.g. after a GEMM closed one.
    void require_encoding()
    {
        if (committed)
            fail("command batch is no longer encoding");
        if (encoder != nil)
            return;
        encoder = [command_buffer
            computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if (encoder == nil)
            fail("could not reopen a compute command encoder");
        for (const auto& [buffer, usage] : resident)
            [encoder useResource:buffer usage:usage];
        for (std::uint32_t index = 0; index < argument_table_entries; ++index)
            if (threadgroup_memory[index] != 0)
                [encoder setThreadgroupMemoryLength:threadgroup_memory[index]
                                            atIndex:index];
    }

    std::size_t dynamic_threadgroup_bytes() const
    {
        std::size_t bytes = 0;
        for (const std::size_t entry : threadgroup_memory)
            bytes += entry;
        return bytes;
    }

    void end_encoding()
    {
        if (encoder != nil) {
            [encoder endEncoding];
            encoder = nil;
        }
    }
};

struct Device::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    MetalInformation information;
};

MetalInformation metal_information()
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            MetalInformation info;
            info.reason = "MTLCreateSystemDefaultDevice returned no device";
            return info;
        }
        return describe_device(device);
    }
}

// ----- Buffer -----

std::size_t Buffer::size() const
{
    return impl_ ? static_cast<std::size_t>(impl_->buffer.length) : 0;
}

void* Buffer::contents() const
{
    return impl_ ? impl_->buffer.contents : nullptr;
}

bool Buffer::owns_storage() const
{
    return impl_ && impl_->owns_storage;
}

std::uint64_t Buffer::gpu_address() const
{
    if (!impl_)
        return 0;
    if (@available(macOS 13.0, *))
        return impl_->buffer.gpuAddress;
    fail("embedded GPU addresses require macOS 13 or newer");
}

// ----- Library -----

std::string Library::compile_log() const
{
    return impl_ ? impl_->log : std::string();
}

bool Library::has_function(const std::string_view name) const
{
    if (!impl_)
        return false;
    @autoreleasepool {
        return [impl_->library.functionNames containsObject:to_ns(name)];
    }
}

// ----- Pipeline -----

const std::string& Pipeline::function_name() const
{
    static const std::string empty;
    return impl_ ? impl_->function_name : empty;
}

std::uint32_t Pipeline::thread_execution_width() const
{
    return impl_
        ? static_cast<std::uint32_t>(impl_->state.threadExecutionWidth) : 0;
}

std::uint32_t Pipeline::max_total_threads_per_threadgroup() const
{
    return impl_
        ? static_cast<std::uint32_t>(
            impl_->state.maxTotalThreadsPerThreadgroup)
        : 0;
}

std::uint32_t Pipeline::static_threadgroup_memory_bytes() const
{
    return impl_
        ? static_cast<std::uint32_t>(impl_->state.staticThreadgroupMemoryLength)
        : 0;
}

// ----- CommandBatch -----

CommandBatch::CommandBatch(CommandBatch&&) noexcept = default;

namespace {

// Wrapped host memory may be released by the caller right after a batch goes
// out of scope or is replaced, so an in-flight batch is drained first.
template <class Impl>
void drain(Impl& impl)
{
    if (!impl)
        return;
    @autoreleasepool {
        impl->end_encoding();
        if (impl->committed && !impl->waited)
            [impl->command_buffer waitUntilCompleted];
    }
}

}  // namespace

CommandBatch& CommandBatch::operator=(CommandBatch&& other) noexcept
{
    if (this != &other) {
        drain(impl_);
        impl_ = std::move(other.impl_);
    }
    return *this;
}

CommandBatch::~CommandBatch()
{
    drain(impl_);
}

CommandBatch& CommandBatch::set_buffer(
    const std::uint32_t index, const Buffer& buffer, const std::size_t offset)
{
    if (!impl_)
        fail("set_buffer on an empty command batch");
    impl_->require_encoding();
    if (index >= argument_table_entries)
        fail("buffer index "+std::to_string(index)+" exceeds the argument table");
    if (!buffer)
        fail("set_buffer received an empty buffer");
    if (offset > buffer.size())
        fail("buffer offset exceeds the buffer length");
    [impl_->encoder setBuffer:buffer.impl_->buffer offset:offset atIndex:index];
    return *this;
}

CommandBatch& CommandBatch::set_bytes(
    const std::uint32_t index, const void* bytes, const std::size_t size)
{
    if (!impl_)
        fail("set_bytes on an empty command batch");
    impl_->require_encoding();
    if (index >= argument_table_entries)
        fail("bytes index "+std::to_string(index)+" exceeds the argument table");
    if (size == 0 || size > inline_bytes_limit)
        fail("inline argument size must be in [1, 4096] bytes");
    [impl_->encoder setBytes:bytes length:size atIndex:index];
    return *this;
}

CommandBatch& CommandBatch::set_threadgroup_memory(
    const std::uint32_t index, const std::size_t bytes)
{
    if (!impl_)
        fail("set_threadgroup_memory on an empty command batch");
    impl_->require_encoding();
    if (index >= argument_table_entries)
        fail("threadgroup memory index exceeds the argument table");
    // Metal requires threadgroup allocations in 16-byte multiples.
    const std::size_t rounded = (bytes+15) & ~static_cast<std::size_t>(15);
    impl_->threadgroup_memory[index] = rounded;
    [impl_->encoder setThreadgroupMemoryLength:rounded atIndex:index];
    return *this;
}

CommandBatch& CommandBatch::use_buffer(const Buffer& buffer, const bool written)
{
    if (!impl_)
        fail("use_buffer on an empty command batch");
    impl_->require_encoding();
    if (!buffer)
        fail("use_buffer received an empty buffer");
    const MTLResourceUsage usage = written
        ? (MTLResourceUsageRead | MTLResourceUsageWrite)
        : MTLResourceUsageRead;
    [impl_->encoder useResource:buffer.impl_->buffer usage:usage];
    impl_->resident.emplace_back(buffer.impl_->buffer, usage);
    return *this;
}

namespace {

void validate_matrix(const MatrixView& view, const char* role)
{
    if (view.buffer == nullptr || !*view.buffer)
        fail(std::string("GEMM ")+role+" matrix has no buffer");
    if (view.rows == 0 || view.columns == 0)
        fail(std::string("GEMM ")+role+" matrix is empty");
    if (view.row_bytes % sizeof(float) != 0 || view.offset_bytes % sizeof(float) != 0
            || view.row_bytes < view.columns*sizeof(float))
        fail(std::string("GEMM ")+role+" matrix stride or offset is invalid");
    const std::size_t extent = view.offset_bytes
        +(static_cast<std::size_t>(view.rows)-1)*view.row_bytes
        +view.columns*sizeof(float);
    if (extent > view.buffer->size())
        fail(std::string("GEMM ")+role+" matrix exceeds its buffer");
}

}  // namespace

namespace {

// Native FP32 GEMM. MPSMatrixMultiplication was found to skip whole output
// tiles without reporting an error for large strided matrices, so the
// runtime encodes its own kernel. A 128-thread threadgroup owns a 32 x 32
// output tile and walks the interior in chunks of 32: the four SIMD groups
// stage zero-padded A and B chunks in threadgroup memory, then each
// accumulates a 16 x 16 quadrant with 8 x 8 simdgroup matrices. Padding lets
// partial edge tiles take the same path; their stores are guarded.
constexpr const char* gemm_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct GemmArgs {
    device const float* left;
    device const float* right;
    device float* result;
    uint rows;
    uint columns;
    uint interior;
    uint left_pitch;
    uint right_pitch;
    uint result_pitch;
    uint transpose_left;
    uint transpose_right;
    float alpha;
    float beta;
};

constant constexpr uint tile = 32;

kernel void symmetrix_gemm(
    constant GemmArgs& a [[buffer(0)]],
    uint2 position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float left_tile[tile * tile];
    threadgroup float right_tile[tile * tile];
    threadgroup float out_tile[tile * tile];
    const uint row0 = position.y * tile;
    const uint column0 = position.x * tile;
    const uint quadrant_row = (group / 2u) * 16u;
    const uint quadrant_column = (group % 2u) * 16u;
    simdgroup_float8x8 accumulator[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j)
            accumulator[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const bool aligned = a.transpose_left == 0 && a.transpose_right == 0
        && a.left_pitch % 4u == 0u && a.right_pitch % 4u == 0u
        && ((ulong(a.left) | ulong(a.right)) & 15ul) == 0ul;
    for (uint k0 = 0; k0 < a.interior; k0 += tile) {
        const bool interior_chunk = aligned && row0 + tile <= a.rows
            && column0 + tile <= a.columns && k0 + tile <= a.interior;
        if (interior_chunk) {
            // Two float4 of A and two of B per thread: row r, columns 4q..4q+3.
            for (uint vector = thread_index; vector < tile * tile / 4u; vector += 128u) {
                const uint r = vector / (tile / 4u);
                const uint q = vector % (tile / 4u);
                const float4 x = *reinterpret_cast<device const float4*>(
                    a.left + ulong(row0 + r) * a.left_pitch + k0 + 4u * q);
                const float4 y = *reinterpret_cast<device const float4*>(
                    a.right + ulong(k0 + r) * a.right_pitch + column0 + 4u * q);
                *reinterpret_cast<threadgroup float4*>(left_tile + r * tile + 4u * q) = x;
                *reinterpret_cast<threadgroup float4*>(right_tile + r * tile + 4u * q) = y;
            }
        } else
        // Guarded scalar staging with zero padding at the matrix edges.
        for (uint element = thread_index; element < tile * tile; element += 128u) {
            const uint r = element / tile;
            const uint c = element % tile;
            const uint i = row0 + r, k = k0 + c;
            float value = 0.0f;
            if (i < a.rows && k < a.interior)
                value = a.transpose_left != 0 ? a.left[ulong(k) * a.left_pitch + i]
                                              : a.left[ulong(i) * a.left_pitch + k];
            left_tile[element] = value;
            const uint kk = k0 + r, j = column0 + c;
            float other = 0.0f;
            if (kk < a.interior && j < a.columns)
                other = a.transpose_right != 0 ? a.right[ulong(j) * a.right_pitch + kk]
                                               : a.right[ulong(kk) * a.right_pitch + j];
            right_tile[element] = other;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < tile; k += 8u) {
            simdgroup_float8x8 left[2];
            simdgroup_float8x8 right[2];
            for (uint i = 0; i < 2; ++i)
                simdgroup_load(left[i], left_tile, tile,
                    ulong2(k, quadrant_row + 8u * i));
            for (uint j = 0; j < 2; ++j)
                simdgroup_load(right[j], right_tile, tile,
                    ulong2(quadrant_column + 8u * j, k));
            for (uint i = 0; i < 2; ++i)
                for (uint j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(
                        accumulator[i][j], left[i], right[j], accumulator[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j)
            simdgroup_store(accumulator[i][j], out_tile, tile,
                ulong2(quadrant_column + 8u * j, quadrant_row + 8u * i));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint element = thread_index; element < tile * tile; element += 128u) {
        const uint i = row0 + element / tile;
        const uint j = column0 + element % tile;
        if (i >= a.rows || j >= a.columns)
            continue;
        device float* out = a.result + ulong(i) * a.result_pitch + j;
        const float sum = out_tile[element];
        *out = a.beta != 0.0f ? a.alpha * sum + a.beta * *out : a.alpha * sum;
    }
}
)MSL";

struct GemmArgs {
    std::uint64_t left;
    std::uint64_t right;
    std::uint64_t result;
    std::uint32_t rows;
    std::uint32_t columns;
    std::uint32_t interior;
    std::uint32_t left_pitch;
    std::uint32_t right_pitch;
    std::uint32_t result_pitch;
    std::uint32_t transpose_left;
    std::uint32_t transpose_right;
    float alpha;
    float beta;
};
static_assert(sizeof(GemmArgs) == 64);

// One compiled GEMM pipeline per device for the process lifetime.
id<MTLComputePipelineState> gemm_pipeline(id<MTLDevice> device)
{
    static std::mutex mutex;
    static std::map<void*, id<MTLComputePipelineState>> pipelines;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = pipelines.find((__bridge void*)device);
    if (found != pipelines.end())
        return found->second;
    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_0;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:gemm_source]
                     options:options
                       error:&error];
    if (library == nil)
        fail("could not compile the GEMM kernel: "+describe(error));
    id<MTLFunction> function = [library newFunctionWithName:@"symmetrix_gemm"];
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (state == nil)
        fail("could not create the GEMM pipeline: "+describe(error));
    pipelines.emplace((__bridge void*)device, state);
    return state;
}

}  // namespace

CommandBatch& CommandBatch::gemm(
    const MatrixView& left, const MatrixView& right, const MatrixView& result,
    const bool transpose_left, const bool transpose_right,
    const double alpha, const double beta)
{
    if (!impl_ || impl_->committed)
        fail("gemm requires an encoding command batch");
    validate_matrix(left, "left");
    validate_matrix(right, "right");
    validate_matrix(result, "result");
    const std::uint32_t rows = transpose_left ? left.columns : left.rows;
    const std::uint32_t interior = transpose_left ? left.rows : left.columns;
    const std::uint32_t right_interior = transpose_right ? right.columns : right.rows;
    const std::uint32_t columns = transpose_right ? right.rows : right.columns;
    if (interior != right_interior || result.rows != rows || result.columns != columns)
        fail("GEMM matrix shapes do not conform");
    for (const MatrixView* view : {&left, &right, &result})
        if (view->row_bytes % sizeof(float) != 0 || view->offset_bytes % sizeof(float) != 0)
            fail("GEMM rows and offsets must be whole FP32 elements");
    impl_->require_encoding();
    @autoreleasepool {
        id<MTLDevice> device = impl_->command_buffer.device;
        const GemmArgs args{
            left.buffer->impl_->buffer.gpuAddress+left.offset_bytes,
            right.buffer->impl_->buffer.gpuAddress+right.offset_bytes,
            result.buffer->impl_->buffer.gpuAddress+result.offset_bytes,
            rows, columns, interior,
            static_cast<std::uint32_t>(left.row_bytes/sizeof(float)),
            static_cast<std::uint32_t>(right.row_bytes/sizeof(float)),
            static_cast<std::uint32_t>(result.row_bytes/sizeof(float)),
            transpose_left ? 1u : 0u, transpose_right ? 1u : 0u,
            static_cast<float>(alpha), static_cast<float>(beta)};
        [impl_->encoder useResource:left.buffer->impl_->buffer usage:MTLResourceUsageRead];
        [impl_->encoder useResource:right.buffer->impl_->buffer usage:MTLResourceUsageRead];
        [impl_->encoder useResource:result.buffer->impl_->buffer
                              usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [impl_->encoder setComputePipelineState:gemm_pipeline(device)];
        [impl_->encoder setBytes:&args length:sizeof(args) atIndex:0];
        [impl_->encoder dispatchThreadgroups:MTLSizeMake((columns+31)/32, (rows+31)/32, 1)
                       threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    ++impl_->dispatch_count;
    return *this;
}

namespace {

void validate_dispatch(
    id<MTLComputePipelineState> state, const std::string& function_name,
    const std::size_t dynamic_threadgroup_bytes,
    const std::uint32_t max_threadgroup_memory_bytes,
    const Size3& grid, const Size3& threadgroup)
{
    if (grid.x == 0 || grid.y == 0 || grid.z == 0)
        fail("dispatch grid must be nonempty in every dimension");
    if (threadgroup.x == 0 || threadgroup.y == 0 || threadgroup.z == 0)
        fail("threadgroup must be nonempty in every dimension");
    const std::uint64_t limit = state.maxTotalThreadsPerThreadgroup;
    if (threadgroup.product() > limit)
        fail("threadgroup of "+std::to_string(threadgroup.product())
            +" threads exceeds the pipeline limit "+std::to_string(limit)
            +" for '"+function_name+"'");
    const std::size_t total =
        dynamic_threadgroup_bytes+state.staticThreadgroupMemoryLength;
    if (total > max_threadgroup_memory_bytes)
        fail("threadgroup memory "+std::to_string(total)
            +" bytes exceeds the device limit "
            +std::to_string(max_threadgroup_memory_bytes));
}

}  // namespace

CommandBatch& CommandBatch::dispatch_threads(
    const Pipeline& pipeline, const Size3 threads, const Size3 threadgroup)
{
    if (!impl_ || !pipeline)
        fail("dispatch_threads requires a batch and a pipeline");
    impl_->require_encoding();
    validate_dispatch(
        pipeline.impl_->state, pipeline.impl_->function_name,
        impl_->dynamic_threadgroup_bytes(),
        impl_->max_threadgroup_memory_bytes, threads, threadgroup);
    [impl_->encoder setComputePipelineState:pipeline.impl_->state];
    [impl_->encoder dispatchThreads:to_mtl(threads)
              threadsPerThreadgroup:to_mtl(threadgroup)];
    ++impl_->dispatch_count;
    return *this;
}

CommandBatch& CommandBatch::dispatch_threadgroups(
    const Pipeline& pipeline, const Size3 threadgroups, const Size3 threadgroup)
{
    if (!impl_ || !pipeline)
        fail("dispatch_threadgroups requires a batch and a pipeline");
    impl_->require_encoding();
    validate_dispatch(
        pipeline.impl_->state, pipeline.impl_->function_name,
        impl_->dynamic_threadgroup_bytes(),
        impl_->max_threadgroup_memory_bytes, threadgroups, threadgroup);
    [impl_->encoder setComputePipelineState:pipeline.impl_->state];
    [impl_->encoder dispatchThreadgroups:to_mtl(threadgroups)
                   threadsPerThreadgroup:to_mtl(threadgroup)];
    ++impl_->dispatch_count;
    return *this;
}

void CommandBatch::commit()
{
    if (!impl_)
        fail("commit on an empty command batch");
    if (impl_->committed)
        fail("command batch was already committed");
    @autoreleasepool {
        impl_->end_encoding();
        [impl_->command_buffer commit];
        impl_->committed = true;
    }
}

BatchTiming CommandBatch::wait()
{
    if (!impl_ || !impl_->committed)
        fail("wait requires a committed command batch");
    std::string error;
    BatchTiming timing;
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = impl_->command_buffer;
        [command_buffer waitUntilCompleted];
        impl_->waited = true;
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            error = "command buffer failed: "+describe(command_buffer.error);
        } else {
            timing.gpu_seconds =
                command_buffer.GPUEndTime-command_buffer.GPUStartTime;
            timing.kernel_seconds =
                command_buffer.kernelEndTime-command_buffer.kernelStartTime;
            timing.dispatch_count = impl_->dispatch_count;
        }
    }
    if (!error.empty())
        fail(error);
    return timing;
}

// ----- Device -----

Device Device::system_default()
{
    std::string error;
    Device result;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            error = "no Metal device is available";
        } else {
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (queue == nil) {
                error = "could not create a command queue";
            } else {
                auto impl = std::make_shared<Impl>();
                impl->device = device;
                impl->queue = queue;
                impl->information = describe_device(device);
                result.impl_ = std::move(impl);
            }
        }
    }
    if (!error.empty())
        fail(error);
    return result;
}

const MetalInformation& Device::information() const
{
    if (!impl_)
        fail("information on an empty device");
    return impl_->information;
}

Buffer Device::allocate(const std::size_t bytes) const
{
    if (!impl_)
        fail("allocate on an empty device");
    if (bytes == 0)
        fail("cannot allocate an empty buffer");
    if (bytes > impl_->information.max_buffer_bytes)
        fail("allocation of "+std::to_string(bytes)
            +" bytes exceeds the device buffer limit");
    Buffer buffer;
    @autoreleasepool {
        id<MTLBuffer> handle =
            [impl_->device newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
        if (handle != nil) {
            auto impl = std::make_shared<Buffer::Impl>();
            impl->buffer = handle;
            buffer.impl_ = std::move(impl);
        }
    }
    if (!buffer)
        fail("could not allocate "+std::to_string(bytes)+" bytes");
    return buffer;
}

Buffer Device::wrap(void* host, const std::size_t bytes) const
{
    if (!impl_)
        fail("wrap on an empty device");
    const std::size_t page = impl_->information.page_bytes;
    if (host == nullptr || bytes == 0)
        fail("cannot wrap an empty host region");
    if (reinterpret_cast<std::uintptr_t>(host) % page != 0 || bytes % page != 0)
        fail("wrapped host memory must be aligned to and sized in multiples "
            "of the "+std::to_string(page)+"-byte page size");
    Buffer buffer;
    @autoreleasepool {
        id<MTLBuffer> handle =
            [impl_->device newBufferWithBytesNoCopy:host
                                             length:bytes
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
        if (handle != nil) {
            auto impl = std::make_shared<Buffer::Impl>();
            impl->buffer = handle;
            impl->owns_storage = false;
            buffer.impl_ = std::move(impl);
        }
    }
    if (!buffer)
        fail("could not wrap host memory of "+std::to_string(bytes)+" bytes");
    return buffer;
}

Library Device::compile(
    const std::string_view source, const CompileOptions& options) const
{
    if (!impl_)
        fail("compile on an empty device");
    std::string error;
    Library library;
    @autoreleasepool {
        MTLCompileOptions* settings = [[MTLCompileOptions alloc] init];
        const unsigned major =
            static_cast<unsigned>(options.language_version/100);
        const unsigned minor =
            static_cast<unsigned>(options.language_version%100);
        settings.languageVersion =
            static_cast<MTLLanguageVersion>((major << 16) | minor);
        if (@available(macOS 15.0, *)) {
            switch (options.math_mode) {
            case MathMode::safe: settings.mathMode = MTLMathModeSafe; break;
            case MathMode::relaxed: settings.mathMode = MTLMathModeRelaxed; break;
            case MathMode::fast: settings.mathMode = MTLMathModeFast; break;
            }
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            settings.fastMathEnabled =
                options.math_mode == MathMode::safe ? NO : YES;
#pragma clang diagnostic pop
        }
        if (!options.macros.empty()) {
            NSMutableDictionary<NSString*, NSObject*>* macros =
                [NSMutableDictionary dictionary];
            for (const auto& [name, value] : options.macros)
                macros[to_ns(name)] = to_ns(value);
            settings.preprocessorMacros = macros;
        }
        NSError* compile_error = nil;
        id<MTLLibrary> handle =
            [impl_->device newLibraryWithSource:to_ns(source)
                                        options:settings
                                          error:&compile_error];
        if (handle == nil) {
            error = "MSL compilation failed: "+describe(compile_error);
        } else {
            auto impl = std::make_shared<Library::Impl>();
            impl->library = handle;
            // A non-nil error alongside a library carries compiler warnings.
            if (compile_error != nil)
                impl->log = describe(compile_error);
            library.impl_ = std::move(impl);
        }
    }
    if (!error.empty())
        fail(error);
    return library;
}

Library Device::load_library(const std::span<const std::uint8_t> metallib) const
{
    if (!impl_)
        fail("load_library on an empty device");
    if (metallib.empty())
        fail("metallib data is empty");
    std::string error;
    Library library;
    @autoreleasepool {
        dispatch_data_t data = dispatch_data_create(
            metallib.data(), metallib.size(), nullptr,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError* load_error = nil;
        id<MTLLibrary> handle =
            [impl_->device newLibraryWithData:data error:&load_error];
        if (handle == nil) {
            error = "could not load metallib: "+describe(load_error);
        } else {
            auto impl = std::make_shared<Library::Impl>();
            impl->library = handle;
            library.impl_ = std::move(impl);
        }
    }
    if (!error.empty())
        fail(error);
    return library;
}

Pipeline Device::pipeline(
    const Library& library, const std::string_view function_name) const
{
    if (!impl_ || !library)
        fail("pipeline requires a device and a library");
    std::string error;
    Pipeline pipeline;
    @autoreleasepool {
        id<MTLFunction> function =
            [library.impl_->library newFunctionWithName:to_ns(function_name)];
        if (function == nil) {
            error = "library has no function '"+std::string(function_name)+"'";
        } else {
            NSError* state_error = nil;
            id<MTLComputePipelineState> state =
                [impl_->device newComputePipelineStateWithFunction:function
                                                             error:&state_error];
            if (state == nil) {
                error = "could not create pipeline '"+std::string(function_name)
                    +"': "+describe(state_error);
            } else {
                auto impl = std::make_shared<Pipeline::Impl>();
                impl->state = state;
                impl->function_name = std::string(function_name);
                pipeline.impl_ = std::move(impl);
            }
        }
    }
    if (!error.empty())
        fail(error);
    return pipeline;
}

CommandBatch Device::begin() const
{
    if (!impl_)
        fail("begin on an empty device");
    std::string error;
    CommandBatch batch;
    @autoreleasepool {
        MTLCommandBufferDescriptor* descriptor =
            [[MTLCommandBufferDescriptor alloc] init];
        descriptor.errorOptions =
            MTLCommandBufferErrorOptionEncoderExecutionStatus;
        id<MTLCommandBuffer> command_buffer =
            [impl_->queue commandBufferWithDescriptor:descriptor];
        id<MTLComputeCommandEncoder> encoder = command_buffer == nil
            ? nil
            : [command_buffer
                computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if (encoder == nil) {
            error = "could not create a compute command encoder";
        } else {
            auto impl = std::make_unique<CommandBatch::Impl>();
            impl->command_buffer = command_buffer;
            impl->encoder = encoder;
            impl->max_threadgroup_memory_bytes =
                impl_->information.max_threadgroup_memory_bytes;
            batch.impl_ = std::move(impl);
        }
    }
    if (!error.empty())
        fail(error);
    return batch;
}

}  // namespace symmetrix::execution::metal
