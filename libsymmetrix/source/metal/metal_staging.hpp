#pragma once

// Shared-buffer staging for Metal modules that read Kokkos host views.
// Slots grow on demand and are reused across launches; read-only tables that
// live for the model lifetime are uploaded once and keyed by host pointer.
// Host ranges registered in a HostMemoryMap are bound in place instead.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "metal_runtime.hpp"

namespace symmetrix::execution::metal {

// Compile policy shared by the Symmetrix Metal modules. Relaxed math permits
// fused multiply-add contraction, which the edge kernels depend on for speed,
// while keeping the NaN and infinity semantics used by model validation.
inline CompileOptions module_compile_options()
{
    CompileOptions options;
    options.math_mode = MathMode::relaxed;
    return options;
}

// A device address inside a buffer, for packets that embed GPU pointers.
struct DeviceSpan {
    const Buffer* buffer = nullptr;
    std::size_t offset = 0;

    std::uint64_t address() const { return buffer->gpu_address()+offset; }
};

// Host memory that Metal kernels may access in place. Apple GPUs share the
// host's memory, so a page-rounded no-copy wrap of a live host allocation
// lets kernels read and write it without staging copies. The owner must keep
// each mapped allocation alive until it is unmapped.
class HostMemoryMap {
public:
    explicit HostMemoryMap(Device device) : device_(std::move(device)) {}

    // Idempotent for a range that is already mapped.
    void map(const void* host, const std::size_t bytes)
    {
        const auto begin = reinterpret_cast<std::uintptr_t>(host);
        if (host == nullptr || bytes == 0 || find(host, bytes))
            return;
        const std::uintptr_t page = device_.information().page_bytes;
        const std::uintptr_t first = begin/page*page;
        const std::uintptr_t last = (begin+bytes+page-1)/page*page;
        Entry entry{begin, begin+bytes, first,
            device_.wrap(reinterpret_cast<void*>(first), last-first)};
        entries_.push_back(std::make_unique<Entry>(std::move(entry)));
    }

    void unmap(const void* host)
    {
        const auto begin = reinterpret_cast<std::uintptr_t>(host);
        std::erase_if(entries_, [&](const std::unique_ptr<Entry>& entry) {
            return entry->begin == begin;
        });
    }

    std::optional<DeviceSpan> find(const void* host, const std::size_t bytes) const
    {
        const auto begin = reinterpret_cast<std::uintptr_t>(host);
        for (const auto& entry : entries_)
            if (begin >= entry->begin && begin+bytes <= entry->end)
                return DeviceSpan{&entry->buffer, begin-entry->page_begin};
        return std::nullopt;
    }

private:
    struct Entry {
        std::uintptr_t begin = 0;
        std::uintptr_t end = 0;
        std::uintptr_t page_begin = 0;
        Buffer buffer;
    };

    Device device_;
    // Stable addresses: spans point at entry buffers.
    std::vector<std::unique_ptr<Entry>> entries_;
};

class MetalStaging {
public:
    MetalStaging(Device device, std::string owner)
        : device_(std::move(device)), owner_(std::move(owner))
    {}

    void set_host_memory(std::shared_ptr<const HostMemoryMap> host_memory)
    {
        host_memory_ = std::move(host_memory);
    }

    // Starts one launch: forgets the previous launch's inputs and copy-backs.
    void begin()
    {
        inputs_.clear();
        copy_backs_.clear();
    }

    // Read-only kernel input: the mapped host range itself, or a staged copy.
    template <class T>
    DeviceSpan input(const std::string& name, const T* values, const std::size_t count)
    {
        const std::size_t bytes = count*sizeof(T);
        if (const auto span = mapped(values, bytes)) {
            inputs_.push_back({reinterpret_cast<std::uintptr_t>(values), bytes});
            return *span;
        }
        return {&upload(name, values, count), 0};
    }

    // Kernel output: written in place when the host range is mapped and does
    // not overlap an input of this launch, otherwise copied back by
    // complete(). With initialize, the kernel also reads the current values.
    template <class T>
    DeviceSpan output(const std::string& name, T* values, const std::size_t count,
        const bool initialize = false)
    {
        const std::size_t bytes = count*sizeof(T);
        const auto begin = reinterpret_cast<std::uintptr_t>(values);
        const bool overlaps = std::any_of(inputs_.begin(), inputs_.end(),
            [&](const std::pair<std::uintptr_t, std::size_t>& input) {
                return begin < input.first+input.second && input.first < begin+bytes;
            });
        if (!overlaps)
            if (const auto span = mapped(values, bytes))
                return *span;
        Buffer& buffer = initialize ? upload(name, values, count) : slot(name, bytes);
        copy_backs_.push_back({&buffer, values, bytes});
        return {&buffer, 0};
    }

    bool is_mapped(const void* host, const std::size_t bytes) const
    {
        return mapped(host, bytes).has_value();
    }

    // Copies staged outputs to their host ranges after the launch completes.
    void complete()
    {
        for (const CopyBack& copy : copy_backs_)
            std::memcpy(copy.host, copy.buffer->contents(), copy.bytes);
        copy_backs_.clear();
    }

    Buffer& slot(const std::string& name, const std::size_t bytes)
    {
        Buffer& buffer = slots_[name];
        const std::size_t required = std::max<std::size_t>(bytes, 16);
        if (!buffer || buffer.size() < required)
            buffer = device_.allocate(required);
        return buffer;
    }

    template <class T>
    Buffer& upload(const std::string& name, const T* values, const std::size_t count)
    {
        Buffer& buffer = slot(name, count*sizeof(T));
        if (count != 0) {
            if (values == nullptr)
                fail(name+" is null");
            std::memcpy(buffer.contents(), values, count*sizeof(T));
        }
        return buffer;
    }

    // Metal has no FP64; double views are narrowed at the packet boundary.
    Buffer& upload_narrowed(
        const std::string& name, const double* values, const std::size_t count)
    {
        Buffer& buffer = slot(name, count*sizeof(float));
        if (count != 0 && values == nullptr)
            fail(name+" is null");
        float* out = buffer.data<float>();
        for (std::size_t i = 0; i < count; ++i)
            out[i] = static_cast<float>(values[i]);
        return buffer;
    }

    Buffer& zeroed(const std::string& name, const std::size_t count)
    {
        Buffer& buffer = slot(name, count*sizeof(float));
        std::memset(buffer.contents(), 0, buffer.size());
        return buffer;
    }

    // Drops every cached table. The owner calls this whenever host tables are
    // rebuilt, since a new allocation can reuse a freed table's address.
    void release_tables() { tables_.clear(); }

    // For tables whose host storage is unchanged until release_tables().
    const Buffer& persistent(const void* host, const std::size_t bytes)
    {
        auto cached = std::find_if(tables_.begin(), tables_.end(),
            [&](const Table& entry) {
                return !entry.interleaved && entry.host == host && entry.bytes == bytes;
            });
        if (cached != tables_.end())
            return cached->buffer;
        if (host == nullptr || bytes == 0)
            fail("persistent table is empty");
        Table entry{host, bytes, device_.allocate(bytes)};
        std::memcpy(entry.buffer.contents(), host, bytes);
        tables_.push_back(std::move(entry));
        return tables_.back().buffer;
    }

    // Cubic spline coefficients stored [block, 4, functions] on the host are
    // interleaved to [block, functions, 4] once, so a kernel fetches the four
    // coefficients of one function with a single float4 load.
    const Buffer& persistent_spline4(
        const float* host, const std::size_t blocks, const std::size_t functions)
    {
        const std::size_t bytes = blocks*4*functions*sizeof(float);
        auto cached = std::find_if(tables_.begin(), tables_.end(),
            [&](const Table& entry) {
                return entry.interleaved && entry.host == host && entry.bytes == bytes;
            });
        if (cached != tables_.end())
            return cached->buffer;
        if (host == nullptr || bytes == 0)
            fail("spline coefficient table is empty");
        Table entry{host, bytes, device_.allocate(bytes), true};
        float* out = entry.buffer.data<float>();
        for (std::size_t block = 0; block < blocks; ++block)
            for (std::size_t k = 0; k < 4; ++k)
                for (std::size_t function = 0; function < functions; ++function)
                    out[(block*functions+function)*4+k] =
                        host[(block*4+k)*functions+function];
        tables_.push_back(std::move(entry));
        return tables_.back().buffer;
    }

    [[noreturn]] void fail(const std::string& message) const
    {
        throw std::runtime_error("Metal "+owner_+": "+message);
    }

private:
    struct Table {
        const void* host = nullptr;
        std::size_t bytes = 0;
        Buffer buffer;
        bool interleaved = false;
    };

    struct CopyBack {
        const Buffer* buffer = nullptr;
        void* host = nullptr;
        std::size_t bytes = 0;
    };

    std::optional<DeviceSpan> mapped(const void* host, const std::size_t bytes) const
    {
        if (!host_memory_ || host == nullptr || bytes == 0)
            return std::nullopt;
        return host_memory_->find(host, bytes);
    }

    Device device_;
    std::string owner_;
    std::shared_ptr<const HostMemoryMap> host_memory_;
    std::vector<std::pair<std::uintptr_t, std::size_t>> inputs_;
    std::vector<CopyBack> copy_backs_;
    std::map<std::string, Buffer> slots_;
    // A deque keeps returned references valid as tables are added.
    std::deque<Table> tables_;
};

}  // namespace symmetrix::execution::metal
