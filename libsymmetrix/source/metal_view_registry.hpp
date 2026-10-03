#pragma once

// Keeps Kokkos host allocations mapped for in-place Metal access while the
// evaluator still references them. Each mapped view is pinned by a copy, so
// its allocation cannot be freed while the GPU wrap exists; once the pinned
// copy is the only reference left, the wrap is released before the memory.

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "metal_staging.hpp"

namespace symmetrix::execution::metal {

class MetalViewRegistry {
public:
    // Small views gain nothing from mapping and may share pages with
    // unrelated allocations.
    static constexpr std::size_t minimum_bytes = 256*1024;

    explicit MetalViewRegistry(Device device)
        : map_(std::make_shared<HostMemoryMap>(std::move(device)))
    {}

    ~MetalViewRegistry() { release_all(); }
    MetalViewRegistry(const MetalViewRegistry&) = delete;
    MetalViewRegistry& operator=(const MetalViewRegistry&) = delete;

    std::shared_ptr<const HostMemoryMap> host_memory() const { return map_; }

    std::size_t size() const { return pinned_.size(); }

    template <class ViewType>
    void map(const ViewType& view)
    {
        static_assert(std::is_same_v<
            typename ViewType::memory_space, Kokkos::HostSpace>);
        using Value = typename ViewType::non_const_value_type;
        const std::size_t bytes = view.span()*sizeof(Value);
        // Unmanaged views have no reference count to pin.
        if (view.data() == nullptr || view.use_count() < 1
                || !view.span_is_contiguous() || bytes < minimum_bytes)
            return;
        release_unused();
        for (const auto& entry : pinned_)
            if (entry->data() == view.data())
                return;
        map_->map(view.data(), bytes);
        pinned_.push_back(std::make_unique<Pinned<ViewType>>(view));
    }

    // Drops mappings whose allocation only this registry still references.
    void release_unused()
    {
        std::erase_if(pinned_, [&](const std::unique_ptr<PinnedBase>& entry) {
            if (entry->use_count() > 1)
                return false;
            map_->unmap(entry->data());
            return true;
        });
    }

    void release_all()
    {
        for (const auto& entry : pinned_)
            map_->unmap(entry->data());
        pinned_.clear();
    }

private:
    struct PinnedBase {
        virtual ~PinnedBase() = default;
        virtual const void* data() const = 0;
        virtual int use_count() const = 0;
    };

    template <class ViewType>
    struct Pinned final : PinnedBase {
        explicit Pinned(ViewType pinned) : view(std::move(pinned)) {}
        const void* data() const override { return view.data(); }
        int use_count() const override { return view.use_count(); }
        ViewType view;
    };

    std::shared_ptr<HostMemoryMap> map_;
    std::vector<std::unique_ptr<PinnedBase>> pinned_;
};

}  // namespace symmetrix::execution::metal
