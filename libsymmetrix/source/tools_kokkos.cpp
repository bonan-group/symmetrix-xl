#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <numeric>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

#include <Kokkos_Core.hpp>

#include "tools_kokkos.hpp"

namespace {
std::mutex kokkos_lifecycle_mutex;
std::size_t kokkos_live_object_count = 0;

int requested_kokkos_threads()
{
    const char* name = "KOKKOS_NUM_THREADS";
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        name = "OMP_NUM_THREADS";
        raw = std::getenv(name);
    }
    if (raw == nullptr || raw[0] == '\0')
        return 0;
    const std::string_view value(raw);
    const auto end = value.find(',');
    const auto length = end == std::string_view::npos ? value.size() : end;
    int threads = 0;
    const auto parsed = std::from_chars(
        value.data(), value.data()+length, threads);
    if (parsed.ec != std::errc() || parsed.ptr != value.data()+length
            || threads <= 0)
        throw std::invalid_argument(
            std::string(name)+" must start with a positive integer.");
    return threads;
}
}

void _init_kokkos()
{
    const std::scoped_lock lock(kokkos_lifecycle_mutex);
    Kokkos::InitializationSettings settings;
#if defined(KOKKOS_ENABLE_OPENMP)
    const int threads = requested_kokkos_threads();
    if (threads > 0)
        settings.set_num_threads(threads);
#endif
    settings.set_map_device_id_by("random");
    Kokkos::initialize(settings);
}

void _finalize_kokkos()
{
    const std::scoped_lock lock(kokkos_lifecycle_mutex);
    const auto live_objects = kokkos_live_object_count;
    if (live_objects != 0)
        throw std::runtime_error(
            "Cannot finalize Kokkos while "+std::to_string(live_objects)
            +" Symmetrix evaluator object(s) are alive.");
    Kokkos::finalize();
}

bool _kokkos_is_initialized()
{
    const std::scoped_lock lock(kokkos_lifecycle_mutex);
    return Kokkos::is_initialized();
}

std::size_t _kokkos_live_object_count()
{
    const std::scoped_lock lock(kokkos_lifecycle_mutex);
    return kokkos_live_object_count;
}

void retain_kokkos_live_object()
{
    const std::scoped_lock lock(kokkos_lifecycle_mutex);
    if (Kokkos::is_finalized())
        throw std::runtime_error(
            "Cannot construct a Symmetrix evaluator after Kokkos finalization.");
    kokkos_live_object_count += 1;
}

void release_kokkos_live_object() noexcept
{
    const std::scoped_lock lock(kokkos_lifecycle_mutex);
    kokkos_live_object_count -= 1;
}

std::string _kokkos_default_execution_space()
{
    return Kokkos::DefaultExecutionSpace::name();
}

struct InitView {
  explicit InitView(view_type _v) : m_view(_v) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    m_view(i) = -(i + 1);
    //m_view(i) = (i + 1);
  }

 private:
  view_type m_view;
};

struct ModifyView {
  explicit ModifyView(view_type _v) : m_view(_v) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    m_view(i) *= 2;
  }

 private:
  view_type m_view;
};

using exec_space = typename view_type::traits::execution_space;

view_type generate_view(size_t n)
{
    if (!Kokkos::is_initialized()) {
    std::cerr << "[user-bindings]> Initializing Kokkos..." << std::endl;
    Kokkos::initialize();
    }
    std::cerr << "[user-bindings]> Generating View..." << std::flush;
    view_type _v("user_view", n);
    Kokkos::RangePolicy<exec_space, int> range(0, n);
    Kokkos::parallel_for("generate_view", range, InitView{_v});
    std::cerr << " Done." << std::endl;
    return _v;
}

void modify_view(view_type _v) {
  std::cerr << "[user-bindings]> Modifying View..." << std::flush;
  Kokkos::RangePolicy<exec_space, int> range(0, _v.extent(0));
  Kokkos::parallel_for("modify_view", range, ModifyView{_v});
  std::cerr << " Done." << std::endl;
}
