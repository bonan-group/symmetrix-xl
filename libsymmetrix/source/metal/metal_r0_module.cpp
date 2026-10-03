#include "metal_r0_module.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

#include "metal_runtime.hpp"
#include "metal_spherical_harmonics.hpp"
#include "metal_staging.hpp"

namespace symmetrix::execution::metal {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t threadgroup_size = 128;
constexpr std::uint64_t simd_width = 32;
constexpr std::int32_t max_degree = 3;
// Edges per R0 reverse chunk: bounds the GPU-only harmonic and gradient
// workspaces to 2M edges, 512 MB at l_max = 3. SYMMETRIX_METAL_R0_REVERSE_CHUNK
// overrides it so tests can exercise multi-chunk execution on small graphs.
std::size_t reverse_chunk_size()
{
    static const std::size_t size = [] {
        const char* value = std::getenv("SYMMETRIX_METAL_R0_REVERSE_CHUNK");
        const long long parsed = value != nullptr ? std::atoll(value) : 0;
        return parsed > 0 ? static_cast<std::size_t>(parsed) : std::size_t(1) << 21;
    }();
    return size;
}

// Device pointers are 64-bit GPU addresses; the scalar header is 48 bytes so
// the pointer block starts 8-byte aligned in both layouts.
struct R0Args {
    std::uint32_t num_nodes;
    std::uint32_t active_type_count;
    std::uint32_t channels;
    std::uint32_t harmonic_count;
    std::uint32_t intervals;
    std::uint32_t functions;
    std::uint32_t num_edges;
    std::uint32_t apply_scale;
    float h;
    float x0;
    float cutoff;
    // First edge of the current reverse chunk; workspaces are chunk-relative.
    std::uint32_t edge_begin;
    std::uint64_t node_types;
    std::uint64_t num_neigh;
    std::uint64_t first_neigh;
    std::uint64_t neigh_types;
    std::uint64_t type_to_active;
    std::uint64_t edge_receivers;
    std::uint64_t radius;
    std::uint64_t coefficients;
    std::uint64_t harmonics;
    std::uint64_t density_scale;
    std::uint64_t unit_xyz;
    std::uint64_t harmonics_out;
    std::uint64_t gradients_out;
    std::uint64_t adjoint;
    std::uint64_t output;
};
static_assert(sizeof(R0Args) == 168);

struct HarmonicValueArgs {
    std::uint32_t num_edges;
    float cutoff;
    std::uint64_t directions;
    std::uint64_t radius;
    std::uint64_t values;
};
static_assert(sizeof(HarmonicValueArgs) == 32);

constexpr const char* r0_kernels = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct R0Args {
    uint num_nodes;
    uint active_type_count;
    uint channels;
    uint harmonic_count;
    uint intervals;
    uint functions;
    uint num_edges;
    uint apply_scale;
    float h;
    float x0;
    float cutoff;
    uint edge_begin;
    device const int* node_types;
    device const int* num_neigh;
    device const int* first_neigh;
    device const int* neigh_types;
    device const int* type_to_active;
    device const int* edge_receivers;
    device const float* radius;
    // [edge_type, interval, function] float4 of cubic coefficients.
    device const float4* coefficients;
    device const float* harmonics;
    device const float* density_scale;
    device const float* unit_xyz;
    device float* harmonics_out;
    device float* gradients_out;
    device const float* adjoint;
    device float* output;
};

struct SplinePoint {
    ulong base;
    float x;
    float xx;
    float xxx;
};

inline SplinePoint spline_point(constant R0Args& a, int edge_type, float r)
{
    int interval = int(floor((r - a.x0) / a.h));
    float x = r - a.x0 - a.h * float(interval);
    if (interval < 0) {
        interval = 0;
        x = 0.0f;
    } else if (interval >= int(a.intervals)) {
        interval = int(a.intervals) - 1;
        x = a.h;
    }
    const float xx = x * x;
    return {(ulong(edge_type) * a.intervals + ulong(interval)) * a.functions,
            x, xx, xx * x};
}

// One thread owns one (receiver, channel) output row of A0. The degree is a
// template parameter so the harmonic loops unroll and the accumulators stay
// in registers.
template <uint L>
kernel void symmetrix_r0_forward(
    constant R0Args& a [[buffer(0)]],
    uint flat [[thread_position_in_grid]])
{
    constexpr uint size = (L + 1) * (L + 1);
    const uint receiver = flat / a.channels;
    if (receiver >= a.num_nodes)
        return;
    const uint channel = flat % a.channels;
    const int receiver_type = a.type_to_active[a.node_types[receiver]];
    const int begin = a.first_neigh[receiver];
    const int end = begin + a.num_neigh[receiver];
    float accumulator[size];
    for (uint lm = 0; lm < size; ++lm)
        accumulator[lm] = 0.0f;
    for (int edge = begin; edge < end; ++edge) {
        const float r = a.radius[edge];
        if (!(r < a.cutoff))
            continue;
        const int neighbor_type = a.type_to_active[a.neigh_types[edge]];
        const SplinePoint p = spline_point(
            a, receiver_type * int(a.active_type_count) + neighbor_type, r);
        device const float* Y = a.harmonics + ulong(edge) * size;
        for (uint l = 0; l <= L; ++l) {
            const float4 c = a.coefficients[p.base + l * a.channels + channel];
            const float value = c.x + c.y * p.x + c.z * p.xx + c.w * p.xxx;
            for (uint lm = l * l; lm < (l + 1) * (l + 1); ++lm)
                accumulator[lm] += value * Y[lm];
        }
    }
    const float scale = a.apply_scale != 0 ? a.density_scale[receiver] : 1.0f;
    for (uint lm = 0; lm < size; ++lm)
        a.output[(ulong(receiver) * size + lm) * a.channels + channel] =
            accumulator[lm] * scale;
}

#define SYMMETRIX_R0_FORWARD(L) \
    template [[host_name("symmetrix_r0_forward_l" #L)]] kernel void \
    symmetrix_r0_forward<L>(constant R0Args&, uint);
SYMMETRIX_R0_FORWARD(0)
SYMMETRIX_R0_FORWARD(1)
SYMMETRIX_R0_FORWARD(2)
SYMMETRIX_R0_FORWARD(3)

// One thread evaluates the harmonics and Cartesian gradients of one edge
// into GPU-only buffers laid out as the host owner's Y and Y_grad blocks.
template <uint L>
kernel void symmetrix_r0_harmonics(
    constant R0Args& a [[buffer(0)]],
    uint local [[thread_position_in_grid]])
{
    constexpr uint size = (L + 1) * (L + 1);
    const uint edge = a.edge_begin + local;
    if (edge >= a.num_edges)
        return;
    const float3 u = float3(
        a.unit_xyz[3 * edge], a.unit_xyz[3 * edge + 1], a.unit_xyz[3 * edge + 2]);
    float values[size];
    float gradients[3 * size];
    symmetrix_sph_values_gradients<int(L)>(u, a.radius[edge], values, gradients);
    device float* Y = a.harmonics_out + ulong(local) * size;
    device float* G = a.gradients_out + ulong(local) * 3 * size;
    for (uint index = 0; index < size; ++index)
        Y[index] = values[index];
    for (uint index = 0; index < 3 * size; ++index)
        G[index] = gradients[index];
}

struct HarmonicValueArgs {
    uint num_edges;
    float cutoff;
    device const float* directions;
    device const float* radius;
    device float* values;
};

// One thread writes the normalized harmonics of one edge, zero beyond the
// cutoff, as launch_spherical_harmonic_values_from_directions.
template <uint L>
kernel void symmetrix_harmonic_values(
    constant HarmonicValueArgs& a [[buffer(0)]],
    uint edge [[thread_position_in_grid]])
{
    constexpr uint size = (L + 1) * (L + 1);
    if (edge >= a.num_edges)
        return;
    device float* Y = a.values + ulong(edge) * size;
    if (!(a.radius[edge] < a.cutoff)) {
        for (uint index = 0; index < size; ++index)
            Y[index] = 0.0f;
        return;
    }
    float values[size];
    symmetrix_sph_values<int(L)>(float3(a.directions[3 * edge],
        a.directions[3 * edge + 1], a.directions[3 * edge + 2]), values);
    for (uint index = 0; index < size; ++index)
        Y[index] = values[index];
}

#define SYMMETRIX_HARMONIC_VALUES(L) \
    template [[host_name("symmetrix_harmonic_values_l" #L)]] kernel void \
    symmetrix_harmonic_values<L>(constant HarmonicValueArgs&, uint);
SYMMETRIX_HARMONIC_VALUES(0)
SYMMETRIX_HARMONIC_VALUES(1)
SYMMETRIX_HARMONIC_VALUES(2)
SYMMETRIX_HARMONIC_VALUES(3)

#define SYMMETRIX_R0_HARMONICS(L) \
    template [[host_name("symmetrix_r0_harmonics_l" #L)]] kernel void \
    symmetrix_r0_harmonics<L>(constant R0Args&, uint);
SYMMETRIX_R0_HARMONICS(0)
SYMMETRIX_R0_HARMONICS(1)
SYMMETRIX_R0_HARMONICS(2)
SYMMETRIX_R0_HARMONICS(3)

// One 32-lane SIMD group owns one edge; lanes split the channels so adjoint
// and spline loads coalesce, and simd_sum reduces the force components.
template <uint L>
kernel void symmetrix_r0_reverse(
    constant R0Args& a [[buffer(0)]],
    uint flat [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint size = (L + 1) * (L + 1);
    const uint local = flat / 32u;
    const uint edge = a.edge_begin + local;
    if (edge >= a.num_edges)
        return;
    const int receiver = a.edge_receivers[edge];
    const float r = a.radius[edge];
    if (receiver < 0 || !(r < a.cutoff))
        return;
    const int receiver_type = a.type_to_active[a.node_types[receiver]];
    const int neighbor_type = a.type_to_active[a.neigh_types[edge]];
    const SplinePoint p = spline_point(
        a, receiver_type * int(a.active_type_count) + neighbor_type, r);
    const float ux = a.unit_xyz[3 * edge];
    const float uy = a.unit_xyz[3 * edge + 1];
    const float uz = a.unit_xyz[3 * edge + 2];
    device const float* Y = a.harmonics_out + ulong(local) * size;
    device const float* G = a.gradients_out + ulong(local) * 3 * size;
    device const float* adjoint = a.adjoint + ulong(receiver) * size * a.channels;
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    for (uint channel = lane; channel < a.channels; channel += 32u) {
        for (uint l = 0; l <= L; ++l) {
            const float4 c = a.coefficients[p.base + l * a.channels + channel];
            const float value = c.x + c.y * p.x + c.z * p.xx + c.w * p.xxx;
            const float derivative = c.y + 2.0f * c.z * p.x + 3.0f * c.w * p.xx;
            for (uint lm = l * l; lm < (l + 1) * (l + 1); ++lm) {
                const float w = adjoint[lm * a.channels + channel];
                const float radial = derivative * Y[lm] * w;
                const float angular = value * w;
                fx += radial * ux + angular * G[lm];
                fy += radial * uy + angular * G[size + lm];
                fz += radial * uz + angular * G[2 * size + lm];
            }
        }
    }
    fx = simd_sum(fx);
    fy = simd_sum(fy);
    fz = simd_sum(fz);
    if (lane == 0) {
        a.output[3 * edge] = -fx;
        a.output[3 * edge + 1] = -fy;
        a.output[3 * edge + 2] = -fz;
    }
}

#define SYMMETRIX_R0_REVERSE(L) \
    template [[host_name("symmetrix_r0_reverse_l" #L)]] kernel void \
    symmetrix_r0_reverse<L>(constant R0Args&, uint, uint);
SYMMETRIX_R0_REVERSE(0)
SYMMETRIX_R0_REVERSE(1)
SYMMETRIX_R0_REVERSE(2)
SYMMETRIX_R0_REVERSE(3)
)MSL";

bool profiling_enabled()
{
    const char* value = std::getenv("SYMMETRIX_METAL_PROFILE");
    return value != nullptr && std::string(value) == "1";
}

double seconds_since(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now()-start).count();
}

}  // namespace

struct MetalR0Module::Impl {
    Device device;
    // Indexed by l_max, like the harmonic pipelines.
    Pipeline forward[max_degree+1];
    Pipeline reverse[max_degree+1];
    // Indexed by l_max; harmonics are compile-time sized per pipeline.
    Pipeline harmonics[max_degree+1];
    Pipeline harmonic_values[max_degree+1];
    std::unique_ptr<MetalStaging> staging;
    MetalR0Statistics statistics;

    [[noreturn]] void fail(const std::string& message) const
    {
        staging->fail(message);
    }

    static int degree_of(const std::int32_t harmonic_count)
    {
        for (int l = 0; l <= max_degree; ++l)
            if ((l+1)*(l+1) == harmonic_count)
                return l;
        return -1;
    }

    void validate(const MetalR0Graph& graph, const MetalR0Spline& spline,
        const std::int32_t channels, const std::int32_t harmonic_count) const
    {
        if (graph.num_nodes < 0 || channels <= 0)
            fail("graph extents must be nonnegative and channels positive");
        if (degree_of(harmonic_count) < 0)
            fail("R0 supports (l_max+1)^2 harmonics for l_max from 0 through 3");
        if (spline.coefficients == nullptr || spline.intervals == 0
                || spline.functions == 0 || spline.edge_types == 0)
            fail("R0 spline coefficients are empty");
        const std::size_t degrees =
            static_cast<std::size_t>(degree_of(harmonic_count))+1;
        if (spline.functions != degrees*static_cast<std::size_t>(channels))
            fail("R0 spline functions do not match the channel layout");
        if (static_cast<std::size_t>(graph.active_type_count)
                *static_cast<std::size_t>(graph.active_type_count)
                > spline.edge_types)
            fail("R0 spline edge types do not cover the active types");
        if (graph.edge_capacity > UINT32_MAX
                || static_cast<std::size_t>(graph.num_nodes)*channels > UINT32_MAX
                || graph.edge_capacity*simd_width > UINT32_MAX)
            fail("R0 graph exceeds the 32-bit Metal grid range");
    }

    // Uploads the receiver-ordered graph shared by both passes.
    void stage_graph(const MetalR0Graph& graph, const MetalR0Spline& spline,
        CommandBatch& batch, R0Args& args)
    {
        const std::size_t nodes = static_cast<std::size_t>(graph.num_nodes);
        const std::size_t edges = graph.edge_capacity;
        Buffer& node_types =
            staging->upload("node_types", graph.node_types, graph.node_type_count);
        Buffer& num_neigh = staging->upload("num_neigh", graph.num_neigh, nodes);
        Buffer& first_neigh = staging->upload("first_neigh", graph.first_neigh, nodes);
        Buffer& neigh_types = staging->upload("neigh_types", graph.neigh_types, edges);
        Buffer& type_to_active = staging->upload(
            "type_to_active", graph.type_to_active, graph.type_map_count);
        Buffer& radius = staging->upload_narrowed("radius", graph.radius, edges);
        const Buffer& coefficients = staging->persistent_spline4(spline.coefficients,
            spline.edge_types*spline.intervals, spline.functions);
        for (const Buffer* buffer : std::initializer_list<const Buffer*>{
                &node_types, &num_neigh, &first_neigh, &neigh_types,
                &type_to_active, &radius, &coefficients})
            batch.use_buffer(*buffer, false);
        args.num_nodes = static_cast<std::uint32_t>(nodes);
        args.active_type_count = static_cast<std::uint32_t>(graph.active_type_count);
        args.intervals = static_cast<std::uint32_t>(spline.intervals);
        args.functions = static_cast<std::uint32_t>(spline.functions);
        args.num_edges = static_cast<std::uint32_t>(edges);
        args.h = static_cast<float>(spline.h);
        args.x0 = static_cast<float>(spline.x0);
        args.cutoff = static_cast<float>(graph.cutoff);
        args.node_types = node_types.gpu_address();
        args.num_neigh = num_neigh.gpu_address();
        args.first_neigh = first_neigh.gpu_address();
        args.neigh_types = neigh_types.gpu_address();
        args.type_to_active = type_to_active.gpu_address();
        args.radius = radius.gpu_address();
        args.coefficients = coefficients.gpu_address();
    }
};

MetalR0Module::MetalR0Module() : impl_(std::make_unique<Impl>())
{
    impl_->device = Device::system_default();
    impl_->staging = std::make_unique<MetalStaging>(impl_->device, "R0");
    const Library library = impl_->device.compile(
        spherical_harmonics_msl()+r0_kernels, module_compile_options());
    for (int l = 0; l <= max_degree; ++l) {
        impl_->forward[l] = impl_->device.pipeline(
            library, "symmetrix_r0_forward_l"+std::to_string(l));
        impl_->reverse[l] = impl_->device.pipeline(
            library, "symmetrix_r0_reverse_l"+std::to_string(l));
        if (impl_->reverse[l].thread_execution_width() != simd_width)
            impl_->fail("the reverse kernels require 32-wide SIMD groups");
        impl_->harmonics[l] = impl_->device.pipeline(
            library, "symmetrix_r0_harmonics_l"+std::to_string(l));
        impl_->harmonic_values[l] = impl_->device.pipeline(
            library, "symmetrix_harmonic_values_l"+std::to_string(l));
    }
    for (const Pipeline* pipeline : {&impl_->forward[0], &impl_->forward[1],
            &impl_->forward[2], &impl_->forward[3], &impl_->reverse[0],
            &impl_->reverse[1], &impl_->reverse[2], &impl_->reverse[3],
            &impl_->harmonics[0], &impl_->harmonics[1], &impl_->harmonics[2],
            &impl_->harmonics[3], &impl_->harmonic_values[0],
            &impl_->harmonic_values[1], &impl_->harmonic_values[2],
            &impl_->harmonic_values[3]})
        if (pipeline->max_total_threads_per_threadgroup() < threadgroup_size)
            impl_->fail("pipeline '"+pipeline->function_name()
                +"' cannot run the required threadgroup size");
}

MetalR0Module::~MetalR0Module() = default;

const MetalR0Statistics& MetalR0Module::statistics() const
{
    return impl_->statistics;
}

void MetalR0Module::release_persistent_tables()
{
    impl_->staging->release_tables();
}

void MetalR0Module::set_host_memory(std::shared_ptr<const HostMemoryMap> host_memory)
{
    impl_->staging->set_host_memory(std::move(host_memory));
}

void MetalR0Module::harmonic_values(
    const std::size_t edges,
    const std::int32_t harmonic_count,
    const double cutoff,
    const float* unit_directions,
    const double* radius,
    float* values)
{
    auto& m = *impl_;
    const int degree = Impl::degree_of(harmonic_count);
    if (degree < 0)
        m.fail("harmonic values support l_max from 0 through 3");
    if (edges > UINT32_MAX)
        m.fail("harmonic values exceed the 32-bit Metal grid range");
    if (edges == 0)
        return;
    const auto staging = Clock::now();
    auto batch = m.device.begin();
    m.staging->begin();
    const DeviceSpan directions =
        m.staging->input("unit_directions", unit_directions, 3*edges);
    Buffer& r = m.staging->upload_narrowed("radius", radius, edges);
    const DeviceSpan out = m.staging->output("harmonic_values", values,
        edges*static_cast<std::size_t>(harmonic_count));
    const HarmonicValueArgs args{static_cast<std::uint32_t>(edges),
        static_cast<float>(cutoff), directions.address(), r.gpu_address(),
        out.address()};
    batch.use_buffer(*directions.buffer, false).use_buffer(r, false)
        .use_buffer(*out.buffer, true).set_value(0, args)
        .dispatch_threads(m.harmonic_values[degree], {edges}, {threadgroup_size});
    m.statistics.staging_seconds += seconds_since(staging);
    const double gpu_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += gpu_seconds;
    m.statistics.forward_seconds += gpu_seconds;
    const auto readback = Clock::now();
    m.staging->complete();
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.harmonic_launches;
}

void MetalR0Module::forward(
    const MetalR0Graph& graph,
    const MetalR0Spline& spline,
    const std::int32_t channels,
    const std::int32_t harmonic_count,
    const float* harmonics,
    const double* density_scale,
    float* output)
{
    auto& m = *impl_;
    m.validate(graph, spline, channels, harmonic_count);
    const std::size_t nodes = static_cast<std::size_t>(graph.num_nodes);
    if (nodes == 0)
        return;
    const auto staging = Clock::now();
    auto batch = m.device.begin();
    m.staging->begin();
    R0Args args{};
    m.stage_graph(graph, spline, batch, args);
    args.channels = static_cast<std::uint32_t>(channels);
    args.harmonic_count = static_cast<std::uint32_t>(harmonic_count);
    args.apply_scale = density_scale != nullptr ? 1u : 0u;
    const DeviceSpan Y = m.staging->input("harmonics", harmonics,
        graph.edge_capacity*static_cast<std::size_t>(harmonic_count));
    Buffer& scale = density_scale != nullptr
        ? m.staging->upload_narrowed("density_scale", density_scale, nodes)
        : m.staging->slot("density_scale", 16);
    const DeviceSpan out = m.staging->output("forward_output", output,
        nodes*static_cast<std::size_t>(harmonic_count)*channels);
    args.harmonics = Y.address();
    args.density_scale = scale.gpu_address();
    args.output = out.address();
    batch.use_buffer(*Y.buffer, false).use_buffer(scale, false)
        .use_buffer(*out.buffer, true);
    batch.set_value(0, args)
        .dispatch_threads(m.forward[Impl::degree_of(harmonic_count)],
            {nodes*channels}, {threadgroup_size});
    m.statistics.staging_seconds += seconds_since(staging);
    const double gpu_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += gpu_seconds;
    m.statistics.forward_seconds += gpu_seconds;
    const auto readback = Clock::now();
    m.staging->complete();
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.forward_launches;
}

void MetalR0Module::coordinate_reverse(
    const MetalR0Graph& graph,
    const MetalR0Spline& spline,
    const std::int32_t channels,
    const std::int32_t harmonic_count,
    const void* coordinates,
    const std::uint32_t coordinate_scalar_size,
    const bool coordinates_are_unit,
    const float* output_adjoint,
    double* directed_forces)
{
    auto& m = *impl_;
    m.validate(graph, spline, channels, harmonic_count);
    const std::size_t nodes = static_cast<std::size_t>(graph.num_nodes);
    const std::size_t edges = graph.edge_capacity;
    if (nodes == 0 || edges == 0)
        return;
    const auto staging = Clock::now();
    auto batch = m.device.begin();
    m.staging->begin();
    R0Args args{};
    m.stage_graph(graph, spline, batch, args);
    args.channels = static_cast<std::uint32_t>(channels);
    args.harmonic_count = static_cast<std::uint32_t>(harmonic_count);

    // Receivers of the edges they own; edges outside every receiver range
    // are skipped by the kernel.
    Buffer& receivers = m.staging->slot("edge_receivers", edges*sizeof(std::int32_t));
    auto* receiver_of = receivers.data<std::int32_t>();
    std::fill(receiver_of, receiver_of+edges, -1);
    for (std::size_t node = 0; node < nodes; ++node) {
        const std::int64_t begin = graph.first_neigh[node];
        const std::int64_t end = begin+graph.num_neigh[node];
        if (begin < 0 || end > static_cast<std::int64_t>(edges))
            m.fail("receiver edge range exceeds the edge capacity");
        for (std::int64_t edge = begin; edge < end; ++edge)
            receiver_of[edge] = static_cast<std::int32_t>(node);
    }
    Buffer& unit = m.staging->slot("unit_xyz", 3*edges*sizeof(float));
    float* unit_xyz = unit.data<float>();
    if (coordinate_scalar_size != sizeof(float)
            && coordinate_scalar_size != sizeof(double))
        m.fail("coordinates must be FP32 or FP64");
    if (coordinate_scalar_size == sizeof(float) && !coordinates_are_unit)
        m.fail("FP32 coordinates must be unit directions");
    if (coordinates_are_unit && coordinate_scalar_size == sizeof(float)) {
        std::memcpy(unit_xyz, coordinates, 3*edges*sizeof(float));
    } else {
        const auto* xyz = static_cast<const double*>(coordinates);
        for (std::size_t edge = 0; edge < edges; ++edge)
            for (std::size_t k = 0; k < 3; ++k)
                unit_xyz[3*edge+k] = static_cast<float>(coordinates_are_unit
                    ? xyz[3*edge+k] : xyz[3*edge+k]/graph.radius[edge]);
    }
    // GPU-only harmonic workspaces, bounded by processing the edges in
    // chunks; no host storage or transfer.
    const std::size_t chunk = std::min(edges, reverse_chunk_size());
    const std::size_t harmonic_values = chunk*static_cast<std::size_t>(harmonic_count);
    Buffer& Y = m.staging->slot("harmonics", harmonic_values*sizeof(float));
    Buffer& gradients = m.staging->slot("gradients", 3*harmonic_values*sizeof(float));
    const DeviceSpan adjoint = m.staging->input("adjoint", output_adjoint,
        nodes*static_cast<std::size_t>(harmonic_count)*channels);
    Buffer& forces = m.staging->zeroed("forces", 3*edges);
    args.edge_receivers = receivers.gpu_address();
    args.unit_xyz = unit.gpu_address();
    args.harmonics_out = Y.gpu_address();
    args.gradients_out = gradients.gpu_address();
    args.adjoint = adjoint.address();
    args.output = forces.gpu_address();
    for (const Buffer* buffer : std::initializer_list<const Buffer*>{
            &receivers, &unit, adjoint.buffer})
        batch.use_buffer(*buffer, false);
    batch.use_buffer(Y, true).use_buffer(gradients, true).use_buffer(forces, true);
    // The serial encoder orders each chunk's harmonic pass before its edge
    // pass, and both before the next chunk overwrites the workspaces.
    const int degree = Impl::degree_of(harmonic_count);
    for (std::size_t begin = 0; begin < edges; begin += chunk) {
        const std::size_t count = std::min(chunk, edges-begin);
        args.edge_begin = static_cast<std::uint32_t>(begin);
        args.num_edges = static_cast<std::uint32_t>(begin+count);
        batch.set_value(0, args)
            .dispatch_threads(m.harmonics[degree], {count}, {threadgroup_size});
        if (profiling_enabled()) {
            const double harmonics_seconds = batch.submit_and_wait().gpu_seconds;
            m.statistics.gpu_seconds += harmonics_seconds;
            m.statistics.reverse_seconds += harmonics_seconds;
            m.statistics.harmonics_seconds += harmonics_seconds;
            batch = m.device.begin();
            m.stage_graph(graph, spline, batch, args);
            args.edge_begin = static_cast<std::uint32_t>(begin);
            args.num_edges = static_cast<std::uint32_t>(begin+count);
            for (const Buffer* buffer : std::initializer_list<const Buffer*>{
                    &receivers, &unit, adjoint.buffer, &Y, &gradients})
                batch.use_buffer(*buffer, false);
            batch.use_buffer(forces, true).set_value(0, args);
        }
        batch.dispatch_threads(m.reverse[degree], {count*simd_width}, {threadgroup_size});
    }
    m.statistics.staging_seconds += seconds_since(staging);
    const double gpu_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += gpu_seconds;
    m.statistics.reverse_seconds += gpu_seconds;
    const auto readback = Clock::now();
    const float* values = forces.data<float>();
    for (std::size_t i = 0; i < 3*edges; ++i)
        directed_forces[i] += static_cast<double>(values[i]);
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.reverse_launches;
}

}  // namespace symmetrix::execution::metal
