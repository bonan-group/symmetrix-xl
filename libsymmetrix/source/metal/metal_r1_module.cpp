#include "metal_r1_module.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

#include "metal_r1_abi.hpp"
#include "metal_runtime.hpp"
#include "metal_staging.hpp"

namespace symmetrix::execution::metal {

namespace {

using Clock = std::chrono::steady_clock;

// Below the edge kernel's 384-thread pipeline limit and a multiple of the
// 32-wide Apple SIMD group.
constexpr std::uint64_t threadgroup_size = 128;
constexpr std::uint64_t simd_width = 32;

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error("Metal R1: "+message);
}

bool profiling_enabled()
{
    const char* value = std::getenv("SYMMETRIX_METAL_PROFILE");
    return value != nullptr && std::string(value) == "1";
}

double seconds_since(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now()-start).count();
}

std::uint32_t checked_u32(const std::int64_t value, const char* field)
{
    if (value < 0 || value > static_cast<std::int64_t>(UINT32_MAX))
        fail(std::string(field)+" is outside the 32-bit Metal grid range");
    return static_cast<std::uint32_t>(value);
}


}  // namespace

constexpr const char* row_scale_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct RowScaleArgs {
    uint num_nodes;
    uint row_length;
    device float* rows;
    device const float* factors;
};

kernel void symmetrix_scale_node_rows(
    constant RowScaleArgs& a [[buffer(0)]],
    uint flat [[thread_position_in_grid]])
{
    const uint node = flat / a.row_length;
    if (node < a.num_nodes)
        a.rows[flat] *= a.factors[node];
}
)MSL";

struct RowScaleArgs {
    std::uint32_t num_nodes;
    std::uint32_t row_length;
    std::uint64_t rows;
    std::uint64_t factors;
};
static_assert(sizeof(RowScaleArgs) == 24);

struct MetalR1Module::Impl {
    Device device;
    Pipeline forward;
    Pipeline row_scale;
    Pipeline source;
    Pipeline edge;
    // Present when channels split into 32-wide blocks; one threadgroup per
    // (receiver, block) writes one partial force plane per block.
    Pipeline edge_blocked;
    std::size_t edge_blocks = 0;
    MetalR1Shape shape;
    std::string device_name;
    MetalR1Statistics statistics;
    std::unique_ptr<MetalStaging> staging;

    // GPU buffers whose contents equal a host view written by the last call.
    struct Resident {
        const void* host = nullptr;
        std::size_t count = 0;

        bool matches(const void* pointer, const std::size_t elements) const
        {
            return host != nullptr && host == pointer && count == elements;
        }
    };
    Resident phi1;
    Resident phi1_adjoint;
    // A1 already computed by a forward that fused the A1 GEMMs.
    struct FusedA1 {
        const float* phi1 = nullptr;
        const float* a1 = nullptr;
        std::size_t nodes = 0;
    } fused_a1;

    // A1 = Phi1 W as 2l+1 strided GEMMs per degree over every node.
    void encode_a1(CommandBatch& batch, const DeviceSpan& input,
        const DeviceSpan& output, const std::size_t num_nodes,
        const MetalA1Layout& layout)
    {
        const std::size_t channels = static_cast<std::size_t>(shape.channels);
        const auto rows = static_cast<std::uint32_t>(num_nodes);
        const auto c = static_cast<std::uint32_t>(channels);
        const std::size_t input_row = layout.num_lme*channels*sizeof(float);
        const std::size_t output_row = layout.num_lm*channels*sizeof(float);
        for (int l = 0; l <= layout.l_max; ++l) {
            const auto inputs = static_cast<std::uint32_t>(layout.eta[l]*channels);
            if (inputs == 0)
                continue;
            const Buffer& weights = staging->persistent(layout.weights[l],
                static_cast<std::size_t>(inputs)*channels*sizeof(float));
            for (int row = 0; row < 2*l+1; ++row)
                batch.gemm(
                    {input.buffer, input.offset
                        +(layout.lme[l]*channels+static_cast<std::size_t>(row)*inputs)
                        *sizeof(float), rows, inputs, input_row},
                    {&weights, 0, inputs, c, channels*sizeof(float)},
                    {output.buffer, output.offset
                        +(static_cast<std::size_t>(l*l+row)*channels)*sizeof(float),
                        rows, c, output_row});
        }
    }

    // Degrees without paths contribute zero rows, as the zero-depth host GEMM.
    void zero_empty_a1_degrees(float* a1, const std::size_t num_nodes,
        const MetalA1Layout& layout) const
    {
        const std::size_t channels = static_cast<std::size_t>(shape.channels);
        for (int l = 0; l <= layout.l_max; ++l)
            if (layout.eta[l] == 0)
                for (std::size_t node = 0; node < num_nodes; ++node)
                    std::memset(a1+(node*layout.num_lm+l*l)*channels, 0,
                        (2*l+1)*channels*sizeof(float));
    }

    static void validate(const MetalA1Layout& layout, MetalStaging& staging)
    {
        if (layout.l_max < 0 || layout.l_max > 3)
            staging.fail("A1 supports l_max from 0 through 3");
    }

    template <class T>
    Buffer& upload(const std::string& name, const T* values, const std::size_t count)
    {
        return staging->upload(name, values, count);
    }

    Buffer& upload_narrowed(
        const std::string& name, const double* values, const std::size_t count)
    {
        return staging->upload_narrowed(name, values, count);
    }

    Buffer& slot(const std::string& name, const std::size_t bytes)
    {
        return staging->slot(name, bytes);
    }

    Buffer& zeroed(const std::string& name, const std::size_t count)
    {
        return staging->zeroed(name, count);
    }

    // Cached by host address until the evaluator rebuilds the radial tables.
    MetalR1RadialSpline spline(
        const SymmetrixJitHostRadialSplineV2& radial, CommandBatch& batch)
    {
        const Buffer& coefficients = staging->persistent_spline4(
            static_cast<const float*>(radial.coefficients),
            static_cast<std::size_t>(radial.edge_types)*radial.intervals,
            radial.functions);
        batch.use_buffer(coefficients, false);
        return {
            radial.edge_types,
            radial.intervals,
            radial.functions,
            static_cast<float>(radial.h),
            static_cast<float>(radial.x0),
            0,
            coefficients.gpu_address()};
    }
};

MetalR1Module::MetalR1Module(
    const std::string_view msl_source, const MetalR1Shape& shape)
    : impl_(std::make_unique<Impl>())
{
    if (shape.channels <= 0 || shape.edge_harmonics <= 0
            || shape.source_harmonics <= 0 || shape.output_components <= 0)
        fail("module shape must be positive");
    impl_->shape = shape;
    impl_->device = Device::system_default();
    impl_->device_name = impl_->device.information().device_name;
    impl_->staging = std::make_unique<MetalStaging>(impl_->device, "R1");
    const Library library =
        impl_->device.compile(msl_source, module_compile_options());
    impl_->forward = impl_->device.pipeline(library, "symmetrix_r1_forward");
    impl_->source = impl_->device.pipeline(library, "symmetrix_r1_source");
    impl_->edge = impl_->device.pipeline(library, "symmetrix_r1_edge");
    if (library.has_function("symmetrix_r1_edge_blocked")) {
        impl_->edge_blocked =
            impl_->device.pipeline(library, "symmetrix_r1_edge_blocked");
        impl_->edge_blocks = static_cast<std::size_t>(shape.channels)/simd_width;
    }
    for (const Pipeline* pipeline :
            {&impl_->forward, &impl_->source, &impl_->edge}) {
        if (pipeline->max_total_threads_per_threadgroup() < threadgroup_size)
            fail("pipeline '"+pipeline->function_name()
                +"' cannot run the required threadgroup size");
    }
    if (impl_->edge.thread_execution_width() != simd_width)
        fail("the edge kernel requires 32-wide SIMD groups");
    impl_->row_scale = impl_->device.pipeline(
        impl_->device.compile(row_scale_source, module_compile_options()),
        "symmetrix_scale_node_rows");
}

MetalR1Module::~MetalR1Module() = default;

const MetalR1Shape& MetalR1Module::shape() const { return impl_->shape; }

const std::string& MetalR1Module::device_name() const
{
    return impl_->device_name;
}

void MetalR1Module::discard_pending_results()
{
    impl_->phi1 = {};
    impl_->phi1_adjoint = {};
    impl_->fused_a1 = {};
}

void MetalR1Module::release_persistent_tables()
{
    impl_->staging->release_tables();
}

void MetalR1Module::set_host_memory(std::shared_ptr<const HostMemoryMap> host_memory)
{
    impl_->staging->set_host_memory(std::move(host_memory));
}

const MetalR1Statistics& MetalR1Module::statistics() const
{
    return impl_->statistics;
}

void MetalR1Module::forward(
    const SymmetrixJitHostR1ForwardArgsV2& args,
    const MetalR1ForwardExtents& extents,
    const MetalA1Request* a1_request)
{
    if (args.struct_size != sizeof(SymmetrixJitHostR1ForwardArgsV2))
        fail("forward packet size does not match");
    auto& m = *impl_;
    const MetalR1Shape& s = m.shape;
    m.phi1 = {};
    m.fused_a1 = {};
    const std::size_t nodes = checked_u32(args.num_nodes, "num_nodes");
    const std::size_t edges = checked_u32(args.num_edges, "num_edges");
    if (nodes == 0)
        return;
    const std::size_t channels = static_cast<std::size_t>(s.channels);
    checked_u32(static_cast<std::int64_t>(nodes*channels), "forward grid");

    const auto staging = Clock::now();
    auto batch = m.device.begin();
    m.staging->begin();
    Buffer& node_types = m.upload("node_types", args.node_types, extents.node_types);
    Buffer& num_neigh = m.upload("num_neigh", args.num_neigh, nodes);
    Buffer& first_neigh = m.upload("first_neigh", args.first_neigh, nodes);
    Buffer& neigh_indices = m.upload("neigh_indices", args.neigh_indices, edges);
    Buffer& neigh_types = m.upload("neigh_types", args.neigh_types, edges);
    Buffer& type_to_active =
        m.upload("type_to_active", args.type_to_active, extents.type_to_active);
    Buffer& radius = m.upload_narrowed("radius", args.radius, edges);
    const DeviceSpan harmonics = m.staging->input("harmonics_values",
        static_cast<const float*>(args.harmonics_values), edges*s.edge_harmonics);
    const DeviceSpan features = m.staging->input("neighbor_features",
        static_cast<const float*>(args.neighbor_features),
        extents.neighbor_feature_nodes*s.source_harmonics*channels);
    const std::size_t output_count = nodes*s.output_components*channels;
    float* phi1 = static_cast<float*>(args.output);
    const DeviceSpan output = m.staging->output("forward_output", phi1, output_count);
    // A staged Phi1 stays in its slot for a1_forward; mapped Phi1 needs no tag.
    const bool phi1_staged = !m.staging->is_mapped(phi1, output_count*sizeof(float));

    const MetalR1ForwardArgs packet{
        args.active_type_count,
        static_cast<std::uint32_t>(nodes),
        static_cast<float>(args.cutoff),
        0,
        node_types.gpu_address(),
        num_neigh.gpu_address(),
        first_neigh.gpu_address(),
        neigh_indices.gpu_address(),
        neigh_types.gpu_address(),
        type_to_active.gpu_address(),
        radius.gpu_address(),
        m.spline(args.radial, batch),
        harmonics.address(),
        features.address(),
        output.address()};
    for (const Buffer* buffer : std::initializer_list<const Buffer*>{&node_types, &num_neigh, &first_neigh,
            &neigh_indices, &neigh_types, &type_to_active, &radius,
            harmonics.buffer, features.buffer})
        batch.use_buffer(*buffer, false);
    batch.use_buffer(*output.buffer, true);
    batch.set_value(0, packet)
        .dispatch_threads(m.forward, {nodes*channels}, {threadgroup_size});
    // The A1 GEMMs read Phi1 in the same submission, so compute_A1 needs no
    // further launch; the serial command order makes Phi1 complete first.
    if (a1_request != nullptr) {
        Impl::validate(a1_request->layout, *m.staging);
        const DeviceSpan a1 = m.staging->output("a1_output", a1_request->a1,
            nodes*a1_request->layout.num_lm*channels);
        m.encode_a1(batch, output, a1, nodes, a1_request->layout);
    }
    m.statistics.staging_seconds += seconds_since(staging);

    const double forward_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += forward_seconds;
    m.statistics.forward_seconds += forward_seconds;
    const auto readback = Clock::now();
    m.staging->complete();
    if (a1_request != nullptr) {
        m.zero_empty_a1_degrees(a1_request->a1, nodes, a1_request->layout);
        m.fused_a1 = {phi1, a1_request->a1, nodes};
        ++m.statistics.a1_forward_launches;
        ++m.statistics.fused_a1_launches;
    }
    if (phi1_staged)
        m.phi1 = {args.output, output_count};
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.forward_launches;
}

void MetalR1Module::reverse(
    const SymmetrixJitHostR1SourceArgsV2& source,
    const SymmetrixJitHostR1EdgeArgsV2& edge,
    const MetalR1ReverseExtents& extents)
{
    if (source.struct_size != sizeof(SymmetrixJitHostR1SourceArgsV2)
            || edge.struct_size != sizeof(SymmetrixJitHostR1EdgeArgsV2))
        fail("reverse packet size does not match");
    auto& m = *impl_;
    const MetalR1Shape& s = m.shape;
    const std::size_t sources = checked_u32(source.num_nodes, "source num_nodes");
    const std::size_t edges = checked_u32(edge.num_edges, "num_edges");
    if (static_cast<std::size_t>(source.num_edges) != edges)
        fail("source and edge packets disagree on the edge count");
    if (edges == 0)
        return;
    const std::size_t channels = static_cast<std::size_t>(s.channels);
    checked_u32(static_cast<std::int64_t>(sources*channels), "source grid");
    checked_u32(static_cast<std::int64_t>(edges*simd_width), "edge grid");

    const auto staging = Clock::now();
    auto batch = m.device.begin();
    m.staging->begin();
    Buffer& node_types = m.upload("node_types", source.node_types, extents.node_types);
    Buffer& neigh_types = m.upload("neigh_types", source.neigh_types, edges);
    Buffer& type_to_active =
        m.upload("type_to_active", source.type_to_active, extents.type_to_active);
    Buffer& source_offsets =
        m.upload("source_offsets", source.source_offsets, sources+1);
    Buffer& source_edges = m.upload("source_edges", source.source_edges, edges);
    Buffer& edge_receivers = m.upload("edge_receivers", source.edge_receivers, edges);
    Buffer& neigh_indices = m.upload("neigh_indices", edge.neigh_indices, edges);
    Buffer& radius = m.upload_narrowed("radius", source.radius, edges);
    const DeviceSpan harmonics = m.staging->input("harmonics_values",
        static_cast<const float*>(source.harmonics_values), edges*s.edge_harmonics);
    const std::size_t adjoint_elements =
        extents.receivers*s.output_components*channels;
    DeviceSpan output_adjoint;
    if (m.phi1_adjoint.matches(source.output_adjoint, adjoint_elements)) {
        output_adjoint = {&m.slot("output_adjoint", adjoint_elements*sizeof(float)), 0};
        ++m.statistics.resident_uploads_skipped;
    } else {
        output_adjoint = m.staging->input("output_adjoint",
            static_cast<const float*>(source.output_adjoint), adjoint_elements);
    }
    m.phi1_adjoint = {};
    m.phi1 = {};
    const DeviceSpan features = m.staging->input("neighbor_features",
        static_cast<const float*>(edge.neighbor_features),
        extents.neighbor_feature_nodes*s.source_harmonics*channels);
    // The source owner accumulates into the adjoint, so it starts from the
    // evaluator's current values.
    const std::size_t adjoint_count = sources*s.source_harmonics*channels;
    const DeviceSpan source_adjoint = m.staging->output("source_adjoint",
        static_cast<float*>(source.source_adjoint), adjoint_count, true);

    Buffer& unit_xyz = m.slot("unit_xyz", 3*edges*sizeof(float));
    float* unit = unit_xyz.data<float>();
    if (edge.coordinate_scalar_size != sizeof(float)
            && edge.coordinate_scalar_size != sizeof(double))
        fail("coordinates must be FP32 or FP64");
    if (edge.coordinate_scalar_size == sizeof(float) && edge.coordinates_are_unit == 0)
        fail("FP32 coordinates must be unit directions");
    if (edge.coordinates_are_unit != 0 && edge.coordinate_scalar_size == sizeof(float)) {
        std::memcpy(unit, edge.xyz, 3*edges*sizeof(float));
    } else if (edge.coordinates_are_unit != 0) {
        const auto* xyz = static_cast<const double*>(edge.xyz);
        for (std::size_t i = 0; i < 3*edges; ++i)
            unit[i] = static_cast<float>(xyz[i]);
    } else {
        const auto* xyz = static_cast<const double*>(edge.xyz);
        for (std::size_t e = 0; e < edges; ++e)
            for (std::size_t k = 0; k < 3; ++k)
                unit[3*e+k] = static_cast<float>(xyz[3*e+k]/edge.radius[e]);
    }
    const bool has_gradients = edge.harmonics_gradients != nullptr;
    Buffer& gradients = has_gradients
        ? m.upload("harmonics_gradients",
            static_cast<const float*>(edge.harmonics_gradients),
            3*edges*s.edge_harmonics)
        : m.slot("harmonics_gradients", 16);
    // Receiver-ordered edges let threadgroups share receiver adjoint rows;
    // otherwise one 32-lane SIMD group owns each edge.
    bool receiver_ordered = static_cast<bool>(m.edge_blocked);
    std::int32_t previous = 0;
    for (std::size_t e = 0; receiver_ordered && e < edges; ++e) {
        const std::int32_t receiver = source.edge_receivers[e];
        receiver_ordered = receiver >= previous
            && static_cast<std::size_t>(receiver) < extents.receivers;
        previous = receiver;
    }
    const std::size_t force_planes = receiver_ordered ? m.edge_blocks : 1;
    Buffer& forces = m.zeroed("directed_forces", force_planes*3*edges);

    const MetalR1RadialSpline radial = m.spline(source.radial, batch);
    const MetalR1SourceArgs source_packet{
        source.active_type_count,
        static_cast<std::uint32_t>(sources),
        static_cast<float>(source.cutoff),
        0,
        node_types.gpu_address(),
        neigh_types.gpu_address(),
        source_offsets.gpu_address(),
        source_edges.gpu_address(),
        edge_receivers.gpu_address(),
        type_to_active.gpu_address(),
        radius.gpu_address(),
        radial,
        harmonics.address(),
        output_adjoint.address(),
        source_adjoint.address()};
    const MetalR1EdgeArgs edge_packet{
        edge.active_type_count,
        static_cast<std::uint32_t>(edges),
        static_cast<float>(edge.cutoff),
        has_gradients ? 1u : 0u,
        node_types.gpu_address(),
        neigh_indices.gpu_address(),
        neigh_types.gpu_address(),
        edge_receivers.gpu_address(),
        type_to_active.gpu_address(),
        unit_xyz.gpu_address(),
        radius.gpu_address(),
        m.spline(edge.radial, batch),
        harmonics.address(),
        gradients.gpu_address(),
        features.address(),
        output_adjoint.address(),
        forces.gpu_address()};
    for (const Buffer* buffer : std::initializer_list<const Buffer*>{&node_types, &neigh_types, &type_to_active,
            &source_offsets, &source_edges, &edge_receivers, &neigh_indices,
            &radius, harmonics.buffer, output_adjoint.buffer, features.buffer,
            &unit_xyz, &gradients})
        batch.use_buffer(*buffer, false);
    batch.use_buffer(*source_adjoint.buffer, true).use_buffer(forces, true);
    if (sources != 0)
        batch.set_value(0, source_packet)
            .dispatch_threads(m.source, {sources*channels}, {threadgroup_size});
    if (profiling_enabled()) {
        const double source_seconds = batch.submit_and_wait().gpu_seconds;
        m.statistics.gpu_seconds += source_seconds;
        m.statistics.source_seconds += source_seconds;
        batch = m.device.begin();
        for (const Buffer* buffer : std::initializer_list<const Buffer*>{&node_types, &neigh_types, &type_to_active,
                &edge_receivers, &neigh_indices, &radius, harmonics.buffer,
                output_adjoint.buffer, features.buffer, &unit_xyz, &gradients})
            batch.use_buffer(*buffer, false);
        batch.use_buffer(forces, true);
        m.spline(edge.radial, batch);
    }
    if (receiver_ordered) {
        const std::size_t receiver_count = static_cast<std::size_t>(previous)+1;
        Buffer& offsets = m.slot("receiver_offsets",
            (receiver_count+1)*sizeof(std::int32_t));
        auto* offset = offsets.data<std::int32_t>();
        std::fill(offset, offset+receiver_count+1, 0);
        for (std::size_t e = 0; e < edges; ++e)
            ++offset[source.edge_receivers[e]+1];
        for (std::size_t r = 0; r < receiver_count; ++r)
            offset[r+1] += offset[r];
        batch.set_value(0, edge_packet).set_buffer(1, offsets).set_buffer(2, forces)
            .dispatch_threadgroups(m.edge_blocked,
                {receiver_count*m.edge_blocks}, {threadgroup_size});
        ++m.statistics.blocked_edge_launches;
    } else {
        batch.set_value(0, edge_packet)
            .dispatch_threads(m.edge, {edges*simd_width}, {threadgroup_size});
    }
    m.statistics.staging_seconds += seconds_since(staging);

    const double edge_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += edge_seconds;
    if (profiling_enabled())
        m.statistics.edge_seconds += edge_seconds;
    const auto readback = Clock::now();
    m.staging->complete();
    // The edge kernel subtracts from zero; the evaluator's directed forces
    // receive the same contribution in double precision.
    const float* force_values = forces.data<float>();
    if (force_planes == 1)
        for (std::size_t i = 0; i < 3*edges; ++i)
            edge.directed_forces[i] += static_cast<double>(force_values[i]);
    else
        for (std::size_t i = 0; i < 3*edges; ++i) {
            double sum = 0.0;
            for (std::size_t plane = 0; plane < force_planes; ++plane)
                sum += static_cast<double>(force_values[plane*3*edges+i]);
            edge.directed_forces[i] += sum;
        }
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.reverse_launches;
}

// For degree l and row m, the Phi1 rows of all nodes form a strided matrix
// with one row per node, so A1 is 2l+1 GEMMs per degree over every node.
bool MetalR1Module::a1_forward(
    const float* phi1, const std::size_t num_nodes,
    const MetalA1Layout& layout, float* a1)
{
    auto& m = *impl_;
    Impl::validate(layout, *m.staging);
    const std::size_t channels = static_cast<std::size_t>(m.shape.channels);
    if (num_nodes != 0 && m.fused_a1.phi1 == phi1 && m.fused_a1.a1 == a1
            && m.fused_a1.nodes == num_nodes) {
        m.fused_a1 = {};
        return true;
    }
    m.fused_a1 = {};
    const std::size_t phi1_count = num_nodes*layout.num_lme*channels;
    const bool phi1_resident = m.phi1.matches(phi1, phi1_count);
    if (num_nodes == 0
            || !(phi1_resident
                || m.staging->is_mapped(phi1, phi1_count*sizeof(float))))
        return false;
    checked_u32(static_cast<std::int64_t>(num_nodes), "A1 node count");
    auto batch = m.device.begin();
    m.staging->begin();
    const DeviceSpan input = phi1_resident
        ? DeviceSpan{&m.slot("forward_output", phi1_count*sizeof(float)), 0}
        : m.staging->input("phi1", phi1, phi1_count);
    const DeviceSpan output = m.staging->output(
        "a1_output", a1, num_nodes*layout.num_lm*channels);
    m.encode_a1(batch, input, output, num_nodes, layout);
    const double a1_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += a1_seconds;
    m.statistics.a1_seconds += a1_seconds;
    const auto readback = Clock::now();
    m.staging->complete();
    m.zero_empty_a1_degrees(a1, num_nodes, layout);
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.a1_forward_launches;
    return true;
}

void MetalR1Module::a1_reverse(
    const std::size_t num_nodes, const std::size_t capacity_nodes,
    const MetalA1Layout& layout,
    const float* a1_adjoint, float* phi1_adjoint)
{
    auto& m = *impl_;
    Impl::validate(layout, *m.staging);
    if (num_nodes == 0)
        return;
    if (capacity_nodes < num_nodes)
        m.staging->fail("dPhi1 capacity is smaller than the node count");
    checked_u32(static_cast<std::int64_t>(num_nodes), "A1 node count");
    const std::size_t channels = static_cast<std::size_t>(m.shape.channels);
    const auto rows = static_cast<std::uint32_t>(num_nodes);
    const auto c = static_cast<std::uint32_t>(channels);
    const auto staging = Clock::now();
    auto batch = m.device.begin();
    m.staging->begin();
    const DeviceSpan input = m.staging->input("a1_adjoint", a1_adjoint,
        num_nodes*layout.num_lm*channels);
    const std::size_t output_count = num_nodes*layout.num_lme*channels;
    const std::size_t capacity_count = capacity_nodes*layout.num_lme*channels;
    // Mapped dPhi1 is written in place. Otherwise it is written into the R1
    // reverse input slot, sized as the reverse stages it, so the next
    // reverse reuses it without a reallocation or upload.
    const bool phi1_adjoint_mapped =
        m.staging->is_mapped(phi1_adjoint, output_count*sizeof(float));
    const DeviceSpan output = phi1_adjoint_mapped
        ? m.staging->output("output_adjoint", phi1_adjoint, output_count)
        : DeviceSpan{&m.slot("output_adjoint", capacity_count*sizeof(float)), 0};
    const std::size_t input_row = layout.num_lm*channels*sizeof(float);
    const std::size_t output_row = layout.num_lme*channels*sizeof(float);
    for (int l = 0; l <= layout.l_max; ++l) {
        const auto inputs = static_cast<std::uint32_t>(layout.eta[l]*channels);
        if (inputs == 0)
            continue;
        const Buffer& weights_trans = m.staging->persistent(
            layout.weights_trans[l],
            static_cast<std::size_t>(inputs)*channels*sizeof(float));
        for (int row = 0; row < 2*l+1; ++row)
            batch.gemm(
                {input.buffer, input.offset
                    +(static_cast<std::size_t>(l*l+row)*channels)*sizeof(float),
                    rows, c, input_row},
                {&weights_trans, 0, c, inputs, inputs*sizeof(float)},
                {output.buffer, output.offset
                    +(layout.lme[l]*channels+static_cast<std::size_t>(row)*inputs)
                    *sizeof(float), rows, inputs, output_row});
    }
    m.statistics.staging_seconds += seconds_since(staging);
    const double a1_seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += a1_seconds;
    m.statistics.a1_seconds += a1_seconds;
    const auto readback = Clock::now();
    m.staging->complete();
    if (phi1_adjoint_mapped) {
        m.phi1_adjoint = {};
    } else {
        std::memcpy(phi1_adjoint, output.buffer->contents(), output_count*sizeof(float));
        m.phi1_adjoint = {phi1_adjoint, capacity_count};
    }
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.a1_reverse_launches;
}

bool MetalR1Module::scale_node_rows(
    float* rows, const std::size_t num_nodes, const std::size_t row_length,
    const float* factors)
{
    auto& m = *impl_;
    const std::size_t count = num_nodes*row_length;
    if (count == 0 || !m.staging->is_mapped(rows, count*sizeof(float)))
        return false;
    checked_u32(static_cast<std::int64_t>(count), "row scale grid");
    const auto staging = Clock::now();
    auto batch = m.device.begin();
    m.staging->begin();
    const DeviceSpan values = m.staging->output("scaled_rows", rows, count, true);
    Buffer& scale = m.staging->upload("row_factors", factors, num_nodes);
    const RowScaleArgs args{static_cast<std::uint32_t>(num_nodes),
        static_cast<std::uint32_t>(row_length), values.address(), scale.gpu_address()};
    batch.use_buffer(*values.buffer, true).use_buffer(scale, false)
        .set_value(0, args)
        .dispatch_threads(m.row_scale, {count}, {threadgroup_size});
    m.statistics.staging_seconds += seconds_since(staging);
    const double seconds = batch.submit_and_wait().gpu_seconds;
    m.statistics.gpu_seconds += seconds;
    m.statistics.a1_seconds += seconds;
    m.staging->complete();
    ++m.statistics.row_scale_launches;
    return true;
}

}  // namespace symmetrix::execution::metal
