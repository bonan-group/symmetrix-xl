#include "metal_m0_module.hpp"

#include <chrono>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

#include "metal_runtime.hpp"
#include "metal_staging.hpp"

namespace symmetrix::execution::metal {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t threadgroup_size = 128;
constexpr std::uint64_t simd_width = 32;

struct M0Args {
    std::uint32_t num_nodes;
    std::uint32_t capture_input_scale_adjoint;
    std::uint32_t padded_channels;
    std::uint32_t reserved;
    std::uint64_t node_types;
    std::uint64_t input;
    std::uint64_t weights;
    std::uint64_t output_adjoint;
    std::uint64_t output;
    std::uint64_t input_adjoint;
    std::uint64_t scale_partials;
};
static_assert(sizeof(M0Args) == 72);

double seconds_since(const Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now()-start).count();
}

}  // namespace

struct MetalM0Module::Impl {
    Device device;
    Pipeline forward;
    Pipeline reverse;
    MetalM0Shape shape;
    std::unique_ptr<MetalStaging> staging;
    MetalM0Statistics statistics;
    struct FusedLinear {
        const float* output = nullptr;
        const float* linear_output = nullptr;
        std::size_t nodes = 0;
    } fused_linear;

    std::size_t padded_channels() const
    {
        const std::size_t channels = static_cast<std::size_t>(shape.channels);
        return (channels+simd_width-1)/simd_width*simd_width;
    }

    std::size_t checked_nodes(const std::int64_t num_nodes) const
    {
        if (num_nodes < 0
                || static_cast<std::uint64_t>(num_nodes)*padded_channels() > UINT32_MAX)
            staging->fail("node count is outside the 32-bit Metal grid range");
        return static_cast<std::size_t>(num_nodes);
    }

    // Weights are evaluator-owned parameters, fixed for the model lifetime.
    const Buffer& weights(const float* host, const std::size_t count)
    {
        const std::size_t per_type =
            static_cast<std::size_t>(shape.term_count)*shape.channels;
        if (count == 0 || count % per_type != 0)
            staging->fail("weights are not a whole number of [term, channel] tables");
        return staging->persistent(host, count*sizeof(float));
    }
};

MetalM0Module::MetalM0Module(
    const std::string_view msl_source, const MetalM0Shape& shape)
    : impl_(std::make_unique<Impl>())
{
    if (shape.channels <= 0 || shape.input_components <= 0
            || shape.output_components <= 0 || shape.term_count <= 0)
        throw std::runtime_error("Metal M0: module shape must be positive");
    impl_->shape = shape;
    impl_->device = Device::system_default();
    impl_->staging = std::make_unique<MetalStaging>(impl_->device, "M0");
    const Library library = impl_->device.compile(msl_source, module_compile_options());
    impl_->forward = impl_->device.pipeline(library, "symmetrix_m0_forward");
    impl_->reverse = impl_->device.pipeline(library, "symmetrix_m0_reverse");
    if (impl_->reverse.thread_execution_width() != simd_width)
        impl_->staging->fail("the reverse kernel requires 32-wide SIMD groups");
    for (const Pipeline* pipeline : {&impl_->forward, &impl_->reverse})
        if (pipeline->max_total_threads_per_threadgroup() < threadgroup_size)
            impl_->staging->fail("pipeline '"+pipeline->function_name()
                +"' cannot run the required threadgroup size");
}

MetalM0Module::~MetalM0Module() = default;

const MetalM0Shape& MetalM0Module::shape() const { return impl_->shape; }

const MetalM0Statistics& MetalM0Module::statistics() const
{
    return impl_->statistics;
}

void MetalM0Module::discard_pending_results()
{
    impl_->fused_linear = {};
}

void MetalM0Module::release_persistent_tables()
{
    impl_->staging->release_tables();
}

void MetalM0Module::set_host_memory(std::shared_ptr<const HostMemoryMap> host_memory)
{
    impl_->staging->set_host_memory(std::move(host_memory));
}

void MetalM0Module::forward(
    const std::int64_t num_nodes,
    const std::int32_t* node_types,
    const float* input,
    const float* weights,
    const std::size_t weight_count,
    float* output,
    const MetalM0LinearRequest* linear)
{
    auto& m = *impl_;
    m.fused_linear = {};
    const std::size_t nodes = m.checked_nodes(num_nodes);
    if (nodes == 0)
        return;
    const std::size_t channels = static_cast<std::size_t>(m.shape.channels);
    const auto staging = Clock::now();
    auto batch = m.device.begin();
    m.staging->begin();
    const DeviceSpan types = m.staging->input("node_types", node_types, nodes);
    const DeviceSpan x = m.staging->input("input", input,
        nodes*m.shape.input_components*channels);
    const Buffer& w = m.weights(weights, weight_count);
    const DeviceSpan out = m.staging->output("output", output,
        nodes*m.shape.output_components*channels);
    const M0Args args{
        static_cast<std::uint32_t>(nodes), 0u,
        static_cast<std::uint32_t>(m.padded_channels()), 0u,
        types.address(), x.address(), w.gpu_address(), 0,
        out.address(), 0, 0};
    for (const Buffer* buffer :
            std::initializer_list<const Buffer*>{types.buffer, x.buffer, &w})
        batch.use_buffer(*buffer, false);
    batch.use_buffer(*out.buffer, true).set_value(0, args)
        .dispatch_threads(m.forward, {nodes*m.padded_channels()}, {threadgroup_size});
    const std::size_t components = static_cast<std::size_t>(m.shape.output_components);
    if (linear != nullptr) {
        if (linear->l_max < 0 || linear->weights == nullptr || linear->output == nullptr
                || static_cast<std::size_t>((linear->l_max+1)*(linear->l_max+1)) != components)
            m.staging->fail("linear request does not match the output components");
        if (nodes > UINT32_MAX)
            m.staging->fail("linear request exceeds the 32-bit GEMM row range");
        const DeviceSpan mixed = m.staging->output(
            "linear_output", linear->output, nodes*components*channels);
        const auto rows = static_cast<std::uint32_t>(nodes);
        const auto c = static_cast<std::uint32_t>(channels);
        const std::size_t row_bytes = components*channels*sizeof(float);
        // Each lm row of every node is one strided GEMM with its degree's weights.
        for (int l = 0; l <= linear->l_max; ++l) {
            const Buffer& w_l = m.staging->persistent(
                linear->weights+static_cast<std::size_t>(l)*channels*channels,
                channels*channels*sizeof(float));
            for (int lm = l*l; lm < (l+1)*(l+1); ++lm) {
                const std::size_t column = static_cast<std::size_t>(lm)*channels*sizeof(float);
                batch.gemm({out.buffer, out.offset+column, rows, c, row_bytes},
                    {&w_l, 0, c, c, channels*sizeof(float)},
                    {mixed.buffer, mixed.offset+column, rows, c, row_bytes});
            }
        }
    }
    m.statistics.staging_seconds += seconds_since(staging);
    m.statistics.gpu_seconds += batch.submit_and_wait().gpu_seconds;
    const auto readback = Clock::now();
    m.staging->complete();
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.forward_launches;
    if (linear != nullptr) {
        m.fused_linear = {output, linear->output, nodes};
        ++m.statistics.fused_linear_launches;
    }
}

bool MetalM0Module::take_fused_linear(
    const float* output, const float* linear_output, const std::int64_t num_nodes)
{
    auto& m = *impl_;
    const bool ready = num_nodes > 0 && m.fused_linear.output == output
        && m.fused_linear.linear_output == linear_output
        && m.fused_linear.nodes == static_cast<std::size_t>(num_nodes);
    m.fused_linear = {};
    return ready;
}

void MetalM0Module::reverse(
    const std::int64_t num_nodes,
    const std::int32_t* node_types,
    const float* input,
    const float* weights,
    const std::size_t weight_count,
    const float* output_adjoint,
    float* input_adjoint,
    double* input_scale_adjoint,
    const bool capture_input_scale_adjoint)
{
    auto& m = *impl_;
    const std::size_t nodes = m.checked_nodes(num_nodes);
    if (nodes == 0)
        return;
    if (capture_input_scale_adjoint && input_scale_adjoint == nullptr)
        m.staging->fail("input-scale adjoint capture requires an output");
    const std::size_t channels = static_cast<std::size_t>(m.shape.channels);
    const std::size_t blocks = m.padded_channels()/simd_width;
    const auto staging = Clock::now();
    auto batch = m.device.begin();
    m.staging->begin();
    const DeviceSpan types = m.staging->input("node_types", node_types, nodes);
    const DeviceSpan x = m.staging->input("input", input,
        nodes*m.shape.input_components*channels);
    const Buffer& w = m.weights(weights, weight_count);
    const DeviceSpan adjoint = m.staging->input("output_adjoint", output_adjoint,
        nodes*m.shape.output_components*channels);
    const DeviceSpan gradient = m.staging->output("input_adjoint", input_adjoint,
        nodes*m.shape.input_components*channels);
    Buffer& partials = m.staging->slot("scale_partials", nodes*blocks*sizeof(float));
    const M0Args args{
        static_cast<std::uint32_t>(nodes),
        capture_input_scale_adjoint ? 1u : 0u,
        static_cast<std::uint32_t>(m.padded_channels()), 0u,
        types.address(), x.address(), w.gpu_address(),
        adjoint.address(), 0, gradient.address(), partials.gpu_address()};
    for (const Buffer* buffer : std::initializer_list<const Buffer*>{
            types.buffer, x.buffer, &w, adjoint.buffer})
        batch.use_buffer(*buffer, false);
    batch.use_buffer(*gradient.buffer, true).use_buffer(partials, true)
        .set_value(0, args)
        .dispatch_threads(m.reverse, {nodes*m.padded_channels()}, {threadgroup_size});
    m.statistics.staging_seconds += seconds_since(staging);
    m.statistics.gpu_seconds += batch.submit_and_wait().gpu_seconds;
    const auto readback = Clock::now();
    m.staging->complete();
    if (capture_input_scale_adjoint) {
        const float* values = partials.data<float>();
        for (std::size_t node = 0; node < nodes; ++node) {
            double sum = 0.0;
            for (std::size_t block = 0; block < blocks; ++block)
                sum += static_cast<double>(values[node*blocks+block]);
            input_scale_adjoint[node] += sum;
        }
    }
    m.statistics.staging_seconds += seconds_since(readback);
    ++m.statistics.reverse_launches;
}

}  // namespace symmetrix::execution::metal
