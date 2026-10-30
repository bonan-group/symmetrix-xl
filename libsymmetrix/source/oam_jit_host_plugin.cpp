#include "oam_jit_host_plugin.hpp"
#include "jit_device_plugin_common.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(std::is_standard_layout_v<SymmetrixJitOAMHostForwardArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitOAMHostReverseArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitOAMHostLayerV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitOAMHostPluginV1>);
static_assert(
    std::is_standard_layout_v<SymmetrixJitOAMHostReceiverForwardArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitOAMHostSourceReverseArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitOAMHostLayerV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitOAMHostPluginV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitOAMHostReceiverReverseArgsV4>);
static_assert(std::is_standard_layout_v<SymmetrixJitOAMHostPluginV6>);
static_assert(std::is_standard_layout_v<SymmetrixJitOAMHostPluginV7>);

namespace symmetrix::execution {

#if defined(__unix__) || defined(__APPLE__)
namespace {

std::runtime_error plugin_error(
    const std::string& path, const std::string& message)
{
    return std::runtime_error(
        "Could not load Execution OAM host plugin '"+path+"': "+message);
}

void require_extent(
    const std::string& path, const std::int32_t expected,
    const std::int32_t actual, const char* field, const std::size_t layer)
{
    if(expected>=0&&expected!=actual)
        throw plugin_error(
            path,"layer "+std::to_string(layer)+" "+field
                +" does not match");
}

template<class Expectation>
detail::ArtifactEnvelopeExpectation envelope_expectation(
    const Expectation& expectation,
    const std::uint32_t abi_version,
    const std::size_t struct_size,
    const std::string_view abi_tag)
{
    detail::ArtifactEnvelopeExpectation identity;
    identity.abi_version=abi_version;
    identity.struct_size=struct_size;
    identity.byte_order=SYMMETRIX_JIT_OAM_HOST_PLUGIN_BYTE_ORDER;
    identity.abi_tag=abi_tag;
    identity.artifact_id=expectation.artifact_id;
    identity.identity_fingerprint=expectation.generation_fingerprint;
    identity.semantic_fingerprint=expectation.semantic_fingerprint;
    identity.structure_fingerprint=expectation.structure_fingerprint;
    identity.identity_label="generation fingerprint";
    return identity;
}

template<class Descriptor>
detail::ArtifactEnvelope envelope(const Descriptor& descriptor)
{
    return detail::ArtifactEnvelope{
        .abi_version=descriptor.abi_version,
        .struct_size=descriptor.struct_size,
        .pointer_size=descriptor.pointer_size,
        .byte_order=descriptor.byte_order,
        .checks_reserved=true,
        .reserved=descriptor.reserved,
        .abi_tag=descriptor.abi_tag,
        .artifact_id=descriptor.artifact_id,
        .identity_fingerprint=descriptor.generation_fingerprint,
        .semantic_fingerprint=descriptor.semantic_fingerprint,
        .structure_fingerprint=descriptor.structure_fingerprint,
    };
}

void validate_layer_expectation(const OAMHostLayerExpectation& layer)
{
    if(layer.source_dimension<=0||layer.edge_dimension<=0
        ||layer.weight_dimension<=0||layer.output_dimension<=0
        ||layer.instruction_count<=0)
        throw std::invalid_argument(
            "Execution OAM host plugin expectation requires positive layer "
            "dimensions and instruction counts.");
}

void validate_expectation(const OAMHostPluginExpectation& expectation)
{
    if(expectation.artifact_id.empty()
        ||expectation.generation_fingerprint.empty()
        ||expectation.semantic_fingerprint.empty()
        ||expectation.structure_fingerprint.empty())
        throw std::invalid_argument(
            "Execution OAM host plugin expectation requires complete artifact "
            "identity metadata.");
    if(expectation.scalar_size!=sizeof(float)
        &&expectation.scalar_size!=sizeof(double))
        throw std::invalid_argument(
            "Execution OAM host plugin expectation requires float32 or "
            "float64 scalar size.");
    if(expectation.layers.empty())
        throw std::invalid_argument(
            "Execution OAM host plugin expectation requires at least one "
            "layer.");
    constexpr std::uint32_t known=
        SYMMETRIX_JIT_OAM_HOST_FORWARD_OWNER_V1
        |SYMMETRIX_JIT_OAM_HOST_REVERSE_OWNER_V1;
    if(expectation.required_capabilities==0
        ||(expectation.required_capabilities&~known)!=0)
        throw std::invalid_argument(
            "Execution OAM host plugin expectation requires known "
            "capabilities.");
    for(const auto& layer:expectation.layers)
        validate_layer_expectation(layer);
}

void validate_layer(
    const std::string& path,
    const SymmetrixJitOAMHostLayerV1& actual,
    const OAMHostLayerExpectation& expected,
    const std::size_t index)
{
    if(actual.struct_size<sizeof(SymmetrixJitOAMHostLayerV1))
        throw plugin_error(
            path,"layer "+std::to_string(index)
                +" descriptor is smaller than ABI version 1");
    if(actual.layer!=index)
        throw plugin_error(
            path,"layer "+std::to_string(index)+" layer index does not match");
    if(actual.reserved!=0||actual.reserved_2!=0)
        throw plugin_error(
            path,"layer "+std::to_string(index)
                +" reserved field is nonzero");
    if(actual.source_dimension<=0||actual.edge_dimension<=0
        ||actual.weight_dimension<=0||actual.output_dimension<=0
        ||actual.instruction_count<=0)
        throw plugin_error(
            path,"layer "+std::to_string(index)+" dimensions are invalid");
    require_extent(path,expected.source_dimension,actual.source_dimension,
        "source dimension",index);
    require_extent(path,expected.edge_dimension,actual.edge_dimension,
        "edge dimension",index);
    require_extent(path,expected.weight_dimension,actual.weight_dimension,
        "weight dimension",index);
    require_extent(path,expected.output_dimension,actual.output_dimension,
        "output dimension",index);
    require_extent(path,expected.instruction_count,actual.instruction_count,
        "instruction count",index);
}

void validate_descriptor(
    const std::string& path,
    const SymmetrixJitOAMHostPluginV1& descriptor,
    const OAMHostPluginExpectation& expectation)
{
    const detail::PluginLoaderContext context("OAM host",path);
    detail::validate_artifact_envelope(
        context,envelope(descriptor),
        envelope_expectation(
            expectation,SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION,
            sizeof(SymmetrixJitOAMHostPluginV1),
            SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG));
    if(descriptor.scalar_size!=expectation.scalar_size)
        throw context.error("scalar size does not match");

    constexpr std::uint32_t known=
        SYMMETRIX_JIT_OAM_HOST_FORWARD_OWNER_V1
        |SYMMETRIX_JIT_OAM_HOST_REVERSE_OWNER_V1;
    if((descriptor.capabilities&~known)!=0)
        throw context.error("unknown capability bits are set");
    if((descriptor.capabilities&expectation.required_capabilities)
        !=expectation.required_capabilities)
        throw context.error("required capabilities are missing");
    const bool has_forward=(descriptor.capabilities
        &SYMMETRIX_JIT_OAM_HOST_FORWARD_OWNER_V1)!=0;
    const bool has_reverse=(descriptor.capabilities
        &SYMMETRIX_JIT_OAM_HOST_REVERSE_OWNER_V1)!=0;
    if(has_forward!= (descriptor.forward_owners!=nullptr))
        throw plugin_error(path,"forward owner capability and array disagree");
    if(has_reverse!= (descriptor.reverse_owners!=nullptr))
        throw plugin_error(path,"reverse owner capability and array disagree");
    if(descriptor.layer_count!=expectation.layers.size())
        throw plugin_error(path,"layer count does not match");
    if(descriptor.layer_count==0)
        throw plugin_error(path,"layer count is zero");
    if(descriptor.layers==nullptr)
        throw plugin_error(path,"layer metadata array is null");
    if(has_forward&&descriptor.forward_owners==nullptr)
        throw plugin_error(path,"forward owner array is null");
    if(has_reverse&&descriptor.reverse_owners==nullptr)
        throw plugin_error(path,"reverse owner array is null");
    for(std::size_t layer=0;layer<expectation.layers.size();++layer) {
        validate_layer(path,descriptor.layers[layer],expectation.layers[layer],layer);
        if(has_forward&&descriptor.forward_owners[layer]==nullptr)
            throw plugin_error(path,"layer "+std::to_string(layer)
                +" forward owner is null");
        if(has_reverse&&descriptor.reverse_owners[layer]==nullptr)
            throw plugin_error(path,"layer "+std::to_string(layer)
                +" reverse owner is null");
    }
}

void require_positive(
    const std::string& path, const std::int32_t value, const char* field,
    const std::size_t layer)
{
    if(value<=0)
        throw plugin_error(
            path,"layer "+std::to_string(layer)+" "+field+" is invalid");
}

void require_channel_tile(
    const std::string& path, const std::int32_t value, const std::size_t layer)
{
    if(value<1)
        throw plugin_error(
            path,"layer "+std::to_string(layer)+" channel tile is invalid");
}

void validate_layer_expectation_v2(const OAMHostLayerExpectationV2& layer)
{
    // channel_tile == -1 means "no expectation": the runtime does not depend on
    // the generated tile width, and the descriptor still requires a positive
    // value.
    if(layer.source_dimension<=0||layer.edge_dimension<=0
        ||layer.weight_dimension<=0||layer.output_dimension<=0
        ||layer.instruction_count<=0||layer.channel_tile<-1
        ||layer.channel_tile==0
        ||layer.target_dimension<=0)
        throw std::invalid_argument(
            "Execution OAM host plugin expectation requires positive layer "
            "dimensions, instruction counts, and channel tiles.");
}

void validate_expectation_v2(const OAMHostPluginExpectationV2& expectation)
{
    if(expectation.artifact_id.empty()
        ||expectation.generation_fingerprint.empty()
        ||expectation.semantic_fingerprint.empty()
        ||expectation.structure_fingerprint.empty())
        throw std::invalid_argument(
            "Execution OAM host plugin expectation requires complete artifact "
            "identity metadata.");
    if(expectation.scalar_size!=sizeof(float)
        &&expectation.scalar_size!=sizeof(double))
        throw std::invalid_argument(
            "Execution OAM host plugin expectation requires float32 or "
            "float64 scalar size.");
    if(expectation.layers.empty())
        throw std::invalid_argument(
            "Execution OAM host plugin expectation requires at least one "
            "layer.");
    constexpr std::uint32_t known=
        SYMMETRIX_JIT_OAM_HOST_FULL_FORWARD_V3
        |SYMMETRIX_JIT_OAM_HOST_FULL_REVERSE_V3
        |SYMMETRIX_JIT_OAM_HOST_LANE_MAJOR_V3;
    if(expectation.required_capabilities==0
        ||(expectation.required_capabilities&~known)!=0)
        throw std::invalid_argument(
            "Execution OAM host plugin expectation requires known "
            "capabilities.");
    if(expectation.schedule_identity.empty()
        ||expectation.host_target.empty()
        ||expectation.full_reverse_owner_schedule.empty()
        ||expectation.full_reverse_reduction_policy.empty())
        throw std::invalid_argument(
            "Execution OAM ABI v7 expectation requires complete schedule and "
            "active-host target metadata; "
            "regenerate the OAM artifact.");
    for(const auto& layer:expectation.layers)
        validate_layer_expectation_v2(layer);
}

void validate_layer_v2(
    const std::string& path,
    const SymmetrixJitOAMHostLayerV4& actual,
    const OAMHostLayerExpectationV2& expected,
    const std::size_t index)
{
    if(actual.struct_size<sizeof(SymmetrixJitOAMHostLayerV4))
        throw plugin_error(
            path,"layer "+std::to_string(index)
                +" descriptor is smaller than ABI version 7; regenerate the OAM artifact");
    if(actual.layer!=index)
        throw plugin_error(
            path,"layer "+std::to_string(index)+" layer index does not match");
    if(actual.reserved!=0||actual.variant_mask!=1u)
        throw plugin_error(
            path,"layer "+std::to_string(index)
                +" reserved field is nonzero");
    require_positive(path,actual.source_dimension,"source dimension",index);
    require_positive(path,actual.edge_dimension,"edge dimension",index);
    require_positive(path,actual.weight_dimension,"weight dimension",index);
    require_positive(path,actual.output_dimension,"output dimension",index);
    require_positive(path,actual.instruction_count,"instruction count",index);
    require_positive(path,actual.target_dimension,"target dimension",index);
    require_channel_tile(path,actual.channel_tile,index);
    require_extent(path,expected.source_dimension,actual.source_dimension,
        "source dimension",index);
    require_extent(path,expected.edge_dimension,actual.edge_dimension,
        "edge dimension",index);
    require_extent(path,expected.weight_dimension,actual.weight_dimension,
        "weight dimension",index);
    require_extent(path,expected.output_dimension,actual.output_dimension,
        "output dimension",index);
    require_extent(path,expected.instruction_count,actual.instruction_count,
        "instruction count",index);
    require_extent(path,expected.channel_tile,actual.channel_tile,
        "channel tile",index);
    require_extent(path,expected.target_dimension,actual.target_dimension,
        "target dimension",index);
    if(actual.projected_dimension!=0||actual.projection_weight_count!=0
        ||actual.projected_partial_workspace_bytes!=0)
        throw plugin_error(path,"layer "+std::to_string(index)
            +" legacy projected fields are nonzero");
}

void validate_descriptor_v2(
    const std::string& path,
    const SymmetrixJitOAMHostPluginV7& descriptor,
    const OAMHostPluginExpectationV2& expectation)
{
    const detail::PluginLoaderContext context("OAM host",path);
    detail::validate_artifact_envelope(
        context,envelope(descriptor),
        envelope_expectation(
            expectation,SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION_V7,
            sizeof(SymmetrixJitOAMHostPluginV7),
            SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG_V7));
    if(descriptor.scalar_size!=expectation.scalar_size)
        throw context.error("scalar size does not match");

    constexpr std::uint32_t known=
        SYMMETRIX_JIT_OAM_HOST_FULL_FORWARD_V3
        |SYMMETRIX_JIT_OAM_HOST_FULL_REVERSE_V3
        |SYMMETRIX_JIT_OAM_HOST_LANE_MAJOR_V3;
    if((descriptor.capabilities&~known)!=0)
        throw context.error("unknown capability bits are set");
    if((descriptor.capabilities&expectation.required_capabilities)
        !=expectation.required_capabilities)
        throw context.error("required capabilities are missing");
    const bool has_receiver=(descriptor.capabilities
        &SYMMETRIX_JIT_OAM_HOST_FULL_FORWARD_V3)!=0;
    const bool has_source=(descriptor.capabilities
        &SYMMETRIX_JIT_OAM_HOST_FULL_REVERSE_V3)!=0;
    if(has_receiver!= (descriptor.full_forward_owners!=nullptr))
        throw plugin_error(path,
            "receiver forward owner capability and array disagree");
    if(has_source!= (descriptor.full_reverse_owners!=nullptr))
        throw plugin_error(path,
            "source reverse owner capability and array disagree");
    if(descriptor.layer_count!=expectation.layers.size())
        throw plugin_error(path,"layer count does not match");
    if(descriptor.layer_count==0)
        throw plugin_error(path,"layer count is zero");
    if(descriptor.layers==nullptr)
        throw plugin_error(path,"layer metadata array is null");
    if(has_receiver&&descriptor.full_forward_owners==nullptr)
        throw plugin_error(path,"receiver forward owner array is null");
    if(has_source&&descriptor.full_reverse_owners==nullptr)
        throw plugin_error(path,"source reverse owner array is null");
    for(std::size_t layer=0;layer<expectation.layers.size();++layer) {
        validate_layer_v2(
            path,descriptor.layers[layer],expectation.layers[layer],layer);
        if(has_receiver&&descriptor.full_forward_owners[layer]==nullptr)
            throw plugin_error(path,"layer "+std::to_string(layer)
                +" receiver forward owner is null");
        if(has_source&&descriptor.full_reverse_owners[layer]==nullptr)
            throw plugin_error(path,"layer "+std::to_string(layer)
                +" source reverse owner is null");
    }
    if(descriptor.projected_forward_owners!=nullptr
        ||descriptor.projected_reverse_owners!=nullptr
        ||descriptor.projection_layout!=nullptr
        ||descriptor.projected_reverse_owner_schedule!=nullptr
        ||descriptor.projected_reverse_reduction_policy!=nullptr
        ||descriptor.projection_mode!=nullptr)
        throw context.error(
            "legacy projected descriptor fields are populated; regenerate the OAM artifact");
    context.require_match(expectation.schedule_identity,
        context.require_text(descriptor.schedule_identity,"schedule identity").c_str(),
        "schedule identity");
    context.require_match(expectation.host_target,
        context.require_text(descriptor.host_target,"host target").c_str(),
        "host target; the artifact may have been relocated to an incompatible CPU, regenerate the OAM artifact");
    const auto forward_schedule=context.require_text(
        descriptor.forward_owner_schedule,"forward owner schedule");
    const auto full_reverse_schedule=context.require_text(
        descriptor.full_reverse_owner_schedule,"full reverse owner schedule");
    const auto tiling=context.require_text(descriptor.tiling_policy,"tiling policy");
    const bool whole=forward_schedule=="receiver-owner-v1"
        &&full_reverse_schedule=="source-owner-v1"
        &&tiling=="whole-owner-v1";
    const bool tiled=forward_schedule=="receiver-channel-tiled-owner-v1"
        &&full_reverse_schedule=="source-channel-tiled-owner-v1"
        &&tiling=="channel-tiled-owner-v1";
    if(!whole&&!tiled)
        throw context.error(
            "host owner and tiling policies are inconsistent; regenerate the OAM artifact");
    context.require_match(expectation.forward_owner_schedule,forward_schedule.c_str(),
        "forward owner schedule");
    context.require_match(expectation.full_reverse_owner_schedule,
        full_reverse_schedule.c_str(),"full reverse owner schedule");
    context.require_match(expectation.tiling_policy,tiling.c_str(),"tiling policy");
    context.require_match(expectation.vector_policy,
        context.require_text(descriptor.vector_policy,"vector policy").c_str(),
        "vector policy");
    context.require_match("none",
        context.require_text(descriptor.shared_memory_policy,"shared-memory policy").c_str(),
        "shared-memory policy; regenerate the OAM artifact");
    context.require_match(expectation.register_budget_policy,
        context.require_text(descriptor.register_budget_policy,"register-budget policy").c_str(),
        "register-budget policy");
    context.require_match("evaluator-harmonic-gradient-reduction-v1",
        context.require_text(descriptor.direct_geometry_policy,"direct-geometry policy").c_str(),
        "direct-geometry policy; regenerate the OAM artifact");
    if(descriptor.scalar_width!=expectation.scalar_width
        ||descriptor.scalar_width!=descriptor.scalar_size)
        throw context.error("scalar width does not match the exact precision cell");
    if(descriptor.reserved_2!=0)
        throw context.error("resource descriptor reserved field is nonzero");
    context.require_match(expectation.required_isa,
        context.require_text(descriptor.required_isa,"required ISA").c_str(),"required ISA");
    context.require_match(expectation.full_reverse_reduction_policy,
        context.require_text(descriptor.full_reverse_reduction_policy,
            "full reverse reduction policy").c_str(),
        "full reverse reduction policy");
    context.require_match(expectation.schedule_qualification,
        context.require_text(descriptor.schedule_qualification,"schedule qualification").c_str(),
        "schedule qualification");
    if((whole&&(expectation.vector_policy!="host-simd-lane-major-v1"
            ||expectation.register_budget_policy!="compiler-managed-v1"
            ||expectation.schedule_qualification!="conservative-control"))
        ||(tiled&&(expectation.vector_policy!="channel-tiled-lane-major-v1"
            ||expectation.register_budget_policy!="bounded-owner-stack-v1"
            ||expectation.schedule_qualification!="candidate-unqualified-explicit")))
        throw context.error("host schedule resource tuple is unsupported");
    if(expectation.required_isa!=expectation.host_target
        ||expectation.full_reverse_reduction_policy
            !="owner-local-deterministic-v1")
        throw context.error("host ISA or reduction tuple is unsupported");
}

std::string plugin_directory(const std::string& path)
{
    const std::size_t separator=path.find_last_of('/');
    if(separator==std::string::npos)
        return ".";
    if(separator==0)
        return "/";
    return path.substr(0,separator);
}

}  // namespace

OAMHostPlugin OAMHostPlugin::load(
    std::string path, const OAMHostPluginExpectation& expectation)
{
    if(path.empty())
        throw std::invalid_argument("Execution OAM host plugin path is empty.");
    validate_expectation(expectation);
    void* handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(handle==nullptr) {
        const char* error=dlerror();
        throw plugin_error(path,error==nullptr?"dlopen failed":error);
    }
    try {
        dlerror();
        void* symbol=dlsym(
            handle,SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL);
        const char* error=dlerror();
        if(error!=nullptr)
            throw plugin_error(path,error);
        if(symbol==nullptr)
            throw plugin_error(path,"query symbol is null");
        const auto query=
            reinterpret_cast<SymmetrixJitOAMHostPluginQueryV1>(symbol);
        const auto* descriptor=query();
        if(descriptor==nullptr)
            throw plugin_error(path,"query returned a null descriptor");
        validate_descriptor(path,*descriptor,expectation);
        return OAMHostPlugin(handle,descriptor,std::move(path));
    } catch(...) {
        dlclose(handle);
        throw;
    }
}

OAMHostPlugin::OAMHostPlugin(
    void* handle, const SymmetrixJitOAMHostPluginV1* descriptor,
    std::string path) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path))
{}

OAMHostPlugin::~OAMHostPlugin() { reset(); }

OAMHostPlugin::OAMHostPlugin(OAMHostPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      path_(std::move(other.path_))
{}

OAMHostPlugin& OAMHostPlugin::operator=(OAMHostPlugin&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        path_=std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitOAMHostPluginV1& OAMHostPlugin::descriptor() const
{
    if(descriptor_==nullptr)
        throw std::logic_error("Execution OAM host plugin is not loaded.");
    return *descriptor_;
}

void OAMHostPlugin::reset() noexcept
{
    descriptor_=nullptr;
    if(handle_!=nullptr)
        dlclose(handle_);
    handle_=nullptr;
    path_.clear();
}

std::optional<OAMHostPluginV2Handle> OAMHostPluginV2Handle::load(
    std::string path, const OAMHostPluginExpectationV2& expectation)
{
    if(path.empty())
        throw std::invalid_argument("Execution OAM host plugin path is empty.");
    validate_expectation_v2(expectation);
    void* handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(handle==nullptr) {
        const char* error=dlerror();
        throw plugin_error(path,error==nullptr?"dlopen failed":error);
    }
    try {
        dlerror();
        void* symbol=dlsym(
            handle,SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V7);
        const char* error=dlerror();
        if(error!=nullptr||symbol==nullptr) {
            dlerror();
            const bool stale_v6=dlsym(handle,
                SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V6)!=nullptr;
            dlerror();
            const bool stale_v5=dlsym(handle,
                SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V5)!=nullptr;
            dlerror();
            const bool stale_v4=dlsym(handle,
                SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V4)!=nullptr;
            dlerror();
            const bool stale_v3=dlsym(handle,
                SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V3)!=nullptr;
            dlerror();
            const bool stale_v2=dlsym(handle,
                SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V2)!=nullptr;
            dlerror();
            if(stale_v6||stale_v5||stale_v4||stale_v3||stale_v2)
                throw plugin_error(path,
                    "stale OAM host artifact ABI; regenerate for the source-owned ABI v7");
            dlclose(handle);
            return std::nullopt;
        }
        const auto query=
            reinterpret_cast<SymmetrixJitOAMHostPluginQueryV7>(symbol);
        const auto* descriptor=query();
        if(descriptor==nullptr)
            throw plugin_error(path,"query returned a null descriptor");
        validate_descriptor_v2(path,*descriptor,expectation);
        std::string directory=plugin_directory(path);
        return OAMHostPluginV2Handle(
            handle,descriptor,std::move(path),std::move(directory));
    } catch(...) {
        dlclose(handle);
        throw;
    }
}

OAMHostPluginV2Handle::OAMHostPluginV2Handle(
    void* handle, const SymmetrixJitOAMHostPluginV7* descriptor,
    std::string path, std::string directory) noexcept
    : handle_(handle),descriptor_(descriptor),path_(std::move(path)),
      directory_(std::move(directory))
{}

OAMHostPluginV2Handle::~OAMHostPluginV2Handle() { reset(); }

OAMHostPluginV2Handle::OAMHostPluginV2Handle(
    OAMHostPluginV2Handle&& other) noexcept
    : handle_(std::exchange(other.handle_,nullptr)),
      descriptor_(std::exchange(other.descriptor_,nullptr)),
      path_(std::move(other.path_)),
      directory_(std::move(other.directory_))
{}

OAMHostPluginV2Handle& OAMHostPluginV2Handle::operator=(
    OAMHostPluginV2Handle&& other) noexcept
{
    if(this!=&other) {
        reset();
        handle_=std::exchange(other.handle_,nullptr);
        descriptor_=std::exchange(other.descriptor_,nullptr);
        path_=std::move(other.path_);
        directory_=std::move(other.directory_);
    }
    return *this;
}

const SymmetrixJitOAMHostPluginV7& OAMHostPluginV2Handle::descriptor() const
{
    if(descriptor_==nullptr)
        throw std::logic_error("Execution OAM host plugin is not loaded.");
    return *descriptor_;
}

void OAMHostPluginV2Handle::reset() noexcept
{
    descriptor_=nullptr;
    if(handle_!=nullptr)
        dlclose(handle_);
    handle_=nullptr;
    path_.clear();
    directory_.clear();
}

#else

OAMHostPlugin OAMHostPlugin::load(
    std::string, const OAMHostPluginExpectation&)
{
    throw std::runtime_error(
        "Execution OAM host plugins require POSIX dynamic loading.");
}

OAMHostPlugin::~OAMHostPlugin() = default;
OAMHostPlugin::OAMHostPlugin(OAMHostPlugin&&) noexcept = default;
OAMHostPlugin& OAMHostPlugin::operator=(OAMHostPlugin&&) noexcept = default;
const SymmetrixJitOAMHostPluginV1& OAMHostPlugin::descriptor() const
{
    throw std::logic_error("Execution OAM host plugin is not loaded.");
}
void OAMHostPlugin::reset() noexcept {}

std::optional<OAMHostPluginV2Handle> OAMHostPluginV2Handle::load(
    std::string, const OAMHostPluginExpectationV2&)
{
    throw std::runtime_error(
        "Execution OAM host plugins require POSIX dynamic loading.");
}

OAMHostPluginV2Handle::~OAMHostPluginV2Handle() = default;
OAMHostPluginV2Handle::OAMHostPluginV2Handle(
    OAMHostPluginV2Handle&&) noexcept = default;
OAMHostPluginV2Handle& OAMHostPluginV2Handle::operator=(
    OAMHostPluginV2Handle&&) noexcept = default;
const SymmetrixJitOAMHostPluginV7& OAMHostPluginV2Handle::descriptor() const
{
    throw std::logic_error("Execution OAM host plugin is not loaded.");
}
void OAMHostPluginV2Handle::reset() noexcept {}

#endif

}  // namespace symmetrix::execution
