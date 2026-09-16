#pragma once

#include <stdint.h>

#define SYMMETRIX_JIT_M0_HOST_PLUGIN_ABI_VERSION_V1 1u
#define SYMMETRIX_JIT_M0_HOST_PLUGIN_ABI_TAG_V1 \
    "symmetrix.jit.m0-host-plugin/1"
#define SYMMETRIX_JIT_M0_HOST_PLUGIN_QUERY_SYMBOL_V1 \
    "symmetrix_jit_m0_host_plugin_query_v1"
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SYMMETRIX_JIT_M0_HOST_PLUGIN_BYTE_ORDER_V1 0x04030201u
#else
#define SYMMETRIX_JIT_M0_HOST_PLUGIN_BYTE_ORDER_V1 0x01020304u
#endif

#define SYMMETRIX_JIT_M0_HOST_SCALAR_FLOAT32_V1 1u
#define SYMMETRIX_JIT_M0_HOST_SCALAR_FLOAT64_V1 2u
#define SYMMETRIX_JIT_M0_HOST_FORWARD_OWNER_V1 (1u << 0)
#define SYMMETRIX_JIT_M0_HOST_REVERSE_OWNER_V1 (1u << 1)
#define SYMMETRIX_JIT_M0_HOST_ALIAS_SAFE_V1 (1u << 2)
#define SYMMETRIX_JIT_M0_HOST_SCALE_ADJOINT_V1 (1u << 3)
#define SYMMETRIX_JIT_M0_HOST_CHANNEL_TILED_OWNER_V1 (1u << 4)

typedef struct SymmetrixJitM0HostArgsV1 {
    uint32_t struct_size;
    uint32_t reserved;
    int64_t num_nodes;
    int32_t channels;
    int32_t capture_input_scale_adjoint;
    const int32_t* node_types;
    const void* input;
    const void* weights;
    const void* output_adjoint;
    void* output;
    void* input_adjoint;
    double* input_scale_adjoint;
} SymmetrixJitM0HostArgsV1;

typedef void (*SymmetrixJitM0HostOwnerV1)(
    const SymmetrixJitM0HostArgsV1*, int64_t);

typedef struct SymmetrixJitM0HostPluginV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t scalar_kind;
    uint32_t scalar_size;
    uint32_t capabilities;
    int32_t channels;
    int32_t input_components;
    int32_t output_components;
    int32_t correlation;
    int32_t term_count;
    uint32_t owner_channel_tile;
    const char* abi_tag;
    const char* artifact_id;
    const char* structure_fingerprint;
    SymmetrixJitM0HostOwnerV1 forward_owner;
    SymmetrixJitM0HostOwnerV1 reverse_owner;
} SymmetrixJitM0HostPluginV1;

typedef const SymmetrixJitM0HostPluginV1*
    (*SymmetrixJitM0HostPluginQueryV1)(void);
