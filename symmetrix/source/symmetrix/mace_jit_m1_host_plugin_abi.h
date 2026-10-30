#pragma once

#include "mace_jit_m0_host_plugin_abi.h"

#define SYMMETRIX_JIT_M1_HOST_PLUGIN_ABI_VERSION_V1 1u
#define SYMMETRIX_JIT_M1_HOST_PLUGIN_ABI_TAG_V1 \
    "symmetrix.jit.m1-host-plugin/1"
#define SYMMETRIX_JIT_M1_HOST_PLUGIN_QUERY_SYMBOL_V1 \
    "symmetrix_jit_m1_host_plugin_query_v1"
#define SYMMETRIX_JIT_M1_HOST_PLUGIN_BYTE_ORDER_V1 \
    SYMMETRIX_JIT_M0_HOST_PLUGIN_BYTE_ORDER_V1

#define SYMMETRIX_JIT_M1_HOST_SCALAR_FLOAT32_V1 \
    SYMMETRIX_JIT_M0_HOST_SCALAR_FLOAT32_V1
#define SYMMETRIX_JIT_M1_HOST_SCALAR_FLOAT64_V1 \
    SYMMETRIX_JIT_M0_HOST_SCALAR_FLOAT64_V1
#define SYMMETRIX_JIT_M1_HOST_FORWARD_OWNER_V1 \
    SYMMETRIX_JIT_M0_HOST_FORWARD_OWNER_V1
#define SYMMETRIX_JIT_M1_HOST_REVERSE_OWNER_V1 \
    SYMMETRIX_JIT_M0_HOST_REVERSE_OWNER_V1
#define SYMMETRIX_JIT_M1_HOST_ALIAS_SAFE_V1 \
    SYMMETRIX_JIT_M0_HOST_ALIAS_SAFE_V1
#define SYMMETRIX_JIT_M1_HOST_SCALE_ADJOINT_V1 \
    SYMMETRIX_JIT_M0_HOST_SCALE_ADJOINT_V1
#define SYMMETRIX_JIT_M1_HOST_CHANNEL_TILED_OWNER_V1 \
    SYMMETRIX_JIT_M0_HOST_CHANNEL_TILED_OWNER_V1

typedef SymmetrixJitM0HostArgsV1 SymmetrixJitM1HostArgsV1;
typedef SymmetrixJitM0HostOwnerV1 SymmetrixJitM1HostOwnerV1;
typedef SymmetrixJitM0HostPluginV1 SymmetrixJitM1HostPluginV1;
typedef const SymmetrixJitM1HostPluginV1*
    (*SymmetrixJitM1HostPluginQueryV1)(void);

#define SYMMETRIX_JIT_M1_HOST_PLUGIN_ABI_VERSION_V2 2u
#define SYMMETRIX_JIT_M1_HOST_PLUGIN_ABI_TAG_V2 \
    "symmetrix.jit.m1-host-plugin/2"
#define SYMMETRIX_JIT_M1_HOST_PLUGIN_QUERY_SYMBOL_V2 \
    "symmetrix_jit_m1_host_plugin_query_v2"

typedef SymmetrixJitM0HostPluginV2 SymmetrixJitM1HostPluginV2;
typedef const SymmetrixJitM1HostPluginV2*
    (*SymmetrixJitM1HostPluginQueryV2)(void);
