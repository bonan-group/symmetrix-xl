#pragma once

#include <cufft.h>
#include <string>

namespace symmetrix::cufft_runtime {

// This is the provider surface used by the pinned KokkosFFT CUDA headers.
#define SYMMETRIX_CUFFT_FUNCTIONS(X) \
    X(cufftPlan1d) X(cufftPlan2d) X(cufftPlan3d) X(cufftPlanMany) \
    X(cufftCreate) X(cufftDestroy) X(cufftSetStream) \
    X(cufftSetAutoAllocation) X(cufftSetWorkArea) \
    X(cufftMakePlan1d) X(cufftMakePlan2d) X(cufftMakePlan3d) X(cufftMakePlanMany) \
    X(cufftExecR2C) X(cufftExecC2R) X(cufftExecD2Z) X(cufftExecZ2D) \
    X(cufftExecC2C) X(cufftExecZ2Z) X(cufftGetVersion)

struct Api {
#define SYMMETRIX_CUFFT_DECLARE(name) decltype(&::name) name = nullptr;
    SYMMETRIX_CUFFT_FUNCTIONS(SYMMETRIX_CUFFT_DECLARE)
#undef SYMMETRIX_CUFFT_DECLARE
};

struct Information {
    bool available = false;
    std::string state = "unloaded";
    std::string library;
    std::string reason;
    int version = 0;
};

// Registering a candidate never loads the vendor runtime.
void configure_library(const std::string& path);
Information information(bool probe = false);
const Api& api();

}  // namespace symmetrix::cufft_runtime
