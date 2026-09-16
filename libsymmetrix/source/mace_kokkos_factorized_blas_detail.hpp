#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>

// TODO: remove some of these headers?
#include "KokkosBatched_Util.hpp"
#include "KokkosBlas.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBlas_tpl_spec.hpp"
#if defined(KOKKOS_ENABLE_HIP) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
#include <rocblas/rocblas.h>
#endif
#include "nlohmann/json.hpp"
#include "sphericart.hpp"
#include "sphericart_cuda.hpp"
#include "spherical_harmonic_device.hpp"

#include "tools_kokkos.hpp"
#include "mace_kokkos.hpp"
#include "device_backend.hpp"
#include "factorized_blas.hpp"

struct FactorizedBlasContext {
    std::size_t launch_count = 0;
    std::size_t stream_bind_count = 0;

    void record_launch() noexcept { launch_count += 1; }
    void record_stream_binding() noexcept { stream_bind_count += 1; }
#if defined(KOKKOS_ENABLE_CUDA) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;

    ~FactorizedBlasContext()
    {
        if (handle != nullptr)
            cublasDestroy(handle);
    }

    void ensure_bound(cudaStream_t requested_stream)
    {
        if (handle != nullptr) {
            if (stream != requested_stream)
                throw std::runtime_error(
                    "Execution R1 execution stream changed after cuBLAS preparation.");
            return;
        }
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasCreate(&handle));
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(
            cublasSetStream(handle, requested_stream));
        stream = requested_stream;
        record_stream_binding();
    }
#endif
#if defined(KOKKOS_ENABLE_HIP) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
    rocblas_handle hip_handle = nullptr;
    hipStream_t hip_stream = nullptr;
    int hip_device = -1;

    ~FactorizedBlasContext()
    {
        if (hip_handle != nullptr) {
            int previous = -1;
            static_cast<void>(hipGetDevice(&previous));
            if (hip_device >= 0)
                static_cast<void>(hipSetDevice(hip_device));
            static_cast<void>(rocblas_destroy_handle(hip_handle));
            if (previous >= 0 && previous != hip_device)
                static_cast<void>(hipSetDevice(previous));
        }
    }

    void ensure_bound(hipStream_t requested_stream, int requested_device)
    {
        if (hip_handle != nullptr) {
            if (hip_stream != requested_stream || hip_device != requested_device)
                throw std::runtime_error(
                    "Execution R1 HIP execution stream or device changed after rocBLAS preparation.");
            return;
        }
        symmetrix::execution::HipDeviceGuard device_guard(requested_device);
        if (rocblas_create_handle(&hip_handle) != rocblas_status_success)
            throw std::runtime_error("Could not create and bind the Execution rocBLAS handle.");
        hip_device = requested_device;
        if (rocblas_set_stream(hip_handle, requested_stream)
            != rocblas_status_success) {
            static_cast<void>(rocblas_destroy_handle(hip_handle));
            hip_handle = nullptr;
            hip_device = -1;
            throw std::runtime_error(
                "Could not create and bind the Execution rocBLAS handle.");
        }
        hip_stream = requested_stream;
        record_stream_binding();
    }
#endif
};

#if defined(KOKKOS_ENABLE_CUDA) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
inline cublasOperation_t execution_cublas_operation(
    const FactorizedBlasTranspose transpose)
{
    return transpose == FactorizedBlasTranspose::none ? CUBLAS_OP_N : CUBLAS_OP_T;
}

template<typename Precision>
void launch_execution_strided_batched_gemm(
    FactorizedBlasContext& blas,
    const Kokkos::Cuda& execution_space,
    const FactorizedStridedBatchedGemmDescriptor& descriptor,
    const Precision* a,
    const Precision* b,
    Precision* c,
    const char*)
{
    blas.ensure_bound(execution_space.cuda_stream());
    blas.record_launch();
    const Precision alpha = static_cast<Precision>(descriptor.alpha);
    const Precision beta = static_cast<Precision>(descriptor.beta);
    const auto operation_a = execution_cublas_operation(descriptor.transpose_a);
    const auto operation_b = execution_cublas_operation(descriptor.transpose_b);
    if constexpr (std::is_same_v<Precision,float>) {
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
            blas.handle, operation_a, operation_b,
            descriptor.m, descriptor.n, descriptor.k, &alpha,
            a, descriptor.leading_a, descriptor.stride_a,
            b, descriptor.leading_b, descriptor.stride_b,
            &beta, c, descriptor.leading_c, descriptor.stride_c,
            descriptor.batch_count));
    } else {
        KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasDgemmStridedBatched(
            blas.handle, operation_a, operation_b,
            descriptor.m, descriptor.n, descriptor.k, &alpha,
            a, descriptor.leading_a, descriptor.stride_a,
            b, descriptor.leading_b, descriptor.stride_b,
            &beta, c, descriptor.leading_c, descriptor.stride_c,
            descriptor.batch_count));
    }
}

template <typename Precision, typename StateView>
void execution_strided_batched_gemm(
    FactorizedBlasContext& blas,
    const Kokkos::Cuda& execution_space,
    Kokkos::View<Precision**,Kokkos::LayoutRight> radial,
    StateView state,
    Kokkos::View<Precision**,Kokkos::LayoutRight> coupling,
    int batch,
    int degree,
    int embedding,
    int columns,
    int state_columns,
    int coupling_columns)
{
    const auto descriptor = make_execution_reverse_gemm_descriptor<Precision>(
        batch, degree, embedding, columns, state_columns, coupling_columns);
    launch_execution_strided_batched_gemm(
        blas, execution_space, descriptor,
        state.data(), radial.data(), coupling.data(),
        "Execution strided batched GEMM");
}

template <typename Precision, typename AggregateView>
void execution_strided_batched_forward_gemm(
    FactorizedBlasContext& blas,
    const Kokkos::Cuda& execution_space,
    Kokkos::View<Precision**,Kokkos::LayoutRight> radial,
    Kokkos::View<Precision**,Kokkos::LayoutRight> coupling,
    AggregateView aggregate,
    int batch,
    int degree,
    int embedding,
    int columns,
    int maximum_columns)
{
    const auto descriptor = make_execution_forward_gemm_descriptor<Precision>(
        batch, degree, embedding, columns, maximum_columns);
    launch_execution_strided_batched_gemm(
        blas, execution_space, descriptor,
        coupling.data(), radial.data(), aggregate.data(),
        "Execution forward strided batched GEMM");
}

#endif

#if defined(KOKKOS_ENABLE_HIP) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
inline void check_execution_rocblas(rocblas_status status, const char* operation)
{
    if (status != rocblas_status_success)
        throw std::runtime_error(
            std::string(operation)+" failed with rocBLAS status "
            +std::to_string(static_cast<int>(status)));
}

inline rocblas_operation execution_rocblas_operation(
    const FactorizedBlasTranspose transpose)
{
    return transpose == FactorizedBlasTranspose::none
        ? rocblas_operation_none : rocblas_operation_transpose;
}

template<typename Precision>
void launch_execution_strided_batched_gemm(
    FactorizedBlasContext& blas,
    const Kokkos::HIP& execution_space,
    const FactorizedStridedBatchedGemmDescriptor& descriptor,
    const Precision* a,
    const Precision* b,
    Precision* c,
    const char* operation)
{
    blas.ensure_bound(execution_space.hip_stream(), execution_space.hip_device());
    blas.record_launch();
    const Precision alpha = static_cast<Precision>(descriptor.alpha);
    const Precision beta = static_cast<Precision>(descriptor.beta);
    const auto operation_a = execution_rocblas_operation(descriptor.transpose_a);
    const auto operation_b = execution_rocblas_operation(descriptor.transpose_b);
    const auto status = [&] {
        if constexpr (std::is_same_v<Precision,float>)
            return rocblas_sgemm_strided_batched(
                blas.hip_handle, operation_a, operation_b,
                descriptor.m, descriptor.n, descriptor.k, &alpha,
                a, descriptor.leading_a,
                static_cast<rocblas_stride>(descriptor.stride_a),
                b, descriptor.leading_b,
                static_cast<rocblas_stride>(descriptor.stride_b),
                &beta, c, descriptor.leading_c,
                static_cast<rocblas_stride>(descriptor.stride_c),
                descriptor.batch_count);
        else
            return rocblas_dgemm_strided_batched(
                blas.hip_handle, operation_a, operation_b,
                descriptor.m, descriptor.n, descriptor.k, &alpha,
                a, descriptor.leading_a,
                static_cast<rocblas_stride>(descriptor.stride_a),
                b, descriptor.leading_b,
                static_cast<rocblas_stride>(descriptor.stride_b),
                &beta, c, descriptor.leading_c,
                static_cast<rocblas_stride>(descriptor.stride_c),
                descriptor.batch_count);
    }();
    check_execution_rocblas(status, operation);
}

template <typename Precision, typename StateView>
void execution_strided_batched_gemm(
    FactorizedBlasContext& blas, const Kokkos::HIP& execution_space,
    Kokkos::View<Precision**,Kokkos::LayoutRight> radial, StateView state,
    Kokkos::View<Precision**,Kokkos::LayoutRight> coupling, int batch,
    int degree, int embedding, int columns, int state_columns,
    int coupling_columns)
{
    const auto descriptor = make_execution_reverse_gemm_descriptor<Precision>(
        batch, degree, embedding, columns, state_columns, coupling_columns);
    launch_execution_strided_batched_gemm(
        blas, execution_space, descriptor,
        state.data(), radial.data(), coupling.data(),
        "Execution strided batched GEMM");
}

template <typename Precision, typename AggregateView>
void execution_strided_batched_forward_gemm(
    FactorizedBlasContext& blas, const Kokkos::HIP& execution_space,
    Kokkos::View<Precision**,Kokkos::LayoutRight> radial,
    Kokkos::View<Precision**,Kokkos::LayoutRight> coupling,
    AggregateView aggregate, int batch, int degree, int embedding, int columns,
    int maximum_columns)
{
    const auto descriptor = make_execution_forward_gemm_descriptor<Precision>(
        batch, degree, embedding, columns, maximum_columns);
    launch_execution_strided_batched_gemm(
        blas, execution_space, descriptor,
        coupling.data(), radial.data(), aggregate.data(),
        "Execution forward strided batched GEMM");
}
#endif
