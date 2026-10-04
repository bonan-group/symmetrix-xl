#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
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
#include <vector>

// TODO: remove some of these headers?
#include "KokkosBatched_Util.hpp"
#include "KokkosBlas.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBlas_tpl_spec.hpp"
#if defined(KOKKOS_ENABLE_HIP) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
#include <rocblas/rocblas.h>
#endif
#if defined(KOKKOS_ENABLE_CUDA) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
#include <cuda_runtime_api.h>
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
    template<typename Precision>
    struct BatchedPointerArrays {
        std::vector<const Precision*> a;
        std::vector<const Precision*> b;
        std::vector<Precision*> c;
        const Precision** device_a = nullptr;
        const Precision** device_b = nullptr;
        Precision** device_c = nullptr;
        std::size_t device_capacity = 0;

        void release_device()
        {
            static_cast<void>(cudaFree(device_a));
            static_cast<void>(cudaFree(device_b));
            static_cast<void>(cudaFree(device_c));
            device_a = nullptr;
            device_b = nullptr;
            device_c = nullptr;
            device_capacity = 0;
        }
    };

    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    BatchedPointerArrays<float> float_pointer_arrays;
    BatchedPointerArrays<double> double_pointer_arrays;

    ~FactorizedBlasContext()
    {
        float_pointer_arrays.release_device();
        double_pointer_arrays.release_device();
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

    template<typename Precision>
    BatchedPointerArrays<Precision>& batched_pointer_arrays()
    {
        if constexpr (std::is_same_v<Precision,float>)
            return float_pointer_arrays;
        else
            return double_pointer_arrays;
    }

    template<typename Precision>
    void prepare_batched_pointer_arrays(
        const Precision* a,
        const std::int64_t stride_a,
        const Precision* b,
        const std::int64_t stride_b,
        Precision* c,
        const std::int64_t stride_c,
        const std::int32_t batch_count)
    {
        auto& pointers = batched_pointer_arrays<Precision>();
        const auto count = static_cast<std::size_t>(batch_count);
        pointers.a.resize(count);
        pointers.b.resize(count);
        pointers.c.resize(count);
        for (std::size_t i=0; i<count; ++i) {
            const auto index = static_cast<std::int64_t>(i);
            pointers.a[i] = a + static_cast<std::ptrdiff_t>(index*stride_a);
            pointers.b[i] = b + static_cast<std::ptrdiff_t>(index*stride_b);
            pointers.c[i] = c + static_cast<std::ptrdiff_t>(index*stride_c);
        }
        if (pointers.device_capacity < count) {
            if (stream != nullptr && cudaStreamSynchronize(stream) != cudaSuccess)
                throw std::runtime_error(
                    "CUDA synchronization failed before resizing cuBLAS pointer arrays.");
            pointers.release_device();
            const auto bytes = count*sizeof(const Precision*);
            const auto output_bytes = count*sizeof(Precision*);
            if (cudaMalloc(
                    reinterpret_cast<void**>(&pointers.device_a), bytes)
                    != cudaSuccess
                || cudaMalloc(
                    reinterpret_cast<void**>(&pointers.device_b), bytes)
                    != cudaSuccess
                || cudaMalloc(
                    reinterpret_cast<void**>(&pointers.device_c), output_bytes)
                    != cudaSuccess) {
                pointers.release_device();
                throw std::runtime_error(
                    "CUDA allocation failed for cuBLAS pointer arrays.");
            }
            pointers.device_capacity = count;
        }
        if (cudaMemcpyAsync(
                pointers.device_a, pointers.a.data(),
                count*sizeof(const Precision*), cudaMemcpyHostToDevice, stream)
                != cudaSuccess
            || cudaMemcpyAsync(
                pointers.device_b, pointers.b.data(),
                count*sizeof(const Precision*), cudaMemcpyHostToDevice, stream)
                != cudaSuccess
            || cudaMemcpyAsync(
                pointers.device_c, pointers.c.data(),
                count*sizeof(Precision*), cudaMemcpyHostToDevice, stream)
                != cudaSuccess)
            throw std::runtime_error(
                "CUDA upload failed for cuBLAS pointer arrays.");
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

inline bool execution_cublas_sync_check_enabled()
{
    const char* synchronize = std::getenv("SYMMETRIX_CUBLAS_SYNC_CHECK");
    return synchronize != nullptr
        && (synchronize[0] == '1' || synchronize[0] == 't'
            || synchronize[0] == 'T' || synchronize[0] == 'y'
            || synchronize[0] == 'Y');
}

inline void check_execution_cuda_stream(
    const cudaStream_t stream,
    const char* operation,
    const bool before_call = true)
{
    const bool synchronize_stream = execution_cublas_sync_check_enabled();
    if (!before_call && !synchronize_stream)
        return;
    const cudaError_t status = synchronize_stream
        ? cudaStreamSynchronize(stream) : cudaPeekAtLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(
            std::string(operation)+" encountered a CUDA stream error "
            +(before_call ? "before" : "after")+" cuBLAS: "
            +cudaGetErrorString(status));
}

template<typename Precision>
void launch_execution_strided_batched_gemm(
    FactorizedBlasContext& blas,
    const Kokkos::Cuda& execution_space,
    const FactorizedStridedBatchedGemmDescriptor& descriptor,
    const Precision* a,
    const Precision* b,
    Precision* c,
    const char* operation)
{
    blas.ensure_bound(execution_space.cuda_stream());
    blas.record_launch();
    const Precision alpha = static_cast<Precision>(descriptor.alpha);
    const Precision beta = static_cast<Precision>(descriptor.beta);
    const auto operation_a = execution_cublas_operation(descriptor.transpose_a);
    const auto operation_b = execution_cublas_operation(descriptor.transpose_b);
    if (descriptor.batch_count > 1
        && (descriptor.stride_a == 0 || descriptor.stride_b == 0
            || descriptor.stride_c == 0)) {
        blas.prepare_batched_pointer_arrays(
            a, descriptor.stride_a, b, descriptor.stride_b, c,
            descriptor.stride_c, descriptor.batch_count);
        auto& pointers = blas.batched_pointer_arrays<Precision>();
        check_execution_cuda_stream(
            execution_space.cuda_stream(), operation);
        const auto a_array = pointers.device_a;
        const auto b_array = pointers.device_b;
        auto c_array = pointers.device_c;
        if constexpr (std::is_same_v<Precision,float>) {
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSgemmBatched(
                blas.handle, operation_a, operation_b,
                descriptor.m, descriptor.n, descriptor.k, &alpha,
                a_array, descriptor.leading_a,
                b_array, descriptor.leading_b,
                &beta, c_array, descriptor.leading_c,
                descriptor.batch_count));
        } else {
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasDgemmBatched(
                blas.handle, operation_a, operation_b,
                descriptor.m, descriptor.n, descriptor.k, &alpha,
                a_array, descriptor.leading_a,
                b_array, descriptor.leading_b,
                &beta, c_array, descriptor.leading_c,
                descriptor.batch_count));
        }
        check_execution_cuda_stream(
            execution_space.cuda_stream(), operation, false);
        return;
    }
    check_execution_cuda_stream(
        execution_space.cuda_stream(), operation);
    if constexpr (std::is_same_v<Precision,float>) {
        if (descriptor.batch_count == 1)
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSgemm(
                blas.handle, operation_a, operation_b,
                descriptor.m, descriptor.n, descriptor.k, &alpha,
                a, descriptor.leading_a, b, descriptor.leading_b,
                &beta, c, descriptor.leading_c));
        else
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                blas.handle, operation_a, operation_b,
                descriptor.m, descriptor.n, descriptor.k, &alpha,
                a, descriptor.leading_a, descriptor.stride_a,
                b, descriptor.leading_b, descriptor.stride_b,
                &beta, c, descriptor.leading_c, descriptor.stride_c,
                descriptor.batch_count));
    } else {
        if (descriptor.batch_count == 1)
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasDgemm(
                blas.handle, operation_a, operation_b,
                descriptor.m, descriptor.n, descriptor.k, &alpha,
                a, descriptor.leading_a, b, descriptor.leading_b,
                &beta, c, descriptor.leading_c));
        else
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasDgemmStridedBatched(
                blas.handle, operation_a, operation_b,
                descriptor.m, descriptor.n, descriptor.k, &alpha,
                a, descriptor.leading_a, descriptor.stride_a,
                b, descriptor.leading_b, descriptor.stride_b,
                &beta, c, descriptor.leading_c, descriptor.stride_c,
                descriptor.batch_count));
    }
    check_execution_cuda_stream(
        execution_space.cuda_stream(), operation, false);
}

// Kokkos stores the A1 matrices in row-major LayoutRight views, whereas the
// cuBLAS and rocBLAS APIs consume column-major matrices.  Transposing the
// product on the BLAS side lets us retain the existing storage without packing
// each node:
//   C_row = A_row * B_row  =>  C_col^T = B_row^T * A_row^T.
// KokkosKernels does not expose a public strided-batched GEMM wrapper, so this
// uses the backend TPL handle and stream binding rather than a raw stream.
template<typename Precision>
void launch_execution_row_major_strided_batched_gemm(
    FactorizedBlasContext& blas,
    const Kokkos::Cuda& execution_space,
    const Precision* a_row,
    const Precision* b_row,
    Precision* c_row,
    const std::int32_t rows,
    const std::int32_t inner,
    const std::int32_t columns,
    const std::int64_t stride_a,
    const std::int64_t stride_b,
    const std::int64_t stride_c,
    const std::int32_t batch_count,
    const char* operation)
{
    if (rows <= 0 || inner <= 0 || columns <= 0 || batch_count <= 0)
        return;
    blas.ensure_bound(execution_space.cuda_stream());
    blas.record_launch();
    const Precision alpha = Precision(1);
    const Precision beta = Precision(0);
    // C^T (columns x rows) = B^T (columns x inner) * A^T (inner x rows).
    if (batch_count > 1
        && (stride_a == 0 || stride_b == 0 || stride_c == 0)) {
        // The row-major operands are reversed in the column-major cuBLAS call.
        blas.prepare_batched_pointer_arrays(
            b_row, stride_b, a_row, stride_a, c_row, stride_c, batch_count);
        auto& pointers = blas.batched_pointer_arrays<Precision>();
        check_execution_cuda_stream(
            execution_space.cuda_stream(), operation);
        const auto a_array = pointers.device_a;
        const auto b_array = pointers.device_b;
        auto c_array = pointers.device_c;
        if constexpr (std::is_same_v<Precision,float>) {
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSgemmBatched(
                blas.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                columns, rows, inner, &alpha,
                a_array, columns, b_array, inner,
                &beta, c_array, columns, batch_count));
        } else {
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasDgemmBatched(
                blas.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                columns, rows, inner, &alpha,
                a_array, columns, b_array, inner,
                &beta, c_array, columns, batch_count));
        }
        check_execution_cuda_stream(
            execution_space.cuda_stream(), operation, false);
        return;
    }
    check_execution_cuda_stream(
        execution_space.cuda_stream(), operation);
    if constexpr (std::is_same_v<Precision,float>) {
        if (batch_count == 1)
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSgemm(
                blas.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                columns, rows, inner, &alpha,
                b_row, columns, a_row, inner,
                &beta, c_row, columns));
        else
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
                blas.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                columns, rows, inner, &alpha,
                b_row, columns, stride_b,
                a_row, inner, stride_a,
                &beta, c_row, columns, stride_c,
                batch_count));
    } else {
        if (batch_count == 1)
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasDgemm(
                blas.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                columns, rows, inner, &alpha,
                b_row, columns, a_row, inner,
                &beta, c_row, columns));
        else
            KOKKOSBLAS_IMPL_CUBLAS_SAFE_CALL(cublasDgemmStridedBatched(
                blas.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                columns, rows, inner, &alpha,
                b_row, columns, stride_b,
                a_row, inner, stride_a,
                &beta, c_row, columns, stride_c,
                batch_count));
    }
    check_execution_cuda_stream(
        execution_space.cuda_stream(), operation, false);
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

template<typename Precision>
void launch_execution_row_major_strided_batched_gemm(
    FactorizedBlasContext& blas,
    const Kokkos::HIP& execution_space,
    const Precision* a_row,
    const Precision* b_row,
    Precision* c_row,
    const std::int32_t rows,
    const std::int32_t inner,
    const std::int32_t columns,
    const std::int64_t stride_a,
    const std::int64_t stride_b,
    const std::int64_t stride_c,
    const std::int32_t batch_count,
    const char* operation)
{
    if (rows <= 0 || inner <= 0 || columns <= 0 || batch_count <= 0)
        return;
    blas.ensure_bound(execution_space.hip_stream(), execution_space.hip_device());
    blas.record_launch();
    const Precision alpha = Precision(1);
    const Precision beta = Precision(0);
    // C^T (columns x rows) = B^T (columns x inner) * A^T (inner x rows).
    const auto status = [&] {
        if constexpr (std::is_same_v<Precision,float>)
            return rocblas_sgemm_strided_batched(
                blas.hip_handle, rocblas_operation_none, rocblas_operation_none,
                columns, rows, inner, &alpha,
                b_row, columns, static_cast<rocblas_stride>(stride_b),
                a_row, inner, static_cast<rocblas_stride>(stride_a),
                &beta, c_row, columns, static_cast<rocblas_stride>(stride_c),
                batch_count);
        else
            return rocblas_dgemm_strided_batched(
                blas.hip_handle, rocblas_operation_none, rocblas_operation_none,
                columns, rows, inner, &alpha,
                b_row, columns, static_cast<rocblas_stride>(stride_b),
                a_row, inner, static_cast<rocblas_stride>(stride_a),
                &beta, c_row, columns, static_cast<rocblas_stride>(stride_c),
                batch_count);
    }();
    check_execution_rocblas(status, operation);
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

inline bool execution_a1_blas_enabled()
{
    // Keep the historical environment name as a compatibility alias for both
    // CUDA/cuBLAS and HIP/rocBLAS builds.
    const char* value = std::getenv("SYMMETRIX_A1_CUBLAS");
    if (value == nullptr || value[0] == '\0')
        return true;
    const std::string_view requested(value);
    return requested == "1" || requested == "true" || requested == "on";
}

inline bool execution_a1_flatten_enabled()
{
    const char* value = std::getenv("SYMMETRIX_A1_FLATTEN");
    if (value == nullptr || value[0] == '\0')
        return true;
    const std::string_view requested(value);
    return requested == "1" || requested == "true" || requested == "on";
}
