#pragma once
// Each degree flattens (node, component) into the GEMM's rows. All degrees
// share a grid, with one accumulator per output and an ordered FP32 dot product.
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace symmetrix::optimized {

struct RowGemmProblem {
    const float* B;       // K x NC, row stride NC
    std::size_t a_off, a_rs; // Offset and row stride within one node.
    std::size_t c_off, c_rs;
    int K, NC, comps;     // comps = rows per node (2l+1)
    int tiles_n, tile_begin;
};

struct RowGemmBatch {
    const float* A;
    float* C;
    std::size_t a_ns, c_ns; // Node strides in elements.
    int num_nodes, count, total_tiles;
    RowGemmProblem p[4];
    // kRowGemmScaledLoad: every A element is loaded as float(double(a)*load_scale[node]), the operand rounding of the
    // flatten pack, so the contraction carries the density scale.
    const double* load_scale = nullptr;
};

inline constexpr int kRowGemmPlain = 0, kRowGemmScaledLoad = 3;

// 16-byte float vector for the tile loads and stores.  clang (HIP, clang-CUDA) gets the native ext_vector_type, which SROA
// keeps in registers; nvcc and gcc get a 16-byte aligned aggregate with the same members and brace initialisation.
#if defined(__clang__)
typedef float row_gemm_f4 __attribute__((ext_vector_type(4)));
#else
struct alignas(16) row_gemm_f4 { float x, y, z, w; };
#endif

template <int BM, int BN, int BK>
inline bool row_gemm_prepare(RowGemmBatch& gb)
{
    // The loads are 16-byte vectors and the Kokkos league rank is an int.
    if (gb.count < 1 || gb.count > 4 || gb.num_nodes < 0) return false;
    if (reinterpret_cast<std::uintptr_t>(gb.A) % 16 != 0) return false;
    if (gb.a_ns % 4 != 0) return false;
    int tiles = 0;
    for (int q = 0; q < gb.count; ++q) {
        RowGemmProblem& P = gb.p[q];
        if (P.B == nullptr || reinterpret_cast<std::uintptr_t>(P.B) % 16 != 0) return false;
        if (P.K <= 0 || P.K % BK != 0 || P.NC <= 0 || P.NC % BN != 0 || P.comps <= 0) return false;
        if (P.a_off % 4 != 0 || P.a_rs % 4 != 0) return false;
        const long long rows = static_cast<long long>(gb.num_nodes)*P.comps;
        if (rows + BM > std::numeric_limits<int>::max()) return false;
        const long long tiles_m = (rows + BM - 1)/BM;
        P.tiles_n = P.NC/BN;
        P.tile_begin = tiles;
        const long long t = static_cast<long long>(tiles) + tiles_m*P.tiles_n;
        if (t > std::numeric_limits<int>::max()) return false;
        tiles = static_cast<int>(t);
    }
    gb.total_tiles = tiles;
    return true;
}

// A named functor: the host test runs exactly this code with 256 threads per team and a barrier as team_barrier.
template <int BM, int BN, int BK, int TM, int TN, int Epilogue = kRowGemmPlain>
struct RowGemmKernel {
    static_assert((BM/TM)*(BN/TN) == 256, "256 threads per team");
    static_assert(TM % 4 == 0 && TN % 4 == 0, "register blocks are float4 multiples");
    static constexpr int A_LD = BM*BK/1024;
    static constexpr int B_LD = BK*BN/1024;
    static_assert(A_LD >= 1 && B_LD >= 1, "tile too small for 256 threads x float4");
    static constexpr int team_size = (BM/TM)*(BN/TN);
    static constexpr std::size_t shmem_bytes = sizeof(float)*(BK*(BM+4) + BK*(BN+4)) + 64;
    RowGemmBatch gb;

    template <class Member>
    KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
        const int bid = m.league_rank();
        const int t = m.team_rank();
        // Field-wise selection with constant array indices only: copying a whole gb.p[q] through a data-dependent
        // choice spills the kernel-argument problem array to scratch.
        const int q = int(gb.count > 1 && bid >= gb.p[1].tile_begin) + int(gb.count > 2 && bid >= gb.p[2].tile_begin)
            + int(gb.count > 3 && bid >= gb.p[3].tile_begin);
#define ROW_GEMM_PICK(field) (q == 0 ? gb.p[0].field : q == 1 ? gb.p[1].field : q == 2 ? gb.p[2].field : gb.p[3].field)
        const struct {
            const float* B;
            std::size_t a_off, a_rs, c_off, c_rs;
            int K, NC, comps, tiles_n, tile_begin;
        } P = {ROW_GEMM_PICK(B), ROW_GEMM_PICK(a_off), ROW_GEMM_PICK(a_rs), ROW_GEMM_PICK(c_off), ROW_GEMM_PICK(c_rs),
               ROW_GEMM_PICK(K), ROW_GEMM_PICK(NC), ROW_GEMM_PICK(comps), ROW_GEMM_PICK(tiles_n), ROW_GEMM_PICK(tile_begin)};
#undef ROW_GEMM_PICK
        float* As = static_cast<float*>(m.team_scratch(0).get_shmem_aligned(sizeof(float)*BK*(BM+4), 16));
        float* Bs = static_cast<float*>(m.team_scratch(0).get_shmem_aligned(sizeof(float)*BK*(BN+4), 16));
        const int R = gb.num_nodes*P.comps;   // < 2^31 checked in row_gemm_prepare
        const int tile = bid - P.tile_begin;
        const int tn = tile % P.tiles_n;
        const int tm = tile / P.tiles_n;
        const int tx = t % (BN/TN);
        const int ty = t / (BN/TN);
        row_gemm_f4 ra[A_LD], rb[B_LD];
        auto load = [&](const int k0) {
            for (int s = 0; s < A_LD; ++s) {
                const int idx = t + s*256, row = idx/(BK/4), kq = idx%(BK/4);
                const int r = tm*BM + row;
                if (r < R) {
                    const int node = r/P.comps, i = r%P.comps;
                    ra[s] = *reinterpret_cast<const row_gemm_f4*>(
                        gb.A + static_cast<std::size_t>(node)*gb.a_ns + P.a_off
                        + static_cast<std::size_t>(i)*P.a_rs + k0 + kq*4);
                    if constexpr (Epilogue == kRowGemmScaledLoad) {
                        const double scale = gb.load_scale[node];
                        ra[s].x = static_cast<float>(static_cast<double>(ra[s].x)*scale);
                        ra[s].y = static_cast<float>(static_cast<double>(ra[s].y)*scale);
                        ra[s].z = static_cast<float>(static_cast<double>(ra[s].z)*scale);
                        ra[s].w = static_cast<float>(static_cast<double>(ra[s].w)*scale);
                    }
                } else {
                    ra[s] = row_gemm_f4{0.f, 0.f, 0.f, 0.f};
                }
            }
            for (int s = 0; s < B_LD; ++s) {
                const int idx = t + s*256, k = idx/(BN/4), nq = idx%(BN/4);
                rb[s] = *reinterpret_cast<const row_gemm_f4*>(
                    P.B + static_cast<long long>(k0 + k)*P.NC + tn*BN + nq*4);
            }
        };
        auto store = [&]() {
            for (int s = 0; s < A_LD; ++s) {
                const int idx = t + s*256, row = idx/(BK/4), kq = idx%(BK/4);
                As[(kq*4 + 0)*(BM+4) + row] = ra[s].x;
                As[(kq*4 + 1)*(BM+4) + row] = ra[s].y;
                As[(kq*4 + 2)*(BM+4) + row] = ra[s].z;
                As[(kq*4 + 3)*(BM+4) + row] = ra[s].w;
            }
            for (int s = 0; s < B_LD; ++s) {
                const int idx = t + s*256, k = idx/(BN/4), nq = idx%(BN/4);
                *reinterpret_cast<row_gemm_f4*>(&Bs[k*(BN+4) + nq*4]) = rb[s];
            }
        };
        float acc[TM][TN];
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j)
                acc[i][j] = 0.0f;
        load(0);
        store();
        m.team_barrier();
        for (int k0 = 0; k0 < P.K; k0 += BK) {
            const bool more = k0 + BK < P.K;
            if (more) load(k0 + BK);
#pragma unroll 4
            for (int kk = 0; kk < BK; ++kk) {
                float a[TM], b[TN];
                for (int i = 0; i < TM; i += 4) {
                    const row_gemm_f4 v = *reinterpret_cast<const row_gemm_f4*>(&As[kk*(BM+4) + ty*TM + i]);
                    a[i] = v.x; a[i+1] = v.y; a[i+2] = v.z; a[i+3] = v.w;
                }
                for (int j = 0; j < TN; j += 4) {
                    const row_gemm_f4 v = *reinterpret_cast<const row_gemm_f4*>(&Bs[kk*(BN+4) + tx*TN + j]);
                    b[j] = v.x; b[j+1] = v.y; b[j+2] = v.z; b[j+3] = v.w;
                }
                for (int i = 0; i < TM; ++i)
                    for (int j = 0; j < TN; ++j)
                        acc[i][j] += a[i]*b[j];   // Match unblocked TeamGemm and the build’s contraction policy.
            }
            m.team_barrier();
            if (more) {
                store();
                m.team_barrier();
            }
        }
        for (int i = 0; i < TM; ++i) {
            const int r = tm*BM + ty*TM + i;
            if (r >= R) continue;
            const int node = r/P.comps, ii = r%P.comps;
            float* cp = gb.C + static_cast<std::size_t>(node)*gb.c_ns + P.c_off
                + static_cast<std::size_t>(ii)*P.c_rs + tn*BN + tx*TN;
            for (int j = 0; j < TN; ++j)
                cp[j] = 0.0f + acc[i][j];
        }
    }
};

template <int BM, int BN, int BK, int TM, int TN, int Epilogue = kRowGemmPlain, class ExecutionSpace>
inline void row_gemm_launch(const char* label, const ExecutionSpace& execution_space, const RowGemmBatch& gb)
{
    using Kernel = RowGemmKernel<BM, BN, BK, TM, TN, Epilogue>;
    static_assert(sizeof(Kernel) < 512, "the row GEMM functor must stay small (kernel-argument launch)");
    if (gb.total_tiles == 0) return;
    using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
    auto policy = Kokkos::Experimental::require(
        Policy(execution_space, gb.total_tiles, Kernel::team_size),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
    policy.set_scratch_size(0, Kokkos::PerTeam(Kernel::shmem_bytes));
    Kokkos::parallel_for(label, policy, Kernel{gb});
}

// Keep the measured tile only where the device can run its full team.
template <int BM, int BN, int BK, int TM, int TN>
bool row_gemm_fits(int subgroup_width, std::size_t scratch_limit)
{
    using Kernel = RowGemmKernel<BM,BN,BK,TM,TN>;
    return subgroup_width > 0 && Kernel::team_size % subgroup_width == 0 && Kernel::shmem_bytes <= scratch_limit;
}

// Ranges [a, a+na) and [c, c+nc) in elements of float.
inline bool ranges_overlap(const float* a, const std::size_t na, const float* c, const std::size_t nc)
{
    const auto ab = reinterpret_cast<std::uintptr_t>(a), ae = ab + na*sizeof(float);
    const auto cb = reinterpret_cast<std::uintptr_t>(c), ce = cb + nc*sizeof(float);
    return ab < ce && cb < ae;
}

} // namespace symmetrix::optimized
