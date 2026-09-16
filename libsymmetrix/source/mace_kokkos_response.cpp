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
#include "standard_m0.hpp"
#include "factorized_blas.hpp"

using Kokkos::ALL;
using Kokkos::LayoutRight;
using Kokkos::make_pair;
using Kokkos::MemoryUnmanaged;
using Kokkos::parallel_for;
using Kokkos::PerTeam;
using Kokkos::subview;
using Kokkos::TeamPolicy;
using Kokkos::TeamVectorRange;
using Kokkos::TeamVectorMDRange;
using Kokkos::View;
template <typename Precision>
struct AnalyticFieldReverseReducer {
    using value_type = double[];

    static constexpr unsigned value_count = 3;

    int num_channels;
    int num_LM;
    int num_entries;
    int seed;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_adj;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_adj_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_pre_field;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_product_adj_dot;
    Kokkos::View<Precision***,Kokkos::LayoutRight> H1_linear_up_weights;
    Kokkos::View<int*> entry_input_lm;
    Kokkos::View<int*> entry_component;
    Kokkos::View<int*> entry_output_lm;
    Kokkos::View<int*> entry_path;
    Kokkos::View<Precision*> entry_coefficient;
    Kokkos::View<Precision***,Kokkos::LayoutRight> path_up_matrix;
    Kokkos::View<const double*> electric_field;
    Kokkos::View<double*> electric_field_hessian;

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t index, double local_hessian[]) const {
        const std::size_t row = static_cast<std::size_t>(num_LM)*num_channels;
        const std::size_t i = index/row;
        const int lm = (index/static_cast<std::size_t>(num_channels))%num_LM;
        const int input = index%static_cast<std::size_t>(num_channels);
        int l = 0;
        while ((l+1)*(l+1) <= lm)
            ++l;
        Precision value = 0.0;
        for (int output=0; output<num_channels; ++output) {
            value += H1_linear_up_weights(l,input,output)
                * H1_adj_dot(i,lm,output);
        }
        const Precision input_feature = H1_pre_field(i,lm,input);
        for (int entry=0; entry<num_entries; ++entry) {
            if (entry_input_lm(entry) != lm)
                continue;
            const int component = entry_component(entry);
            const int output_lm = entry_output_lm(entry);
            const int path = entry_path(entry);
            const Precision coefficient = entry_coefficient(entry);
            const Precision field_value = static_cast<Precision>(
                electric_field(component));
            for (int output=0; output<num_channels; ++output) {
                const Precision weight = coefficient
                    * path_up_matrix(path,input,output);
                const Precision output_adj = H1_adj(i,output_lm,output);
                const Precision output_adj_dot = H1_adj_dot(i,output_lm,output);
                value += weight * (field_value*output_adj_dot
                    + (component == seed ? output_adj : Precision(0)));
                local_hessian[component] += static_cast<double>(
                    weight * input_feature * output_adj_dot);
            }
        }
        H1_product_adj_dot(i,lm,input) = value;
    }

    KOKKOS_INLINE_FUNCTION
    void init(double update[]) const {
        for (int component=0; component<3; ++component)
            update[component] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(double dst[], const double src[]) const {
        for (int component=0; component<3; ++component)
            dst[component] += src[component];
    }

    KOKKOS_INLINE_FUNCTION
    void final(double update[]) const {
        for (int component=0; component<3; ++component)
            electric_field_hessian(component*3+seed) += update[component];
    }
};

#include "mace_kokkos_response.tpp"


template class MACEKokkos<float>;
template class MACEKokkos<double>;
