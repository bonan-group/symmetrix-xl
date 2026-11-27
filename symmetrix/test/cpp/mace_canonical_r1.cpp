// Link against an OpenMP build of libsymmetrix. The fixture must be a compact
// model with three elements, equal R1 splines and distinct active H0 embeddings.
#include "mace_kokkos.hpp"

#include <iostream>

namespace {
using Evaluator = MACEKokkos<float>;

// Exercise real graph preparation with host-accessible memory and a qualified
// device row. Explicit instantiation gives this probe access to the selector,
// without a product-side test switch or a GPU launch.
template <class Tag, typename Tag::Member member>
struct Access {
    friend typename Tag::Member binding(Tag) { return member; }
};
struct DeviceRow {
    using Member = std::string Evaluator::*;
    friend Member binding(DeviceRow);
};
template struct Access<DeviceRow, &Evaluator::kernel_architecture_>;

struct CanonicalMap {
    using Member = Kokkos::View<int*> Evaluator::*;
    friend Member binding(CanonicalMap);
};
template struct Access<CanonicalMap, &Evaluator::r1_canonical_type_map_>;

float r0_coefficient(const Evaluator& evaluator, int receiver, int neighbor)
{
    const int pair = evaluator.type_to_active(receiver)*evaluator.num_active_types
        + evaluator.type_to_active(neighbor);
    return evaluator.R0_spline_coefficients(pair, 1, 0, 0);
}
}

int main(int argc, char* argv[])
{
    if (argc != 2)
        return 2;
    Kokkos::ScopeGuard guard;
    using symmetrix::execution::MaceKernels;
    Evaluator control(argv[1]), optimized(argv[1], {}, MaceKernels::optimized);
    control.set_streamed_edges("generic");
    optimized.set_streamed_edges("generic");
    optimized.*binding(DeviceRow{}) = "gfx1201";
    bool same = true;
    for (const auto& types : {std::vector{0,2}, std::vector{2}, std::vector{1,2}}) {
        control.prepare_active_types(types);
        optimized.prepare_active_types(types);
        const auto canonical = optimized.*binding(CanonicalMap{});
        const auto receiver = types.front();
        const auto neighbor = types.back();
        const float expected = r0_coefficient(control, receiver, neighbor);
        const float actual = r0_coefficient(optimized, receiver, neighbor);
        const float first = r0_coefficient(control, receiver, receiver);
        std::cout << "species=" << types.size()
                  << " canonical=" << canonical.size()
                  << " first_R0=" << first << " expected_R0=" << expected
                  << " actual_R0=" << actual << '\n';
        if (types.size() == 2) {
            same &= first != expected && canonical.extent(0) == 3;
            if (canonical.extent(0) == 3) {
                for (const int type : types)
                    same &= canonical(type) == 0;
                same &= canonical(3-receiver-neighbor) == -1;
            }
        } else {
            same &= canonical.extent(0) == 0;
        }
        same &= expected == actual;
    }
    // A forward contraction consuming a prepared scale skips the divide.
    // Its binding must clear when the compact graph is replaced.
    using symmetrix::execution::MaceKernel;
    const auto scale = static_cast<std::size_t>(MaceKernel::a1_divide);
    optimized.A1_scaled = true;
    optimized.a1_scale_factors_ready = true;
    optimized.compute_A1_scaled(2, {}, {}, {}, {});
    const auto fused = optimized.kernel_binding_report().rows[scale].forward;
    optimized.A1_scaled = false;
    optimized.prepare_active_types({2});
    const auto shrunk = optimized.kernel_binding_report().rows[scale].forward;
    std::cout << "A1 scale fused=" << fused.form
              << " shrunk=" << shrunk.form
              << " forward_launches=" << shrunk.launches << '\n';
    same &= std::string_view(fused.form) == "operand_load"
        && std::string_view(shrunk.form) == "not_run"
        && fused.launches == 0 && shrunk.launches == 0;
    return same ? 0 : 1;
}
