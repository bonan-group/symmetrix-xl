#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

class MultilayerPerceptronKokkos
{
public:

MultilayerPerceptronKokkos();
    MultilayerPerceptronKokkos(
        std::vector<int> shape,
        std::vector<std::vector<double>> weights,
        double activation_scale_factor = 1.0);
    MultilayerPerceptronKokkos(const MultilayerPerceptronKokkos&) = delete;
    MultilayerPerceptronKokkos& operator=(
        const MultilayerPerceptronKokkos&) = delete;
    MultilayerPerceptronKokkos(MultilayerPerceptronKokkos&&) = default;
    MultilayerPerceptronKokkos& operator=(
        MultilayerPerceptronKokkos&&) = default;
    ~MultilayerPerceptronKokkos() = default;

void evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> x, Kokkos::View<double*,Kokkos::LayoutRight> f);
void evaluate(
    const Kokkos::DefaultExecutionSpace& execution_space,
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    bool completion_fence);
void evaluate_gradient(Kokkos::View<const double**,Kokkos::LayoutRight> x, Kokkos::View<double*,Kokkos::LayoutRight> f, Kokkos::View<double**,Kokkos::LayoutRight> g);
void evaluate_gradient(
    const Kokkos::DefaultExecutionSpace& execution_space,
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    bool completion_fence);
void evaluate_gradient_accumulate_recompute(
    const Kokkos::DefaultExecutionSpace& execution_space,
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    bool completion_fence);
void evaluate_gradient_directional(
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<const double**,Kokkos::LayoutRight> x_dot,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    Kokkos::View<double**,Kokkos::LayoutRight> g_dot);
void evaluate_gradient_directional(
    const Kokkos::DefaultExecutionSpace& execution_space,
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<const double**,Kokkos::LayoutRight> x_dot,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    Kokkos::View<double**,Kokkos::LayoutRight> g_dot,
    bool completion_fence);
void evaluate_gradient_directional_recompute(
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<const double**,Kokkos::LayoutRight> x_dot,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    Kokkos::View<double**,Kokkos::LayoutRight> g_dot);
void release_workspace();
std::size_t workspace_bytes() const;
std::size_t estimated_workspace_bytes(
    std::size_t batch_size, bool derivatives, bool directional = false) const;

private:

void ensure_workspace(int batch_size, bool derivatives, bool directional = false);

std::vector<int> shape_host;
Kokkos::View<int*> shape;
Kokkos::View<int*> node_offsets;
Kokkos::View<int*> weight_offsets;
Kokkos::View<double*> weights;
double activation_scale = 1.0;
int total_node_width = 0;

Kokkos::View<double**,Kokkos::LayoutRight> node_values;
Kokkos::View<double**,Kokkos::LayoutRight> node_derivatives;
Kokkos::View<double**,Kokkos::LayoutRight> node_value_dots;
Kokkos::View<double**,Kokkos::LayoutRight> node_derivative_dots;

};
