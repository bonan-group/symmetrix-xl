#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "multilayer_perceptron_kokkos.hpp"
#include "tools_kokkos.hpp"

MultilayerPerceptronKokkos::MultilayerPerceptronKokkos()
{
    // TODO: add sanity checks to default constructor
}

MultilayerPerceptronKokkos::MultilayerPerceptronKokkos(
    std::vector<int> shape,
    std::vector<std::vector<double>> weights,
    double activation_scale)
{
    if (shape.size() < 2 || shape.back() != 1)
        throw std::invalid_argument(
            "MultilayerPerceptronKokkos requires a scalar output layer.");
    if (weights.size()+1 != shape.size())
        throw std::invalid_argument(
            "MultilayerPerceptronKokkos shape and weight counts differ.");
    if (std::any_of(shape.begin(), shape.end(),
                    [] (const int width) { return width <= 0; }))
        throw std::invalid_argument(
            "MultilayerPerceptronKokkos layer widths must be positive.");

    shape_host = shape;
    std::vector<int> node_offsets_host(shape.size());
    std::vector<int> weight_offsets_host(weights.size());
    std::vector<double> flat_weights;
    for (std::size_t layer=0; layer<shape.size(); ++layer) {
        node_offsets_host[layer] = total_node_width;
        total_node_width += shape[layer];
        if (layer == weights.size())
            continue;
        weight_offsets_host[layer] = flat_weights.size();
        const auto expected = static_cast<std::size_t>(
            shape[layer+1]*shape[layer]);
        if (weights[layer].size() != expected)
            throw std::invalid_argument(
                "MultilayerPerceptronKokkos weight extent does not match shape.");
        flat_weights.insert(
            flat_weights.end(), weights[layer].begin(), weights[layer].end());
    }

    this->shape = Kokkos::View<int*>("mlp_shape", shape_host.size());
    node_offsets = Kokkos::View<int*>(
        "mlp_node_offsets", node_offsets_host.size());
    weight_offsets = Kokkos::View<int*>(
        "mlp_weight_offsets", weight_offsets_host.size());
    this->weights = Kokkos::View<double*>(
        "mlp_weights", flat_weights.size());
    auto host_shape = Kokkos::create_mirror_view(this->shape);
    auto host_node_offsets = Kokkos::create_mirror_view(node_offsets);
    auto host_weight_offsets = Kokkos::create_mirror_view(weight_offsets);
    auto host_weights = Kokkos::create_mirror_view(this->weights);
    for (std::size_t index=0; index<shape_host.size(); ++index)
        host_shape(index) = shape_host[index];
    for (std::size_t index=0; index<node_offsets_host.size(); ++index)
        host_node_offsets(index) = node_offsets_host[index];
    for (std::size_t index=0; index<weight_offsets_host.size(); ++index)
        host_weight_offsets(index) = weight_offsets_host[index];
    for (std::size_t index=0; index<flat_weights.size(); ++index)
        host_weights(index) = flat_weights[index];
    Kokkos::deep_copy(this->shape, host_shape);
    Kokkos::deep_copy(node_offsets, host_node_offsets);
    Kokkos::deep_copy(weight_offsets, host_weight_offsets);
    Kokkos::deep_copy(this->weights, host_weights);
    this->activation_scale = activation_scale;
}

void MultilayerPerceptronKokkos::ensure_workspace(
    const int batch_size,
    const bool derivatives,
    const bool directional)
{
    if (batch_size < 0)
        throw std::invalid_argument(
            "MultilayerPerceptronKokkos batch size cannot be negative.");
    const auto capacity = std::max<std::size_t>(
        static_cast<std::size_t>(batch_size), node_values.extent(0));
    if (node_values.extent(0) < static_cast<std::size_t>(batch_size)
        || node_values.extent(1) != static_cast<std::size_t>(total_node_width))
        Kokkos::realloc(
            Kokkos::WithoutInitializing, node_values,
            capacity, total_node_width);
    if (derivatives
        && (node_derivatives.extent(0) < static_cast<std::size_t>(batch_size)
            || node_derivatives.extent(1)
                != static_cast<std::size_t>(total_node_width)))
        Kokkos::realloc(
            Kokkos::WithoutInitializing, node_derivatives,
            capacity, total_node_width);
    if (directional
        && (node_value_dots.extent(0) < static_cast<std::size_t>(batch_size)
            || node_value_dots.extent(1)
                != static_cast<std::size_t>(total_node_width)))
        Kokkos::realloc(
            Kokkos::WithoutInitializing, node_value_dots,
            capacity, total_node_width);
    if (directional
        && (node_derivative_dots.extent(0)
                < static_cast<std::size_t>(batch_size)
            || node_derivative_dots.extent(1)
                != static_cast<std::size_t>(total_node_width)))
        Kokkos::realloc(
            Kokkos::WithoutInitializing, node_derivative_dots,
            capacity, total_node_width);
}

void MultilayerPerceptronKokkos::evaluate(
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<double*,Kokkos::LayoutRight> f)
{
    evaluate(Kokkos::DefaultExecutionSpace(), x, f, true);
}

void MultilayerPerceptronKokkos::evaluate(
    const Kokkos::DefaultExecutionSpace& execution_space,
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    const bool completion_fence)
{
    if (shape_host.empty())
        throw std::logic_error(
            "MultilayerPerceptronKokkos is not initialized.");
    const int batch_size = x.extent(0);
    if (x.extent(1) != static_cast<std::size_t>(shape_host.front())
        || f.extent(0) < static_cast<std::size_t>(batch_size))
        throw std::invalid_argument(
            "MultilayerPerceptronKokkos input or output extent is invalid.");
    if (batch_size == 0)
        return;
    ensure_workspace(batch_size, false);

    const auto activation_scale = this->activation_scale;
    const auto node_values = this->node_values;
    const auto shape = this->shape;
    const auto node_offsets = this->node_offsets;
    const auto weight_offsets = this->weight_offsets;
    const auto weights = this->weights;
    const int layer_count = shape_host.size()-1;
    if constexpr (std::is_same_v<
            Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        Kokkos::parallel_for(
            "MultilayerPerceptronKokkos::evaluate_host",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                execution_space, 0, batch_size),
            KOKKOS_LAMBDA (const int i) {
                for (int input=0; input<shape(0); ++input)
                    node_values(i,node_offsets(0)+input) = x(i,input);
                for (int layer=0; layer<layer_count; ++layer) {
                    const int input_width = shape(layer);
                    const int output_width = shape(layer+1);
                    for (int output=0; output<output_width; ++output) {
                        double value = 0.0;
                        for (int input=0; input<input_width; ++input)
                            value += weights(
                                weight_offsets(layer)+output*input_width+input)
                                *node_values(i,node_offsets(layer)+input);
                        if (layer+1 < layer_count)
                            value = activation_scale*value
                                /(1.0+Kokkos::exp(-value));
                        node_values(i,node_offsets(layer+1)+output) = value;
                    }
                }
                f(i) = node_values(i,node_offsets(layer_count));
            });
    } else {
        Kokkos::parallel_for(
            "MultilayerPerceptronKokkos::evaluate",
            Kokkos::TeamPolicy<>(execution_space, batch_size, Kokkos::AUTO),
            KOKKOS_LAMBDA (
                const Kokkos::TeamPolicy<>::member_type& team_member) {
                const int i = team_member.league_rank();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, shape(0)),
                    [=] (const int input) {
                        node_values(i,node_offsets(0)+input) = x(i,input);
                    });
                team_member.team_barrier();
                for (int layer=0; layer<layer_count; ++layer) {
                    const int input_width = shape(layer);
                    const int output_width = shape(layer+1);
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, output_width),
                        [=] (const int output) {
                        double value = 0.0;
                        for (int input=0; input<input_width; ++input)
                            value += weights(
                                weight_offsets(layer)+output*input_width+input)
                                *node_values(i,node_offsets(layer)+input);
                        if (layer+1 < layer_count)
                            value = activation_scale*value
                                /(1.0+Kokkos::exp(-value));
                        node_values(i,node_offsets(layer+1)+output) = value;
                        });
                    team_member.team_barrier();
                }
                Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
                    f(i) = node_values(i,node_offsets(layer_count));
                });
            });
    }
    if (completion_fence)
        execution_space.fence("MultilayerPerceptronKokkos::evaluate");
}

void MultilayerPerceptronKokkos::evaluate_gradient(
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g)
{
    evaluate_gradient(Kokkos::DefaultExecutionSpace(), x, f, g, true);
}

void MultilayerPerceptronKokkos::evaluate_gradient(
    const Kokkos::DefaultExecutionSpace& execution_space,
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    const bool completion_fence)
{
    if (shape_host.empty())
        throw std::logic_error(
            "MultilayerPerceptronKokkos is not initialized.");
    const int batch_size = x.extent(0);
    if (x.extent(1) != static_cast<std::size_t>(shape_host.front())
        || f.extent(0) < static_cast<std::size_t>(batch_size)
        || g.extent(0) < static_cast<std::size_t>(batch_size)
        || g.extent(1) != static_cast<std::size_t>(shape_host.front()))
        throw std::invalid_argument(
            "MultilayerPerceptronKokkos gradient extent is invalid.");
    if (batch_size == 0)
        return;
    ensure_workspace(batch_size, true);

    const auto activation_scale = this->activation_scale;
    const auto node_derivatives = this->node_derivatives;
    const auto node_values = this->node_values;
    const auto shape = this->shape;
    const auto node_offsets = this->node_offsets;
    const auto weight_offsets = this->weight_offsets;
    const auto weights = this->weights;
    const int layer_count = shape_host.size()-1;

    if constexpr (std::is_same_v<
            Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>) {
        Kokkos::parallel_for(
            "MultilayerPerceptronKokkos::evaluate_gradient_host",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                execution_space, 0, batch_size),
            KOKKOS_LAMBDA (const int i) {
                for (int input=0; input<shape(0); ++input)
                    node_values(i,node_offsets(0)+input) = x(i,input);
                for (int layer=0; layer<layer_count; ++layer) {
                    const int input_width = shape(layer);
                    const int output_width = shape(layer+1);
                    for (int output=0; output<output_width; ++output) {
                        double value = 0.0;
                        for (int input=0; input<input_width; ++input)
                            value += weights(
                                weight_offsets(layer)+output*input_width+input)
                                *node_values(i,node_offsets(layer)+input);
                        if (layer+1 < layer_count)
                            value = activation_scale*value
                                /(1.0+Kokkos::exp(-value));
                        node_values(i,node_offsets(layer+1)+output) = value;
                    }
                }
                const int final_weight_layer = layer_count-1;
                const int final_input_layer = layer_count-1;
                for (int input=0; input<shape(final_input_layer); ++input)
                    node_derivatives(
                        i,node_offsets(final_input_layer)+input) = weights(
                            weight_offsets(final_weight_layer)+input);
                for (int layer=layer_count-2; layer>=0; --layer) {
                    const int input_width = shape(layer);
                    const int output_width = shape(layer+1);
                    for (int output=0; output<output_width; ++output) {
                        double preactivation = 0.0;
                        for (int input=0; input<input_width; ++input)
                            preactivation += weights(
                                weight_offsets(layer)+output*input_width+input)
                                *node_values(i,node_offsets(layer)+input);
                        const double sigmoid =
                            1.0/(1.0+Kokkos::exp(-preactivation));
                        node_derivatives(
                            i,node_offsets(layer+1)+output) *=
                            activation_scale*sigmoid
                            +activation_scale*preactivation*sigmoid
                                *(1.0-sigmoid);
                    }
                    for (int input=0; input<input_width; ++input) {
                        double value = 0.0;
                        for (int output=0; output<output_width; ++output)
                            value += weights(
                                weight_offsets(layer)+output*input_width+input)
                                *node_derivatives(
                                    i,node_offsets(layer+1)+output);
                        node_derivatives(i,node_offsets(layer)+input) = value;
                    }
                }
                for (int input=0; input<shape(0); ++input)
                    g(i,input) = node_derivatives(i,node_offsets(0)+input);
                f(i) = node_values(i,node_offsets(layer_count));
            });
    } else {
        Kokkos::parallel_for(
            "MultilayerPerceptronKokkos::evaluate_gradient",
            Kokkos::TeamPolicy<>(execution_space, batch_size, Kokkos::AUTO),
            KOKKOS_LAMBDA (
                const Kokkos::TeamPolicy<>::member_type& team_member) {
                const int i = team_member.league_rank();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, shape(0)),
                    [=] (const int input) {
                        node_values(i,node_offsets(0)+input) = x(i,input);
                    });
                team_member.team_barrier();

                for (int layer=0; layer<layer_count; ++layer) {
                    const int input_width = shape(layer);
                    const int output_width = shape(layer+1);
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, output_width),
                        [=] (const int output) {
                            double value = 0.0;
                            for (int input=0; input<input_width; ++input)
                                value += weights(
                                    weight_offsets(layer)
                                        +output*input_width+input)
                                    *node_values(i,node_offsets(layer)+input);
                            if (layer+1 < layer_count)
                                value = activation_scale*value
                                    /(1.0+Kokkos::exp(-value));
                            node_values(i,node_offsets(layer+1)+output) = value;
                        });
                    team_member.team_barrier();
                }

                const int final_weight_layer = layer_count-1;
                const int final_input_layer = layer_count-1;
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(
                        team_member, shape(final_input_layer)),
                    [=] (const int input) {
                        node_derivatives(
                            i,node_offsets(final_input_layer)+input) = weights(
                                weight_offsets(final_weight_layer)+input);
                    });
                team_member.team_barrier();

                for (int layer=layer_count-2; layer>=0; --layer) {
                    const int input_width = shape(layer);
                    const int output_width = shape(layer+1);
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, output_width),
                        [=] (const int output) {
                            double preactivation = 0.0;
                            for (int input=0; input<input_width; ++input)
                                preactivation += weights(
                                    weight_offsets(layer)
                                        +output*input_width+input)
                                    *node_values(i,node_offsets(layer)+input);
                            const double sigmoid =
                                1.0/(1.0+Kokkos::exp(-preactivation));
                            node_derivatives(
                                i,node_offsets(layer+1)+output) *=
                                activation_scale*sigmoid
                                +activation_scale*preactivation*sigmoid
                                    *(1.0-sigmoid);
                        });
                    team_member.team_barrier();
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team_member, input_width),
                        [=] (const int input) {
                            double value = 0.0;
                            for (int output=0; output<output_width; ++output)
                                value += weights(
                                    weight_offsets(layer)
                                        +output*input_width+input)
                                    *node_derivatives(
                                        i,node_offsets(layer+1)+output);
                            node_derivatives(i,node_offsets(layer)+input) = value;
                        });
                    team_member.team_barrier();
                }

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, shape(0)),
                    [=] (const int input) {
                        g(i,input) = node_derivatives(i,node_offsets(0)+input);
                    });
                Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
                    f(i) = node_values(i,node_offsets(layer_count));
                });
            });
    }
    if (completion_fence)
        execution_space.fence("MultilayerPerceptronKokkos::evaluate_gradient");
}

void MultilayerPerceptronKokkos::evaluate_gradient_accumulate_recompute(
    const Kokkos::DefaultExecutionSpace& execution_space,
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    const bool completion_fence)
{
    if (shape_host.size() != 3 || shape_host.back() != 1)
        throw std::logic_error(
            "Recomputed MLP gradients require one hidden layer and a scalar output.");
    const int batch_size = x.extent(0);
    const auto input_width = static_cast<std::size_t>(shape_host.front());
    if (x.extent(1) != input_width
        || f.extent(0) < static_cast<std::size_t>(batch_size)
        || g.extent(0) < static_cast<std::size_t>(batch_size)
        || g.extent(1) != input_width)
        throw std::invalid_argument(
            "Recomputed MLP gradient extent is invalid.");
    if (batch_size == 0)
        return;

    const auto activation_scale = this->activation_scale;
    const auto shape = this->shape;
    const auto weight_offsets = this->weight_offsets;
    const auto weights = this->weights;
    const int hidden_width = shape_host[1];
    using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using Member = TeamPolicy::member_type;
    using ScratchSpace = Member::scratch_memory_space;
    const auto scratch_bytes = admitted_team_scratch_bytes<>(
        "MultilayerPerceptronKokkos::evaluate_gradient_recompute",
        {static_cast<std::size_t>(hidden_width), sizeof(double)});
    auto policy = TeamPolicy(execution_space, batch_size, Kokkos::AUTO)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
    Kokkos::parallel_for(
        "MultilayerPerceptronKokkos::evaluate_gradient_recompute",
        policy,
        KOKKOS_LAMBDA (const Member& team_member) {
            const int batch = team_member.league_rank();
            Kokkos::View<double*,ScratchSpace,Kokkos::MemoryUnmanaged>
                preactivation(team_member.team_scratch(0), hidden_width);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, hidden_width),
                [=] (const int hidden) {
                    double value = 0.0;
                    for (int input=0; input<shape(0); ++input)
                        value += weights(
                            weight_offsets(0)+hidden*shape(0)+input)
                            *x(batch,input);
                    preactivation(hidden) = value;
                });
            team_member.team_barrier();
            double output = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, hidden_width),
                [=] (const int hidden, double& value_sum) {
                    const double value = preactivation(hidden);
                    const double sigmoid = 1.0/(1.0+Kokkos::exp(-value));
                    const double output_weight = weights(
                        weight_offsets(1)+hidden);
                    value_sum += output_weight*activation_scale*value*sigmoid;
                    preactivation(hidden) = output_weight*activation_scale
                        *(sigmoid+value*sigmoid*(1.0-sigmoid));
                },
                output);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, shape(0)),
                [=] (const int input) {
                    double derivative = 0.0;
                    for (int hidden=0; hidden<hidden_width; ++hidden)
                        derivative += preactivation(hidden)*weights(
                            weight_offsets(0)+hidden*shape(0)+input);
                    g(batch,input) = derivative;
                });
            Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
                f(batch) += output;
            });
        });
    if (completion_fence)
        execution_space.fence(
            "MultilayerPerceptronKokkos::evaluate_gradient_recompute");
}

void MultilayerPerceptronKokkos::evaluate_gradient_directional(
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<const double**,Kokkos::LayoutRight> x_dot,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    Kokkos::View<double**,Kokkos::LayoutRight> g_dot)
{
    evaluate_gradient_directional(
        Kokkos::DefaultExecutionSpace(), x, x_dot, f, g, g_dot, true);
}

void MultilayerPerceptronKokkos::evaluate_gradient_directional(
    const Kokkos::DefaultExecutionSpace& execution_space,
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<const double**,Kokkos::LayoutRight> x_dot,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    Kokkos::View<double**,Kokkos::LayoutRight> g_dot,
    const bool completion_fence)
{
    if (shape_host.empty())
        throw std::logic_error(
            "MultilayerPerceptronKokkos is not initialized.");
    const int batch_size = x.extent(0);
    const auto input_width = static_cast<std::size_t>(shape_host.front());
    if (x.extent(1) != input_width
        || x_dot.extent(0) != x.extent(0)
        || x_dot.extent(1) != x.extent(1))
        throw std::invalid_argument(
            "MultilayerPerceptronKokkos directional input shape mismatch.");
    if (f.extent(0) < static_cast<std::size_t>(batch_size)
        || g.extent(0) < static_cast<std::size_t>(batch_size)
        || g.extent(1) != input_width
        || g_dot.extent(0) < static_cast<std::size_t>(batch_size)
        || g_dot.extent(1) != input_width)
        throw std::invalid_argument(
            "MultilayerPerceptronKokkos directional output extent is invalid.");
    if (batch_size == 0)
        return;
    ensure_workspace(batch_size, true, true);

    const auto activation_scale = this->activation_scale;
    const auto shape = this->shape;
    const auto node_offsets = this->node_offsets;
    const auto weight_offsets = this->weight_offsets;
    const auto weights = this->weights;
    const auto values = this->node_values;
    const auto value_dots = this->node_value_dots;
    const auto derivatives = this->node_derivatives;
    const auto derivative_dots = this->node_derivative_dots;
    const int layer_count = shape_host.size()-1;

    Kokkos::parallel_for(
        "MultilayerPerceptronKokkos::evaluate_gradient_directional",
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, batch_size, Kokkos::AUTO),
        KOKKOS_LAMBDA (
            const Kokkos::TeamPolicy<>::member_type& team_member) {
            const int batch = team_member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, shape(0)),
                [=] (const int input) {
                    values(batch,node_offsets(0)+input) = x(batch,input);
                    value_dots(batch,node_offsets(0)+input) = x_dot(batch,input);
                });
            team_member.team_barrier();

            for (int layer=0; layer<layer_count; ++layer) {
                const int layer_input_width = shape(layer);
                const int layer_output_width = shape(layer+1);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, layer_output_width),
                    [=] (const int output) {
                        double z = 0.0;
                        double z_dot = 0.0;
                        for (int input=0; input<layer_input_width; ++input) {
                            const double weight = weights(
                                weight_offsets(layer)
                                    +output*layer_input_width+input);
                            z += weight*values(
                                batch,node_offsets(layer)+input);
                            z_dot += weight*value_dots(
                                batch,node_offsets(layer)+input);
                        }
                        if (layer+1 == layer_count) {
                            values(batch,node_offsets(layer+1)+output) = z;
                            value_dots(
                                batch,node_offsets(layer+1)+output) = z_dot;
                        } else {
                            const double sigmoid = 1.0/(1.0+Kokkos::exp(-z));
                            const double activation_derivative = activation_scale
                                *(sigmoid + z*sigmoid*(1.0-sigmoid));
                            values(batch,node_offsets(layer+1)+output) =
                                activation_scale*z*sigmoid;
                            value_dots(batch,node_offsets(layer+1)+output) =
                                activation_derivative*z_dot;
                        }
                    });
                team_member.team_barrier();
            }

            const int final_weight_layer = layer_count-1;
            const int final_input_layer = layer_count-1;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(
                    team_member, shape(final_input_layer)),
                [=] (const int input) {
                    derivatives(
                        batch,node_offsets(final_input_layer)+input) = weights(
                            weight_offsets(final_weight_layer)+input);
                    derivative_dots(
                        batch,node_offsets(final_input_layer)+input) = 0.0;
                });
            team_member.team_barrier();

            for (int layer=layer_count-2; layer>=0; --layer) {
                const int layer_input_width = shape(layer);
                const int layer_output_width = shape(layer+1);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, layer_output_width),
                    [=] (const int output) {
                        double z = 0.0;
                        double z_dot = 0.0;
                        for (int input=0; input<layer_input_width; ++input) {
                            const double weight = weights(
                                weight_offsets(layer)
                                    +output*layer_input_width+input);
                            z += weight*values(
                                batch,node_offsets(layer)+input);
                            z_dot += weight*value_dots(
                                batch,node_offsets(layer)+input);
                        }
                        const double sigmoid = 1.0/(1.0+Kokkos::exp(-z));
                        const double sigmoid_derivative = sigmoid*(1.0-sigmoid);
                        const double activation_derivative = activation_scale
                            *(sigmoid + z*sigmoid_derivative);
                        const double activation_second_derivative = activation_scale
                            *(2.0*sigmoid_derivative
                              +z*sigmoid_derivative*(1.0-2.0*sigmoid));
                        const int output_offset = node_offsets(layer+1)+output;
                        derivative_dots(batch,output_offset) =
                            derivative_dots(batch,output_offset)
                                *activation_derivative
                            +derivatives(batch,output_offset)
                                *activation_second_derivative*z_dot;
                        derivatives(batch,output_offset) *= activation_derivative;
                    });
                team_member.team_barrier();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, layer_input_width),
                    [=] (const int input) {
                        double derivative = 0.0;
                        double derivative_dot = 0.0;
                        for (int output=0; output<layer_output_width; ++output) {
                            const double weight = weights(
                                weight_offsets(layer)
                                    +output*layer_input_width+input);
                            derivative += weight*derivatives(
                                batch,node_offsets(layer+1)+output);
                            derivative_dot += weight*derivative_dots(
                                batch,node_offsets(layer+1)+output);
                        }
                        derivatives(batch,node_offsets(layer)+input) = derivative;
                        derivative_dots(batch,node_offsets(layer)+input) =
                            derivative_dot;
                    });
                team_member.team_barrier();
            }

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, shape(0)),
                [=] (const int input) {
                    g(batch,input) = derivatives(batch,node_offsets(0)+input);
                    g_dot(batch,input) = derivative_dots(
                        batch,node_offsets(0)+input);
                });
            Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
                f(batch) = values(batch,node_offsets(layer_count));
            });
        });

    if (completion_fence)
        execution_space.fence(
            "MultilayerPerceptronKokkos::evaluate_gradient_directional");
}

void MultilayerPerceptronKokkos::evaluate_gradient_directional_recompute(
    Kokkos::View<const double**,Kokkos::LayoutRight> x,
    Kokkos::View<const double**,Kokkos::LayoutRight> x_dot,
    Kokkos::View<double*,Kokkos::LayoutRight> f,
    Kokkos::View<double**,Kokkos::LayoutRight> g,
    Kokkos::View<double**,Kokkos::LayoutRight> g_dot)
{
    if (shape_host.size() != 3 || shape_host.back() != 1)
        throw std::logic_error(
            "Recomputed directional MLP gradients require one hidden layer "
            "and a scalar output.");
    const int batch_size = x.extent(0);
    const auto input_width = static_cast<std::size_t>(shape_host.front());
    if (x.extent(1) != input_width
        || x_dot.extent(0) != x.extent(0)
        || x_dot.extent(1) != x.extent(1)
        || f.extent(0) < static_cast<std::size_t>(batch_size)
        || g.extent(0) < static_cast<std::size_t>(batch_size)
        || g.extent(1) != input_width
        || g_dot.extent(0) < static_cast<std::size_t>(batch_size)
        || g_dot.extent(1) != input_width)
        throw std::invalid_argument(
            "Recomputed directional MLP gradient extent is invalid.");
    if (batch_size == 0)
        return;

    const auto activation_scale = this->activation_scale;
    const auto shape = this->shape;
    const auto weight_offsets = this->weight_offsets;
    const auto weights = this->weights;
    const int hidden_width = shape_host[1];
    using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using Member = TeamPolicy::member_type;
    using ScratchSpace = Member::scratch_memory_space;
    const auto scratch_bytes = admitted_team_scratch_bytes<>(
        "MultilayerPerceptronKokkos::evaluate_gradient_directional_recompute",
        {std::size_t(2), static_cast<std::size_t>(hidden_width), sizeof(double)});
    auto policy = TeamPolicy(batch_size, Kokkos::AUTO)
        .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
    Kokkos::parallel_for(
        "MultilayerPerceptronKokkos::evaluate_gradient_directional_recompute",
        policy,
        KOKKOS_LAMBDA (const Member& team_member) {
            const int batch = team_member.league_rank();
            Kokkos::View<double*,ScratchSpace,Kokkos::MemoryUnmanaged>
                state(team_member.team_scratch(0), 2*hidden_width);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, hidden_width),
                [=] (const int hidden) {
                    double value = 0.0;
                    double value_dot = 0.0;
                    for (int input=0; input<shape(0); ++input) {
                        const double weight = weights(
                            weight_offsets(0)+hidden*shape(0)+input);
                        value += weight*x(batch,input);
                        value_dot += weight*x_dot(batch,input);
                    }
                    state(hidden) = value;
                    state(hidden_width+hidden) = value_dot;
                });
            team_member.team_barrier();
            double output = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team_member, hidden_width),
                [=] (const int hidden, double& value_sum) {
                    const double value = state(hidden);
                    const double sigmoid = 1.0/(1.0+Kokkos::exp(-value));
                    const double sigmoid_derivative = sigmoid*(1.0-sigmoid);
                    const double output_weight = weights(
                        weight_offsets(1)+hidden);
                    value_sum += output_weight*activation_scale*value*sigmoid;
                    state(hidden) = output_weight*activation_scale
                        *(sigmoid+value*sigmoid_derivative);
                    state(hidden_width+hidden) *= output_weight*activation_scale
                        *(2.0*sigmoid_derivative
                          +value*sigmoid_derivative*(1.0-2.0*sigmoid));
                },
                output);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, shape(0)),
                [=] (const int input) {
                    double derivative = 0.0;
                    double derivative_dot = 0.0;
                    for (int hidden=0; hidden<hidden_width; ++hidden) {
                        const double input_weight = weights(
                            weight_offsets(0)+hidden*shape(0)+input);
                        derivative += input_weight*state(hidden);
                        derivative_dot += input_weight
                            *state(hidden_width+hidden);
                    }
                    g(batch,input) = derivative;
                    g_dot(batch,input) = derivative_dot;
                });
            Kokkos::single(Kokkos::PerTeam(team_member), [=] () {
                f(batch) = output;
            });
        });
    Kokkos::fence(
        "MultilayerPerceptronKokkos::evaluate_gradient_directional_recompute");
}

void MultilayerPerceptronKokkos::release_workspace()
{
    node_values = decltype(node_values)();
    node_derivatives = decltype(node_derivatives)();
    node_value_dots = decltype(node_value_dots)();
    node_derivative_dots = decltype(node_derivative_dots)();
}

std::size_t MultilayerPerceptronKokkos::workspace_bytes() const
{
    return sizeof(double)*(node_values.size()+node_derivatives.size()
        +node_value_dots.size()+node_derivative_dots.size());
}

std::size_t MultilayerPerceptronKokkos::estimated_workspace_bytes(
    const std::size_t batch_size,
    const bool derivatives,
    const bool directional) const
{
    const std::size_t arrays = 1+std::size_t(derivatives)
        +2*std::size_t(directional);
    const std::size_t width = static_cast<std::size_t>(total_node_width);
    if (width != 0
        && batch_size > std::numeric_limits<std::size_t>::max()/width)
        throw std::length_error("MLP workspace estimate overflows size_t.");
    const std::size_t elements = batch_size*width;
    if (arrays != 0
        && elements > std::numeric_limits<std::size_t>::max()/arrays)
        throw std::length_error("MLP workspace estimate overflows size_t.");
    const std::size_t total_elements = arrays*elements;
    if (total_elements > std::numeric_limits<std::size_t>::max()/sizeof(double))
        throw std::length_error("MLP workspace byte estimate overflows size_t.");
    return sizeof(double)*total_elements;
}
