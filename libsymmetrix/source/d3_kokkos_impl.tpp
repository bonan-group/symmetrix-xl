#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace symmetrix::dispersion {
namespace d3_detail {

inline constexpr std::size_t element_extent =
    static_cast<std::size_t>(d3_max_atomic_number)+1;
inline constexpr std::size_t reference_extent =
    static_cast<std::size_t>(d3_max_coordination_references);

template<class View>
void grow_view(View& view, const std::size_t extent, const char* label)
{
    if (view.extent(0) < extent)
        view = View(Kokkos::view_alloc(std::string(label)),extent);
}

template<class Precision>
struct InterpolationResult {
    Precision c6 = Precision(0);
    Precision receiver_derivative = Precision(0);
    Precision source_derivative = Precision(0);
};

KOKKOS_INLINE_FUNCTION
std::size_t c6_index(
    int zi, int zj, std::size_t receiver_reference,
    std::size_t source_reference, std::size_t field);

template<class Precision>
KOKKOS_INLINE_FUNCTION Precision coordination_distance(
    Precision receiver_delta, Precision source_delta, bool& overflowed);

template<class Precision>
struct NormalizedReferenceWeights {
    Precision value[d3_max_coordination_references];
    Precision derivative[d3_max_coordination_references];
    int fallback_reference = -1;
};

template<class Precision, class CountView, class ReferenceView>
KOKKOS_INLINE_FUNCTION NormalizedReferenceWeights<Precision>
normalized_reference_weights(
    const int atomic_number, const Precision coordination,
    const CountView& reference_counts,
    const ReferenceView& reference_coordination)
{
    NormalizedReferenceWeights<Precision> result{};
    const int count = reference_counts(atomic_number);
    Precision maximum_log_weight =
        -std::numeric_limits<Precision>::infinity();
    Precision nearest_distance = std::numeric_limits<Precision>::infinity();
    int nearest_reference = 0;
    bool requires_fallback = false;
    const Precision safe_delta = Kokkos::sqrt(
        std::numeric_limits<Precision>::max()/Precision(2));
    for (int reference=0; reference<count; ++reference) {
        const Precision coordinate = reference_coordination(
            static_cast<std::size_t>(atomic_number)*reference_extent
            +static_cast<std::size_t>(reference));
        const Precision delta = coordinate-coordination;
        Precision distance = std::numeric_limits<Precision>::max();
        if (Kokkos::abs(delta) > safe_delta) {
            requires_fallback = true;
        } else {
            distance = delta*delta;
            if (coordinate != Precision(0) && delta == -coordination)
                requires_fallback = true;
        }
        if (reference == 0 || distance < nearest_distance) {
            nearest_distance = distance;
            nearest_reference = reference;
        }
        const Precision log_weight = Precision(-4)*distance;
        if (log_weight > maximum_log_weight)
            maximum_log_weight = log_weight;
    }
    if (requires_fallback || !Kokkos::isfinite(maximum_log_weight)) {
        result.value[nearest_reference] = Precision(1);
        result.fallback_reference = nearest_reference;
        return result;
    }

    Precision denominator = Precision(0);
    Precision mean_log_derivative = Precision(0);
    for (int reference=0; reference<count; ++reference) {
        const Precision delta = reference_coordination(
            static_cast<std::size_t>(atomic_number)*reference_extent
            +static_cast<std::size_t>(reference))-coordination;
        const Precision weight = Kokkos::exp(
            Precision(-4)*delta*delta-maximum_log_weight);
        result.value[reference] = weight;
        denominator += weight;
        mean_log_derivative += weight*Precision(8)*delta;
    }
    if (!(denominator > Precision(0)) || !Kokkos::isfinite(denominator)) {
        result = {};
        result.value[nearest_reference] = Precision(1);
        result.fallback_reference = nearest_reference;
        return result;
    }
    mean_log_derivative /= denominator;
    for (int reference=0; reference<count; ++reference) {
        const Precision delta = reference_coordination(
            static_cast<std::size_t>(atomic_number)*reference_extent
            +static_cast<std::size_t>(reference))-coordination;
        result.value[reference] /= denominator;
        result.derivative[reference] = result.value[reference]
            *(Precision(8)*delta-mean_log_derivative);
    }
    return result;
}

template<class Precision, class C6View>
KOKKOS_INLINE_FUNCTION InterpolationResult<Precision> contract_c6(
    const int zi, const int zj, const int receiver_count,
    const int source_count, const Precision* receiver_weights,
    const Precision* receiver_derivatives, const Precision* source_weights,
    const Precision* source_derivatives, const C6View& table)
{
    InterpolationResult<Precision> result;
    if constexpr (std::is_same_v<Precision,float>) {
        // Centered rows prevent FP32 derivative cancellation. FP64 keeps the
        // faster joint contraction because it already meets tight tolerances.
        Precision row_c6[reference_extent]{};
        for (int receiver_reference=0;
                receiver_reference<receiver_count; ++receiver_reference) {
            const Precision receiver_weight =
                receiver_weights[receiver_reference];
            for (int source_reference=0;
                    source_reference<source_count; ++source_reference) {
                const Precision reference_c6 = table(c6_index(
                    zi,zj,static_cast<std::size_t>(receiver_reference),
                    static_cast<std::size_t>(source_reference),0));
                const Precision source_weight =
                    source_weights[source_reference];
                row_c6[receiver_reference] += source_weight*reference_c6;
            }
            result.c6 += receiver_weight*row_c6[receiver_reference];
        }
        for (int receiver_reference=0;
                receiver_reference<receiver_count; ++receiver_reference) {
            const Precision receiver_weight =
                receiver_weights[receiver_reference];
            Precision row_source_derivative = Precision(0);
            for (int source_reference=0;
                    source_reference<source_count; ++source_reference) {
                const Precision reference_c6 = table(c6_index(
                    zi,zj,static_cast<std::size_t>(receiver_reference),
                    static_cast<std::size_t>(source_reference),0));
                row_source_derivative += source_derivatives[source_reference]
                    *(reference_c6-row_c6[receiver_reference]);
            }
            result.source_derivative +=
                receiver_weight*row_source_derivative;
            result.receiver_derivative +=
                receiver_derivatives[receiver_reference]
                *(row_c6[receiver_reference]-result.c6);
        }
    } else {
        for (int receiver_reference=0;
                receiver_reference<receiver_count; ++receiver_reference) {
            for (int source_reference=0;
                    source_reference<source_count; ++source_reference) {
                const Precision reference_c6 = table(c6_index(
                    zi,zj,static_cast<std::size_t>(receiver_reference),
                    static_cast<std::size_t>(source_reference),0));
                const Precision receiver_weight =
                    receiver_weights[receiver_reference];
                const Precision source_weight =
                    source_weights[source_reference];
                result.c6 += receiver_weight*source_weight*reference_c6;
                result.receiver_derivative +=
                    receiver_derivatives[receiver_reference]
                    *source_weight*reference_c6;
                result.source_derivative += receiver_weight
                    *source_derivatives[source_reference]*reference_c6;
            }
        }
    }
    return result;
}

template<class Precision, class C6View>
KOKKOS_FUNCTION InterpolationResult<Precision> fallback_c6(
    const int zi, const int zj, const int receiver_count,
    const int source_count, const Precision receiver_coordination,
    const Precision source_coordination, const C6View& table)
{
    InterpolationResult<Precision> result;
    Precision nearest_distance = std::numeric_limits<Precision>::infinity();
    std::size_t nearest_index = std::numeric_limits<std::size_t>::max();
    for (int receiver_reference=0;
            receiver_reference<receiver_count; ++receiver_reference) {
        for (int source_reference=0;
                source_reference<source_count; ++source_reference) {
            const std::size_t base = c6_index(
                zi,zj,static_cast<std::size_t>(receiver_reference),
                static_cast<std::size_t>(source_reference),0);
            const Precision reference_c6 = table(base);
            if (!(reference_c6 > Precision(0)))
                continue;
            bool overflowed = false;
            const Precision distance = coordination_distance(
                table(base+1)-receiver_coordination,
                table(base+2)-source_coordination,overflowed);
            const std::size_t table_index =
                static_cast<std::size_t>(receiver_reference)*reference_extent
                +static_cast<std::size_t>(source_reference);
            if (nearest_index == std::numeric_limits<std::size_t>::max()
                    || distance < nearest_distance
                    || (distance == nearest_distance
                        && table_index < nearest_index)) {
                nearest_distance = distance;
                nearest_index = table_index;
                result.c6 = reference_c6;
            }
        }
    }
    return result;
}

template<class Precision>
KOKKOS_INLINE_FUNCTION Precision coordination_distance(
    const Precision receiver_delta, const Precision source_delta,
    bool& overflowed)
{
    const Precision safe_delta = Kokkos::sqrt(
        std::numeric_limits<Precision>::max()/Precision(2));
    if (Kokkos::abs(receiver_delta) > safe_delta
            || Kokkos::abs(source_delta) > safe_delta) {
        overflowed = true;
        return std::numeric_limits<Precision>::max();
    }
    return receiver_delta*receiver_delta+source_delta*source_delta;
}

KOKKOS_INLINE_FUNCTION
std::size_t c6_index(
    const int zi, const int zj, const std::size_t receiver_reference,
    const std::size_t source_reference, const std::size_t field)
{
    return (((static_cast<std::size_t>(zi)*element_extent
        +static_cast<std::size_t>(zj))*reference_extent
        +receiver_reference)*reference_extent+source_reference)*3+field;
}

template<class Precision, class View>
KOKKOS_INLINE_FUNCTION InterpolationResult<Precision> interpolate_c6(
    const int zi, const int zj, const Precision receiver_coordination,
    const Precision source_coordination, const View& table)
{
    constexpr Precision k3 = Precision(-4);
    Precision maximum_log_weight =
        -std::numeric_limits<Precision>::infinity();
    Precision nearest_distance = std::numeric_limits<Precision>::infinity();
    Precision nearest_c6 = Precision(0);
    std::size_t nearest_index = std::numeric_limits<std::size_t>::max();
    bool requires_fallback = false;

    for (std::size_t receiver_reference=0;
            receiver_reference<reference_extent; ++receiver_reference) {
        for (std::size_t source_reference=0;
                source_reference<reference_extent; ++source_reference) {
            const std::size_t base = c6_index(
                zi,zj,receiver_reference,source_reference,0);
            const Precision reference_c6 = table(base);
            if (!(reference_c6 > Precision(0)))
                continue;
            const Precision receiver_delta =
                table(base+1)-receiver_coordination;
            const Precision source_delta = table(base+2)-source_coordination;
            bool overflowed = false;
            if ((table(base+1) != Precision(0)
                        && receiver_delta == -receiver_coordination)
                    || (table(base+2) != Precision(0)
                        && source_delta == -source_coordination))
                overflowed = true;
            const Precision distance = coordination_distance(
                receiver_delta,source_delta,overflowed);
            requires_fallback = requires_fallback || overflowed;
            const Precision log_weight = k3*distance;
            if (log_weight > maximum_log_weight)
                maximum_log_weight = log_weight;
            const std::size_t table_index =
                receiver_reference*reference_extent+source_reference;
            if (nearest_index == std::numeric_limits<std::size_t>::max()
                    || distance < nearest_distance
                    || (distance == nearest_distance
                        && table_index < nearest_index)) {
                nearest_distance = distance;
                nearest_c6 = reference_c6;
                nearest_index = table_index;
            }
        }
    }

    InterpolationResult<Precision> result;
    if (nearest_index == std::numeric_limits<std::size_t>::max())
        return result;
    if (requires_fallback || !Kokkos::isfinite(maximum_log_weight)) {
        result.c6 = nearest_c6;
        return result;
    }

    Precision denominator = Precision(0);
    Precision c6_numerator = Precision(0);
    Precision receiver_log_derivative = Precision(0);
    Precision source_log_derivative = Precision(0);
    for (std::size_t receiver_reference=0;
            receiver_reference<reference_extent; ++receiver_reference) {
        for (std::size_t source_reference=0;
                source_reference<reference_extent; ++source_reference) {
            const std::size_t base = c6_index(
                zi,zj,receiver_reference,source_reference,0);
            const Precision reference_c6 = table(base);
            if (!(reference_c6 > Precision(0)))
                continue;
            const Precision receiver_delta =
                table(base+1)-receiver_coordination;
            const Precision source_delta = table(base+2)-source_coordination;
            bool overflowed = false;
            const Precision distance = coordination_distance(
                receiver_delta,source_delta,overflowed);
            const Precision weight = Kokkos::exp(
                k3*distance-maximum_log_weight);
            denominator += weight;
            c6_numerator += weight*reference_c6;
            receiver_log_derivative +=
                weight*(-Precision(2)*k3*receiver_delta);
            source_log_derivative +=
                weight*(-Precision(2)*k3*source_delta);
        }
    }
    if (!(denominator > Precision(0)) || !Kokkos::isfinite(denominator)) {
        result.c6 = nearest_c6;
        return result;
    }

    result.c6 = c6_numerator/denominator;
    receiver_log_derivative /= denominator;
    source_log_derivative /= denominator;
    for (std::size_t receiver_reference=0;
            receiver_reference<reference_extent; ++receiver_reference) {
        for (std::size_t source_reference=0;
                source_reference<reference_extent; ++source_reference) {
            const std::size_t base = c6_index(
                zi,zj,receiver_reference,source_reference,0);
            const Precision reference_c6 = table(base);
            if (!(reference_c6 > Precision(0)))
                continue;
            const Precision receiver_delta =
                table(base+1)-receiver_coordination;
            const Precision source_delta = table(base+2)-source_coordination;
            bool overflowed = false;
            const Precision distance = coordination_distance(
                receiver_delta,source_delta,overflowed);
            const Precision normalized_weight = Kokkos::exp(
                k3*distance-maximum_log_weight)/denominator;
            result.receiver_derivative += normalized_weight*reference_c6
                *((-Precision(2)*k3*receiver_delta)
                    -receiver_log_derivative);
            result.source_derivative += normalized_weight*reference_c6
                *((-Precision(2)*k3*source_delta)-source_log_derivative);
        }
    }
    return result;
}

template<class Precision>
struct PairEnergyFactor {
    Precision value = Precision(0);
    Precision radial_derivative = Precision(0);
};

template<class Precision>
KOKKOS_INLINE_FUNCTION PairEnergyFactor<Precision> pair_energy_factor(
    const D3DampingForm form, const Precision distance,
    const Precision pair_r0, const Precision pair_r2r4,
    const Precision s6, const Precision rs6, const Precision s8,
    const Precision rs8, const Precision alpha)
{
    const Precision distance2 = distance*distance;
    const Precision distance6 = distance2*distance2*distance2;
    const Precision distance8 = distance6*distance2;
    Precision e6 = Precision(0);
    Precision e8 = Precision(0);
    Precision e6_derivative = Precision(0);
    Precision e8_derivative = Precision(0);
    if (form == D3DampingForm::bj || form == D3DampingForm::bjm) {
        const Precision damping_radius =
            rs6*Kokkos::sqrt(Precision(3)*pair_r2r4)
            +rs8*Precision(bohr_to_angstrom);
        const Precision damping2 = damping_radius*damping_radius;
        const Precision damping6 = damping2*damping2*damping2;
        const Precision damping8 = damping6*damping2;
        e6 = Precision(1)/(distance6+damping6);
        e8 = Precision(1)/(distance8+damping8);
        e6_derivative = -Precision(6)*distance6/distance*e6*e6;
        e8_derivative = -Precision(8)*distance8/distance*e8*e8;
    } else if (form == D3DampingForm::zero) {
        const Precision ratio = pair_r0/distance;
        const Precision damping6 = Precision(1)/(
            Precision(1)+Precision(6)*Kokkos::pow(rs6*ratio,alpha));
        const Precision damping8 = Precision(1)/(
            Precision(1)+Precision(6)*Kokkos::pow(
                rs8*ratio,alpha+Precision(2)));
        e6 = damping6/distance6;
        e8 = damping8/distance8;
        e6_derivative = e6/distance*(
            alpha*(Precision(1)-damping6)-Precision(6));
        e8_derivative = e8/distance*((alpha+Precision(2))
            *(Precision(1)-damping8)-Precision(8));
    } else {
        const Precision ratio = distance/pair_r0;
        const Precision r0_beta = rs8*pair_r0/Precision(bohr_to_angstrom);
        const Precision damping6 = Precision(1)/(
            Precision(1)+Precision(6)*Kokkos::pow(
                ratio/rs6+r0_beta,-alpha));
        const Precision damping8 = Precision(1)/(
            Precision(1)+Precision(6)*Kokkos::pow(
                ratio+r0_beta,-alpha-Precision(2)));
        e6 = damping6/distance6;
        e8 = damping8/distance8;
        const Precision damping6_argument = ratio/rs6+r0_beta;
        const Precision damping8_argument = ratio+r0_beta;
        e6_derivative = e6*(alpha*(Precision(1)-damping6)
            /(pair_r0*rs6*damping6_argument)-Precision(6)/distance);
        e8_derivative = e8*((alpha+Precision(2))
            *(Precision(1)-damping8)/(pair_r0*damping8_argument)
            -Precision(8)/distance);
    }
    const Precision c8_ratio = Precision(3)*pair_r2r4;
    return {
        -Precision(0.5)*(s6*e6+s8*c8_ratio*e8),
        -Precision(0.5)*(
            s6*e6_derivative+s8*c8_ratio*e8_derivative)};
}

template<
    class Precision, class NumbersView, class OffsetsView, class SourcesView,
    class DistancesView, class ParameterView>
struct CoordinationTeamKernel {
    using member_type = typename Kokkos::TeamPolicy<
        Kokkos::DefaultExecutionSpace>::member_type;

    NumbersView numbers;
    OffsetsView offsets;
    SourcesView sources;
    DistancesView distances;
    ParameterView rcov;
    ParameterView coordination_numbers;
    ParameterView coordination_values;
    ParameterView coordination_radial_derivatives;
    Precision coordination_cutoff;

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type& team) const
    {
        const std::size_t receiver =
            static_cast<std::size_t>(team.league_rank());
        Precision coordination = Precision(0);
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(
                team,offsets(receiver),offsets(receiver+1)),
            [&] (const int edge, Precision& update) {
                const Precision distance =
                    static_cast<Precision>(distances(edge));
                Precision value = Precision(0);
                Precision radial_derivative = Precision(0);
                if (distance > Precision(0) && distance < coordination_cutoff) {
                    const int zi = numbers(receiver);
                    const int zj = numbers(sources(edge));
                    const Precision ratio = (rcov(zi)+rcov(zj))/distance;
                    value = Precision(1)/(
                        Precision(1)+Kokkos::exp(
                            Precision(-16)*(ratio-Precision(1))));
                    radial_derivative = -Precision(16)*ratio*value
                        *(Precision(1)-value)/distance;
                }
                coordination_values(edge) = value;
                coordination_radial_derivatives(edge) = radial_derivative;
                update += value;
            },coordination);
        team.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team),[&] () {
            coordination_numbers(receiver) = coordination;
        });
    }
};

template<
    class Precision, class NumbersView, class OffsetsView, class SourcesView,
    class DistancesView, class ParameterView>
struct CoordinationSerialOracleKernel {
    NumbersView numbers;
    OffsetsView offsets;
    SourcesView sources;
    DistancesView distances;
    ParameterView rcov;
    ParameterView coordination_numbers;
    ParameterView coordination_values;
    ParameterView coordination_radial_derivatives;
    Precision coordination_cutoff;

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t receiver) const
    {
        Precision coordination = Precision(0);
        const int zi = numbers(receiver);
        for (int edge=offsets(receiver); edge<offsets(receiver+1); ++edge) {
            const Precision distance =
                static_cast<Precision>(distances(edge));
            Precision value = Precision(0);
            Precision radial_derivative = Precision(0);
            if (distance > Precision(0) && distance < coordination_cutoff) {
                const int zj = numbers(sources(edge));
                const Precision ratio = (rcov(zi)+rcov(zj))/distance;
                value = Precision(1)/(
                    Precision(1)+Kokkos::exp(
                        Precision(-16)*(ratio-Precision(1))));
                radial_derivative = -Precision(16)*ratio*value
                    *(Precision(1)-value)/distance;
            }
            coordination_values(edge) = value;
            coordination_radial_derivatives(edge) = radial_derivative;
            coordination += value;
        }
        coordination_numbers(receiver) = coordination;
    }
};

template<
    class Precision, class NumbersView, class ParameterView, class CountView,
    class ReferenceWorkspaceView>
struct ReferenceWeightsKernel {
    NumbersView numbers;
    ParameterView coordination_numbers;
    CountView reference_counts;
    ParameterView reference_coordination;
    ReferenceWorkspaceView reference_weights;
    ReferenceWorkspaceView reference_derivatives;

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t atom) const
    {
        const auto normalized = normalized_reference_weights<Precision>(
            numbers(atom),coordination_numbers(atom),reference_counts,
            reference_coordination);
        for (std::size_t reference=0; reference<reference_extent; ++reference) {
            reference_weights(atom,reference) = normalized.value[reference];
            reference_derivatives(atom,reference) =
                normalized.derivative[reference];
        }
        if (normalized.fallback_reference >= 0)
            reference_weights(atom,0) = static_cast<Precision>(
                -normalized.fallback_reference-1);
    }
};

template<class Precision>
struct D3ReceiverAccumulation {
    double energy;
    Precision coordination_derivative;
    double force[3];
    double virial[6];

    KOKKOS_INLINE_FUNCTION
    void clear()
    {
        energy = 0.0;
        coordination_derivative = Precision(0);
        for (int axis=0; axis<3; ++axis)
            force[axis] = 0.0;
        for (int component=0; component<6; ++component)
            virial[component] = 0.0;
    }
};

template<
    class Precision, class NumbersView, class SourcesView, class DistancesView,
    class VectorsView, class CountView, class ReferenceView,
    class ParameterView>
struct PairEdgeContribution {
    using value_type = D3ReceiverAccumulation<Precision>;

    std::size_t receiver;
    int receiver_atomic_number;
    NumbersView numbers;
    SourcesView sources;
    DistancesView distances;
    VectorsView vectors;
    CountView reference_counts;
    ReferenceView reference_weights;
    ReferenceView reference_derivatives;
    ParameterView coordination_numbers;
    ParameterView c6;
    ParameterView r0;
    ParameterView r2r4;
    Precision pair_cutoff;
    D3DampingForm damping_form;
    Precision s6;
    Precision rs6;
    Precision s8;
    Precision rs8;
    Precision alpha;
    bool compute_forces;
    bool compute_stress;

    KOKKOS_INLINE_FUNCTION
    void init(value_type& update) const
    {
        update.clear();
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& destination, const value_type& source) const
    {
        destination.energy += source.energy;
        destination.coordination_derivative +=
            source.coordination_derivative;
        for (int axis=0; axis<3; ++axis)
            destination.force[axis] += source.force[axis];
        for (int component=0; component<6; ++component)
            destination.virial[component] += source.virial[component];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int edge, value_type& update) const
    {
        const Precision distance = static_cast<Precision>(distances(edge));
        if (!(distance > Precision(0)) || distance >= pair_cutoff)
            return;
        const int source = sources(edge);
        const int source_atomic_number = numbers(source);
        InterpolationResult<Precision> interpolation;
        if (reference_weights(receiver,0) < Precision(0)
                || reference_weights(source,0) < Precision(0)) {
            interpolation = fallback_c6(
                receiver_atomic_number,source_atomic_number,
                reference_counts(receiver_atomic_number),
                reference_counts(source_atomic_number),
                coordination_numbers(receiver),coordination_numbers(source),c6);
        } else {
            interpolation = contract_c6(
                receiver_atomic_number,source_atomic_number,
                reference_counts(receiver_atomic_number),
                reference_counts(source_atomic_number),
                &reference_weights(receiver,0),
                &reference_derivatives(receiver,0),
                &reference_weights(source,0),
                &reference_derivatives(source,0),c6);
        }
        const std::size_t pair =
            static_cast<std::size_t>(receiver_atomic_number)*element_extent
            +static_cast<std::size_t>(source_atomic_number);
        const auto factor = pair_energy_factor(
            damping_form,distance,r0(pair),
            r2r4(receiver_atomic_number)*r2r4(source_atomic_number),
            s6,rs6,s8,rs8,alpha);
        update.energy += static_cast<double>(factor.value*interpolation.c6);
        update.coordination_derivative += Precision(2)*factor.value
            *interpolation.receiver_derivative;
        if (!(compute_forces || compute_stress))
            return;
        const double force_scale = static_cast<double>(
            Precision(2)*factor.radial_derivative*interpolation.c6/distance);
        double edge_force[3];
        for (int axis=0; axis<3; ++axis) {
            edge_force[axis] = force_scale*vectors(edge,axis);
            if (compute_forces)
                update.force[axis] += edge_force[axis];
        }
        if (compute_stress) {
            update.virial[0] += 0.5*vectors(edge,0)*edge_force[0];
            update.virial[1] += 0.5*vectors(edge,1)*edge_force[1];
            update.virial[2] += 0.5*vectors(edge,2)*edge_force[2];
            update.virial[3] += 0.25*(
                vectors(edge,1)*edge_force[2]
                +vectors(edge,2)*edge_force[1]);
            update.virial[4] += 0.25*(
                vectors(edge,0)*edge_force[2]
                +vectors(edge,2)*edge_force[0]);
            update.virial[5] += 0.25*(
                vectors(edge,0)*edge_force[1]
                +vectors(edge,1)*edge_force[0]);
        }
    }
};

template<
    class Precision, class NumbersView, class OffsetsView, class SourcesView,
    class DistancesView, class VectorsView, class CountView,
    class ReferenceView, class ParameterView, class AtomicEnergyView,
    class DerivativeView, class ForceView, class VirialView>
struct PairTeamKernel {
    using member_type = typename Kokkos::TeamPolicy<
        Kokkos::DefaultExecutionSpace>::member_type;
    using edge_functor_type = PairEdgeContribution<
        Precision,NumbersView,SourcesView,DistancesView,VectorsView,CountView,
        ReferenceView,ParameterView>;

    NumbersView numbers;
    OffsetsView offsets;
    SourcesView sources;
    DistancesView distances;
    VectorsView vectors;
    CountView reference_counts;
    ReferenceView reference_weights;
    ReferenceView reference_derivatives;
    DerivativeView coordination_numbers;
    ParameterView c6;
    ParameterView r0;
    ParameterView r2r4;
    AtomicEnergyView atomic_energies;
    DerivativeView energy_coordination_derivatives;
    ForceView forces;
    VirialView virial;
    Precision pair_cutoff;
    D3DampingForm damping_form;
    Precision s6;
    Precision rs6;
    Precision s8;
    Precision rs8;
    Precision alpha;
    bool compute_forces;
    bool compute_stress;

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type& team) const
    {
        const std::size_t receiver =
            static_cast<std::size_t>(team.league_rank());
        D3ReceiverAccumulation<Precision> accumulation;
        const edge_functor_type edge_functor{
            receiver,numbers(receiver),numbers,sources,distances,vectors,
            reference_counts,reference_weights,reference_derivatives,
            coordination_numbers,c6,r0,
            r2r4,pair_cutoff,damping_form,s6,rs6,s8,rs8,alpha,
            compute_forces,compute_stress};
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(
                team,offsets(receiver),offsets(receiver+1)),
            edge_functor,accumulation);
        team.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team),[&] () {
            atomic_energies(receiver) = accumulation.energy;
            energy_coordination_derivatives(receiver) =
                accumulation.coordination_derivative;
            if (compute_forces)
                for (int axis=0; axis<3; ++axis)
                    forces(receiver,axis) = accumulation.force[axis];
            if (compute_stress)
                for (int component=0; component<6; ++component)
                    virial(receiver,component) = accumulation.virial[component];
        });
    }
};

template<
    class Precision, class NumbersView, class OffsetsView, class SourcesView,
    class DistancesView, class VectorsView, class CountView,
    class ReferenceView, class ParameterView, class AtomicEnergyView,
    class DerivativeView, class ForceView, class VirialView>
struct PairSerialOracleKernel {
    using edge_functor_type = PairEdgeContribution<
        Precision,NumbersView,SourcesView,DistancesView,VectorsView,CountView,
        ReferenceView,ParameterView>;

    NumbersView numbers;
    OffsetsView offsets;
    SourcesView sources;
    DistancesView distances;
    VectorsView vectors;
    CountView reference_counts;
    ReferenceView reference_weights;
    ReferenceView reference_derivatives;
    DerivativeView coordination_numbers;
    ParameterView c6;
    ParameterView r0;
    ParameterView r2r4;
    AtomicEnergyView atomic_energies;
    DerivativeView energy_coordination_derivatives;
    ForceView forces;
    VirialView virial;
    Precision pair_cutoff;
    D3DampingForm damping_form;
    Precision s6;
    Precision rs6;
    Precision s8;
    Precision rs8;
    Precision alpha;
    bool compute_forces;
    bool compute_stress;

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t receiver) const
    {
        D3ReceiverAccumulation<Precision> accumulation;
        accumulation.clear();
        const edge_functor_type edge_functor{
            receiver,numbers(receiver),numbers,sources,distances,vectors,
            reference_counts,reference_weights,reference_derivatives,
            coordination_numbers,c6,r0,
            r2r4,pair_cutoff,damping_form,s6,rs6,s8,rs8,alpha,
            compute_forces,compute_stress};
        for (int edge=offsets(receiver); edge<offsets(receiver+1); ++edge)
            edge_functor(edge,accumulation);
        atomic_energies(receiver) = accumulation.energy;
        energy_coordination_derivatives(receiver) =
            accumulation.coordination_derivative;
        if (compute_forces)
            for (int axis=0; axis<3; ++axis)
                forces(receiver,axis) = accumulation.force[axis];
        if (compute_stress)
            for (int component=0; component<6; ++component)
                virial(receiver,component) = accumulation.virial[component];
    }
};

struct AdjointReceiverAccumulation {
    double force[3];
    double virial[6];

    KOKKOS_INLINE_FUNCTION
    void clear()
    {
        for (int axis=0; axis<3; ++axis)
            force[axis] = 0.0;
        for (int component=0; component<6; ++component)
            virial[component] = 0.0;
    }
};

struct D3VirialSum {
    double value[6];
};

template<class VirialView, class StressView>
struct StressReductionKernel {
    using value_type = D3VirialSum;

    VirialView virial;
    StressView stress;
    double inverse_volume;

    KOKKOS_INLINE_FUNCTION
    void init(value_type& update) const
    {
        for (int component=0; component<6; ++component)
            update.value[component] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& destination, const value_type& source) const
    {
        for (int component=0; component<6; ++component)
            destination.value[component] += source.value[component];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t atom, value_type& update) const
    {
        for (int component=0; component<6; ++component)
            update.value[component] += virial(atom,component);
    }

    KOKKOS_INLINE_FUNCTION
    void final(value_type& update) const
    {
        for (int component=0; component<6; ++component)
            stress(component) = update.value[component]*inverse_volume;
    }
};

template<
    class Precision, class SourcesView, class DistancesView, class VectorsView,
    class DerivativeView, class ParameterView>
struct AdjointEdgeContribution {
    using value_type = AdjointReceiverAccumulation;

    std::size_t receiver;
    SourcesView sources;
    DistancesView distances;
    VectorsView vectors;
    DerivativeView energy_coordination_derivatives;
    ParameterView coordination_radial_derivatives;
    bool compute_forces;
    bool compute_stress;

    KOKKOS_INLINE_FUNCTION
    void init(value_type& update) const
    {
        update.clear();
    }

    KOKKOS_INLINE_FUNCTION
    void join(value_type& destination, const value_type& source) const
    {
        for (int axis=0; axis<3; ++axis)
            destination.force[axis] += source.force[axis];
        for (int component=0; component<6; ++component)
            destination.virial[component] += source.virial[component];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int edge, value_type& update) const
    {
        const Precision coordination_derivative =
            coordination_radial_derivatives(edge);
        if (coordination_derivative == Precision(0))
            return;
        const Precision distance = static_cast<Precision>(distances(edge));
        const int source = sources(edge);
        // The reverse directed edge supplies the source CN term. Combining
        // both node adjoints gives the complete pair radial derivative while
        // receiver-owned writes avoid force atomics.
        const Precision radial_derivative =
            (energy_coordination_derivatives(receiver)
                +energy_coordination_derivatives(source))
            *coordination_derivative;
        const double force_scale =
            static_cast<double>(radial_derivative/distance);
        double edge_force[3];
        for (int axis=0; axis<3; ++axis) {
            edge_force[axis] = force_scale*vectors(edge,axis);
            if (compute_forces)
                update.force[axis] += edge_force[axis];
        }
        if (compute_stress) {
            update.virial[0] += 0.5*vectors(edge,0)*edge_force[0];
            update.virial[1] += 0.5*vectors(edge,1)*edge_force[1];
            update.virial[2] += 0.5*vectors(edge,2)*edge_force[2];
            update.virial[3] += 0.25*(
                vectors(edge,1)*edge_force[2]
                +vectors(edge,2)*edge_force[1]);
            update.virial[4] += 0.25*(
                vectors(edge,0)*edge_force[2]
                +vectors(edge,2)*edge_force[0]);
            update.virial[5] += 0.25*(
                vectors(edge,0)*edge_force[1]
                +vectors(edge,1)*edge_force[0]);
        }
    }
};

template<
    class Precision, class OffsetsView, class SourcesView,
    class DistancesView, class VectorsView, class DerivativeView,
    class ParameterView, class ForceView, class VirialView>
struct AdjointTeamKernel {
    using member_type = typename Kokkos::TeamPolicy<
        Kokkos::DefaultExecutionSpace>::member_type;
    using edge_functor_type = AdjointEdgeContribution<
        Precision,SourcesView,DistancesView,VectorsView,DerivativeView,
        ParameterView>;

    OffsetsView offsets;
    SourcesView sources;
    DistancesView distances;
    VectorsView vectors;
    DerivativeView energy_coordination_derivatives;
    ParameterView coordination_radial_derivatives;
    ForceView forces;
    VirialView virial;
    bool compute_forces;
    bool compute_stress;

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type& team) const
    {
        const std::size_t receiver =
            static_cast<std::size_t>(team.league_rank());
        AdjointReceiverAccumulation accumulation;
        const edge_functor_type edge_functor{
            receiver,sources,distances,vectors,
            energy_coordination_derivatives,
            coordination_radial_derivatives,compute_forces,compute_stress};
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(
                team,offsets(receiver),offsets(receiver+1)),
            edge_functor,accumulation);
        team.team_barrier();
        Kokkos::single(Kokkos::PerTeam(team),[&] () {
            if (compute_forces)
                for (int axis=0; axis<3; ++axis)
                    forces(receiver,axis) += accumulation.force[axis];
            if (compute_stress)
                for (int component=0; component<6; ++component)
                    virial(receiver,component) +=
                        accumulation.virial[component];
        });
    }
};

template<
    class Precision, class OffsetsView, class SourcesView,
    class DistancesView, class VectorsView, class DerivativeView,
    class ParameterView, class ForceView, class VirialView>
struct AdjointSerialOracleKernel {
    using edge_functor_type = AdjointEdgeContribution<
        Precision,SourcesView,DistancesView,VectorsView,DerivativeView,
        ParameterView>;

    OffsetsView offsets;
    SourcesView sources;
    DistancesView distances;
    VectorsView vectors;
    DerivativeView energy_coordination_derivatives;
    ParameterView coordination_radial_derivatives;
    ForceView forces;
    VirialView virial;
    bool compute_forces;
    bool compute_stress;

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t receiver) const
    {
        AdjointReceiverAccumulation accumulation;
        accumulation.clear();
        const edge_functor_type edge_functor{
            receiver,sources,distances,vectors,
            energy_coordination_derivatives,
            coordination_radial_derivatives,compute_forces,compute_stress};
        for (int edge=offsets(receiver); edge<offsets(receiver+1); ++edge)
            edge_functor(edge,accumulation);
        if (compute_forces)
            for (int axis=0; axis<3; ++axis)
                forces(receiver,axis) += accumulation.force[axis];
        if (compute_stress)
            for (int component=0; component<6; ++component)
                virial(receiver,component) += accumulation.virial[component];
    }
};

} // namespace d3_detail

template<class Precision>
D3Kokkos<Precision>::D3Kokkos(
    std::string damping, std::string functional, const double pair_cutoff,
    const double coordination_cutoff)
    : damping_(std::move(damping)), functional_(std::move(functional)),
      pair_cutoff_(pair_cutoff), coordination_cutoff_(coordination_cutoff)
{
    if (!std::isfinite(pair_cutoff_) || !(pair_cutoff_ > 0.0))
        throw std::invalid_argument(
            "D3 pair cutoff must be finite and positive.");
    if (!std::isfinite(coordination_cutoff_)
            || !(coordination_cutoff_ > 0.0))
        throw std::invalid_argument(
            "D3 coordination cutoff must be finite and positive.");
    if (damping_ == "zero")
        damping_form_ = D3DampingForm::zero;
    else if (damping_ == "bj")
        damping_form_ = D3DampingForm::bj;
    else if (damping_ == "zerom")
        damping_form_ = D3DampingForm::zerom;
    else if (damping_ == "bjm")
        damping_form_ = D3DampingForm::bjm;
    else
        throw std::invalid_argument("Unsupported D3 damping form.");
    parameters_ = lookup_d3_parameters(damping_,functional_).parameters;

    const auto host_c6 = d3_c6_ev_angstrom6();
    const auto host_r0 = d3_r0_angstrom();
    const auto host_rcov = d3_rcov_angstrom();
    const auto host_r2r4 = d3_r2r4_angstrom();
    const auto host_reference_counts = d3_reference_counts();
    const auto host_reference_coordination = d3_reference_coordination();
    c6_ = ParameterView("D3::c6",host_c6.size());
    r0_ = ParameterView("D3::r0",host_r0.size());
    rcov_ = ParameterView("D3::rcov",host_rcov.size());
    r2r4_ = ParameterView("D3::r2r4",host_r2r4.size());
    reference_counts_ = CountView(
        "D3::reference_counts",host_reference_counts.size());
    reference_coordination_ = ParameterView(
        "D3::reference_coordination",host_reference_coordination.size());
    auto c6_mirror = Kokkos::create_mirror_view(c6_);
    auto r0_mirror = Kokkos::create_mirror_view(r0_);
    auto rcov_mirror = Kokkos::create_mirror_view(rcov_);
    auto r2r4_mirror = Kokkos::create_mirror_view(r2r4_);
    auto reference_counts_mirror =
        Kokkos::create_mirror_view(reference_counts_);
    auto reference_coordination_mirror =
        Kokkos::create_mirror_view(reference_coordination_);
    for (std::size_t index=0; index<host_c6.size(); ++index)
        c6_mirror(index) = static_cast<Precision>(host_c6[index]);
    for (std::size_t index=0; index<host_r0.size(); ++index)
        r0_mirror(index) = static_cast<Precision>(host_r0[index]);
    for (std::size_t index=0; index<host_rcov.size(); ++index)
        rcov_mirror(index) = static_cast<Precision>(host_rcov[index]);
    for (std::size_t index=0; index<host_r2r4.size(); ++index)
        r2r4_mirror(index) = static_cast<Precision>(host_r2r4[index]);
    for (std::size_t index=0; index<host_reference_counts.size(); ++index)
        reference_counts_mirror(index) = host_reference_counts[index];
    for (std::size_t index=0;
            index<host_reference_coordination.size(); ++index)
        reference_coordination_mirror(index) =
            static_cast<Precision>(host_reference_coordination[index]);
    Kokkos::deep_copy(c6_,c6_mirror);
    Kokkos::deep_copy(r0_,r0_mirror);
    Kokkos::deep_copy(rcov_,rcov_mirror);
    Kokkos::deep_copy(r2r4_,r2r4_mirror);
    Kokkos::deep_copy(reference_counts_,reference_counts_mirror);
    Kokkos::deep_copy(
        reference_coordination_,reference_coordination_mirror);
    result_.stress = decltype(result_.stress)("D3::stress");
}

template<class Precision>
void D3Kokkos<Precision>::require_computed() const
{
    if (!computed_)
        throw std::logic_error("D3 result is unavailable before compute.");
}

template<class Precision>
void D3Kokkos<Precision>::validate_build_cutoff(
    const double build_cutoff) const
{
    if (!std::isfinite(build_cutoff)
            || build_cutoff < std::max(pair_cutoff_,coordination_cutoff_))
        throw std::invalid_argument(
            "D3 graph build cutoff must cover pair and coordination cutoffs.");
}

template<class Precision>
void D3Kokkos<Precision>::validate_atomic_numbers(
    const std::span<const int> atomic_numbers) const
{
    for (const int atomic_number : atomic_numbers)
        if (atomic_number < 1 || atomic_number > d3_max_atomic_number)
            throw std::invalid_argument(
                "D3 atomic number is outside the parameter table.");
}

template<class Precision>
void D3Kokkos<Precision>::prepare_periodic_graph(
    const std::span<const int> atomic_numbers,
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const double build_cutoff)
{
    validate_build_cutoff(build_cutoff);
    validate_atomic_numbers(atomic_numbers);
    graph_.prepare_periodic(
        atomic_numbers,positions,cell,inverse_cell,build_cutoff);
    computed_ = false;
}

template<class Precision>
void D3Kokkos<Precision>::prepare_host_graph(
    const std::span<const int> atomic_numbers,
    const std::span<const int> receiver_offsets,
    const std::span<const int> sources,
    const std::span<const int> shifts,
    const std::span<const double> reference_positions,
    const std::span<const double> cell,
    const double build_cutoff)
{
    validate_build_cutoff(build_cutoff);
    validate_atomic_numbers(atomic_numbers);
    graph_.prepare_host(
        atomic_numbers,receiver_offsets,sources,shifts,reference_positions,
        cell,build_cutoff);
    computed_ = false;
}

template<class Precision>
void D3Kokkos<Precision>::update_geometry(
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell)
{
    graph_.update_geometry(positions,cell,inverse_cell);
    computed_ = false;
}

template<class Precision>
void D3Kokkos<Precision>::ensure_workspace_capacity(
    const std::size_t nodes, const std::size_t edges)
{
    const auto grow = [this] (auto& view, const std::size_t extent,
                              const char* label) {
        if (view.extent(0) < extent) {
            d3_detail::grow_view(view,extent,label);
            ++workspace_allocation_count_;
        }
    };
    grow(coordination_numbers_storage_,nodes,"D3::coordination_numbers");
    grow(
        energy_coordination_derivatives_storage_,nodes,
        "D3::energy_coordination_derivatives");
    grow(
        coordination_values_storage_,edges,"D3::coordination_values");
    grow(
        coordination_radial_derivatives_storage_,edges,
        "D3::coordination_radial_derivatives");
    grow(
        reference_weights_storage_,nodes,"D3::reference_weights");
    grow(
        reference_derivatives_storage_,nodes,"D3::reference_derivatives");
    grow(
        atomic_energies_storage_,nodes,"D3::atomic_energies");
    grow(forces_storage_,nodes,"D3::forces");
    grow(virial_storage_,nodes,"D3::virial");
    result_.atomic_energies = Kokkos::subview(
        atomic_energies_storage_,std::make_pair(std::size_t(0),nodes));
    result_.forces = Kokkos::subview(
        forces_storage_,std::make_pair(std::size_t(0),nodes),Kokkos::ALL());
}

template<class Precision>
void D3Kokkos<Precision>::compute_coordination_team()
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using policy_type = Kokkos::TeamPolicy<execution_space>;

    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    const auto numbers = graph_.atomic_numbers();
    const auto offsets = graph_.receiver_offsets();
    const auto sources = graph_.sources();
    const auto distances = graph_.distances();
    const auto rcov = rcov_;
    const auto coordination_numbers = Kokkos::subview(
        coordination_numbers_storage_,std::make_pair(std::size_t(0),nodes));
    const auto coordination_values = coordination_values_storage_;
    const auto coordination_radial_derivatives =
        coordination_radial_derivatives_storage_;
    const Precision coordination_cutoff =
        static_cast<Precision>(coordination_cutoff_);
    const auto coordination_kernel = d3_detail::CoordinationTeamKernel<
        Precision,decltype(numbers),decltype(offsets),decltype(sources),
        decltype(distances),ParameterView>{
            numbers,offsets,sources,distances,rcov,coordination_numbers,
            coordination_values,coordination_radial_derivatives,
            coordination_cutoff};
    const policy_type sizing_policy(nodes,Kokkos::AUTO());
    const int team_size = sizing_policy.team_size_recommended(
        coordination_kernel,Kokkos::ParallelForTag());
    Kokkos::parallel_for(
        "D3::coordination_numbers_team",policy_type(nodes,team_size),
        coordination_kernel);
}

template<class Precision>
void D3Kokkos<Precision>::compute_coordination_serial_oracle()
{
    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    const auto numbers = graph_.atomic_numbers();
    const auto offsets = graph_.receiver_offsets();
    const auto sources = graph_.sources();
    const auto distances = graph_.distances();
    const auto rcov = rcov_;
    const auto coordination_numbers = Kokkos::subview(
        coordination_numbers_storage_,std::make_pair(std::size_t(0),nodes));
    const auto coordination_values = coordination_values_storage_;
    const auto coordination_radial_derivatives =
        coordination_radial_derivatives_storage_;
    const Precision coordination_cutoff =
        static_cast<Precision>(coordination_cutoff_);
    const auto coordination_kernel = d3_detail::CoordinationSerialOracleKernel<
        Precision,decltype(numbers),decltype(offsets),decltype(sources),
        decltype(distances),ParameterView>{
            numbers,offsets,sources,distances,rcov,coordination_numbers,
            coordination_values,coordination_radial_derivatives,
            coordination_cutoff};
    Kokkos::parallel_for(
        "D3::coordination_numbers_serial_oracle",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nodes),
        coordination_kernel);
}

template<class Precision>
void D3Kokkos<Precision>::compute_coordination_diagnostics(const bool use_team)
{
    if (!graph_.prepared())
        throw std::logic_error(
            "D3 graph must be prepared before coordination diagnostics.");
    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    ensure_workspace_capacity(nodes,graph_.num_edges());
    if (use_team)
        compute_coordination_team();
    else
        compute_coordination_serial_oracle();
}

template<class Precision>
void D3Kokkos<Precision>::compute_reference_weights()
{
    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    const auto numbers = graph_.atomic_numbers();
    const auto coordination_numbers = Kokkos::subview(
        coordination_numbers_storage_,std::make_pair(std::size_t(0),nodes));
    const auto reference_counts = reference_counts_;
    const auto reference_coordination = reference_coordination_;
    const auto reference_weights = reference_weights_storage_;
    const auto reference_derivatives = reference_derivatives_storage_;
    const auto reference_kernel = d3_detail::ReferenceWeightsKernel<
        Precision,decltype(numbers),ParameterView,CountView,
        ReferenceWorkspaceView>{
            numbers,coordination_numbers,reference_counts,
            reference_coordination,reference_weights,reference_derivatives};
    Kokkos::parallel_for(
        "D3::reference_weights",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nodes),
        reference_kernel);
}

template<class Precision>
void D3Kokkos<Precision>::compute_pair_team(
    const bool compute_forces, const bool compute_stress)
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using policy_type = Kokkos::TeamPolicy<execution_space>;

    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    const auto numbers = graph_.atomic_numbers();
    const auto offsets = graph_.receiver_offsets();
    const auto sources = graph_.sources();
    const auto distances = graph_.distances();
    const auto vectors = graph_.edge_vectors();
    const auto reference_counts = reference_counts_;
    const auto reference_weights = reference_weights_storage_;
    const auto reference_derivatives = reference_derivatives_storage_;
    const auto atomic_energies = result_.atomic_energies;
    const auto coordination_numbers = Kokkos::subview(
        coordination_numbers_storage_,std::make_pair(std::size_t(0),nodes));
    const auto energy_coordination_derivatives = Kokkos::subview(
        energy_coordination_derivatives_storage_,
        std::make_pair(std::size_t(0),nodes));
    const auto forces = result_.forces;
    const auto virial = virial_storage_;
    const auto pair_kernel = d3_detail::PairTeamKernel<
        Precision,decltype(numbers),decltype(offsets),decltype(sources),
        decltype(distances),decltype(vectors),decltype(reference_counts),
        decltype(reference_weights),ParameterView,decltype(atomic_energies),
        decltype(energy_coordination_derivatives),decltype(forces),
        decltype(virial)>{
            numbers,offsets,sources,distances,vectors,reference_counts,
            reference_weights,reference_derivatives,coordination_numbers,
            c6_,r0_,r2r4_,
            atomic_energies,energy_coordination_derivatives,forces,virial,
            static_cast<Precision>(pair_cutoff_),damping_form_,
            static_cast<Precision>(parameters_.s6),
            static_cast<Precision>(parameters_.rs6),
            static_cast<Precision>(parameters_.s18),
            static_cast<Precision>(parameters_.rs18),
            static_cast<Precision>(parameters_.alp),compute_forces,
            compute_stress};
    const policy_type sizing_policy(nodes,Kokkos::AUTO());
    const int team_size = sizing_policy.team_size_recommended(
        pair_kernel,Kokkos::ParallelForTag());
    Kokkos::parallel_for(
        "D3::pair_energy_team",policy_type(nodes,team_size),pair_kernel);
}

template<class Precision>
void D3Kokkos<Precision>::compute_pair_serial_oracle(
    const bool compute_forces, const bool compute_stress)
{
    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    const auto numbers = graph_.atomic_numbers();
    const auto offsets = graph_.receiver_offsets();
    const auto sources = graph_.sources();
    const auto distances = graph_.distances();
    const auto vectors = graph_.edge_vectors();
    const auto reference_counts = reference_counts_;
    const auto reference_weights = reference_weights_storage_;
    const auto reference_derivatives = reference_derivatives_storage_;
    const auto atomic_energies = result_.atomic_energies;
    const auto coordination_numbers = Kokkos::subview(
        coordination_numbers_storage_,std::make_pair(std::size_t(0),nodes));
    const auto energy_coordination_derivatives = Kokkos::subview(
        energy_coordination_derivatives_storage_,
        std::make_pair(std::size_t(0),nodes));
    const auto forces = result_.forces;
    const auto virial = virial_storage_;
    const auto pair_kernel = d3_detail::PairSerialOracleKernel<
        Precision,decltype(numbers),decltype(offsets),decltype(sources),
        decltype(distances),decltype(vectors),decltype(reference_counts),
        decltype(reference_weights),ParameterView,decltype(atomic_energies),
        decltype(energy_coordination_derivatives),decltype(forces),
        decltype(virial)>{
            numbers,offsets,sources,distances,vectors,reference_counts,
            reference_weights,reference_derivatives,coordination_numbers,
            c6_,r0_,r2r4_,
            atomic_energies,energy_coordination_derivatives,forces,virial,
            static_cast<Precision>(pair_cutoff_),damping_form_,
            static_cast<Precision>(parameters_.s6),
            static_cast<Precision>(parameters_.rs6),
            static_cast<Precision>(parameters_.s18),
            static_cast<Precision>(parameters_.rs18),
            static_cast<Precision>(parameters_.alp),compute_forces,
            compute_stress};
    Kokkos::parallel_for(
        "D3::pair_energy_serial_oracle",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nodes),
        pair_kernel);
}

template<class Precision>
void D3Kokkos<Precision>::compute_pair_diagnostics(
    const bool use_team, const bool compute_forces, const bool compute_stress)
{
    if (!graph_.prepared())
        throw std::logic_error(
            "D3 graph must be prepared before pair diagnostics.");
    computed_ = false;
    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    ensure_workspace_capacity(nodes,graph_.num_edges());
    const auto coordination_numbers = Kokkos::subview(
        coordination_numbers_storage_,std::make_pair(std::size_t(0),nodes));
    const auto energy_coordination_derivatives = Kokkos::subview(
        energy_coordination_derivatives_storage_,
        std::make_pair(std::size_t(0),nodes));
    const auto active_virial = Kokkos::subview(
        virial_storage_,std::make_pair(std::size_t(0),nodes),Kokkos::ALL());
    Kokkos::deep_copy(coordination_numbers,Precision(0));
    Kokkos::deep_copy(energy_coordination_derivatives,Precision(0));
    Kokkos::deep_copy(result_.atomic_energies,0.0);
    Kokkos::deep_copy(result_.forces,0.0);
    Kokkos::deep_copy(active_virial,0.0);
    compute_coordination_team();
    compute_reference_weights();
    if (use_team)
        compute_pair_team(compute_forces,compute_stress);
    else
        compute_pair_serial_oracle(compute_forces,compute_stress);
    Kokkos::DefaultExecutionSpace().fence("Symmetrix D3 pair diagnostics");
}

template<class Precision>
void D3Kokkos<Precision>::compute_adjoint_team(
    const bool compute_forces, const bool compute_stress)
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using policy_type = Kokkos::TeamPolicy<execution_space>;

    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    const auto offsets = graph_.receiver_offsets();
    const auto sources = graph_.sources();
    const auto distances = graph_.distances();
    const auto vectors = graph_.edge_vectors();
    const auto energy_coordination_derivatives = Kokkos::subview(
        energy_coordination_derivatives_storage_,
        std::make_pair(std::size_t(0),nodes));
    const auto forces = result_.forces;
    const auto virial = virial_storage_;
    const auto adjoint_kernel = d3_detail::AdjointTeamKernel<
        Precision,decltype(offsets),decltype(sources),decltype(distances),
        decltype(vectors),decltype(energy_coordination_derivatives),
        ParameterView,decltype(forces),decltype(virial)>{
            offsets,sources,distances,vectors,
            energy_coordination_derivatives,
            coordination_radial_derivatives_storage_,forces,virial,
            compute_forces,compute_stress};
    const policy_type sizing_policy(nodes,Kokkos::AUTO());
    const int team_size = sizing_policy.team_size_recommended(
        adjoint_kernel,Kokkos::ParallelForTag());
    Kokkos::parallel_for(
        "D3::coordination_adjoint_team",policy_type(nodes,team_size),
        adjoint_kernel);
}

template<class Precision>
void D3Kokkos<Precision>::compute_adjoint_serial_oracle(
    const bool compute_forces, const bool compute_stress)
{
    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    const auto offsets = graph_.receiver_offsets();
    const auto sources = graph_.sources();
    const auto distances = graph_.distances();
    const auto vectors = graph_.edge_vectors();
    const auto energy_coordination_derivatives = Kokkos::subview(
        energy_coordination_derivatives_storage_,
        std::make_pair(std::size_t(0),nodes));
    const auto forces = result_.forces;
    const auto virial = virial_storage_;
    const auto adjoint_kernel = d3_detail::AdjointSerialOracleKernel<
        Precision,decltype(offsets),decltype(sources),decltype(distances),
        decltype(vectors),decltype(energy_coordination_derivatives),
        ParameterView,decltype(forces),decltype(virial)>{
            offsets,sources,distances,vectors,
            energy_coordination_derivatives,
            coordination_radial_derivatives_storage_,forces,virial,
            compute_forces,compute_stress};
    Kokkos::parallel_for(
        "D3::coordination_adjoint_serial_oracle",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nodes),
        adjoint_kernel);
}

template<class Precision>
void D3Kokkos<Precision>::compute_adjoint_diagnostics(const bool use_team)
{
    if (!graph_.prepared())
        throw std::logic_error(
            "D3 graph must be prepared before adjoint diagnostics.");
    computed_ = false;
    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    ensure_workspace_capacity(nodes,graph_.num_edges());
    const auto coordination_numbers = Kokkos::subview(
        coordination_numbers_storage_,std::make_pair(std::size_t(0),nodes));
    const auto energy_coordination_derivatives = Kokkos::subview(
        energy_coordination_derivatives_storage_,
        std::make_pair(std::size_t(0),nodes));
    const auto active_virial = Kokkos::subview(
        virial_storage_,std::make_pair(std::size_t(0),nodes),Kokkos::ALL());
    Kokkos::deep_copy(coordination_numbers,Precision(0));
    Kokkos::deep_copy(energy_coordination_derivatives,Precision(0));
    Kokkos::deep_copy(result_.atomic_energies,0.0);
    Kokkos::deep_copy(result_.forces,0.0);
    Kokkos::deep_copy(active_virial,0.0);
    compute_coordination_team();
    compute_reference_weights();
    compute_pair_team(true,true);
    if (use_team)
        compute_adjoint_team(true,true);
    else
        compute_adjoint_serial_oracle(true,true);
    Kokkos::DefaultExecutionSpace().fence("Symmetrix D3 adjoint diagnostics");
}

template<class Precision>
void D3Kokkos<Precision>::compute(
    const bool compute_forces, const bool compute_stress)
{
    computed_ = false;
    forces_available_ = false;
    stress_available_ = false;
    if (!graph_.prepared())
        throw std::logic_error("D3 graph must be prepared before compute.");
    double volume = 0.0;
    if (compute_stress) {
        volume = graph_.cell_volume();
        if (!std::isfinite(volume) || !(volume > 0.0))
            throw std::invalid_argument(
                "D3 stress requires a finite positive cell volume.");
    }
    const std::size_t nodes = static_cast<std::size_t>(graph_.num_nodes());
    const std::size_t edges = graph_.num_edges();
    ensure_workspace_capacity(nodes,edges);
    const auto coordination_numbers = Kokkos::subview(
        coordination_numbers_storage_,std::make_pair(std::size_t(0),nodes));
    const auto energy_coordination_derivatives = Kokkos::subview(
        energy_coordination_derivatives_storage_,
        std::make_pair(std::size_t(0),nodes));

    compute_coordination_team();
    coordination_production_path_ = CoordinationProductionPath::team;
    ++coordination_team_launch_count_;
    compute_reference_weights();
    const auto atomic_energies = result_.atomic_energies;
    const auto forces = result_.forces;
    const auto virial = virial_storage_;
    compute_pair_team(compute_forces,compute_stress);

    if (compute_forces || compute_stress)
        compute_adjoint_team(compute_forces,compute_stress);

    double energy = 0.0;
    Kokkos::parallel_reduce(
        "D3::total_energy",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nodes),
        KOKKOS_LAMBDA (const std::size_t atom, double& sum) {
            sum += atomic_energies(atom);
        },energy);
    result_.energy = energy;

    if (compute_stress) {
        const auto stress = result_.stress;
        Kokkos::parallel_reduce(
            "D3::stress",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nodes),
            d3_detail::StressReductionKernel<decltype(virial),decltype(stress)>{
                virial,stress,1.0/volume});
        ++stress_reduction_launch_count_;
    }
    Kokkos::DefaultExecutionSpace().fence("Symmetrix D3 compute");
    computed_ = true;
    forces_available_ = compute_forces;
    stress_available_ = compute_stress;
}

template<class Precision>
std::array<double,6> D3Kokkos<Precision>::interpolation_diagnostics(
    const int receiver_atomic_number, const int source_atomic_number,
    const double receiver_coordination, const double source_coordination) const
{
    if (receiver_atomic_number < 1
            || receiver_atomic_number > d3_max_atomic_number
            || source_atomic_number < 1
            || source_atomic_number > d3_max_atomic_number)
        throw std::invalid_argument(
            "D3 interpolation atomic number is outside the parameter table.");
    Kokkos::View<Precision[6]> output("D3::interpolation_diagnostics");
    const auto c6 = c6_;
    const auto reference_counts = reference_counts_;
    const auto reference_coordination = reference_coordination_;
    const Precision receiver =
        static_cast<Precision>(receiver_coordination);
    const Precision source = static_cast<Precision>(source_coordination);
    Kokkos::parallel_for(
        "D3::interpolation_diagnostics",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1),
        KOKKOS_LAMBDA (const int) {
            const auto interpolation = d3_detail::interpolate_c6(
                receiver_atomic_number,source_atomic_number,
                receiver,source,c6);
            const auto receiver_normalized =
                d3_detail::normalized_reference_weights<Precision>(
                    receiver_atomic_number,receiver,reference_counts,
                    reference_coordination);
            const auto source_normalized =
                d3_detail::normalized_reference_weights<Precision>(
                    source_atomic_number,source,reference_counts,
                    reference_coordination);
            d3_detail::InterpolationResult<Precision> factorized;
            if (receiver_normalized.fallback_reference >= 0
                    || source_normalized.fallback_reference >= 0) {
                factorized = d3_detail::fallback_c6(
                    receiver_atomic_number,source_atomic_number,
                    reference_counts(receiver_atomic_number),
                    reference_counts(source_atomic_number),receiver,source,c6);
            } else {
                factorized = d3_detail::contract_c6(
                    receiver_atomic_number,source_atomic_number,
                    reference_counts(receiver_atomic_number),
                    reference_counts(source_atomic_number),
                    receiver_normalized.value,receiver_normalized.derivative,
                    source_normalized.value,source_normalized.derivative,c6);
            }
            output(0) = interpolation.c6;
            output(1) = interpolation.receiver_derivative;
            output(2) = interpolation.source_derivative;
            output(3) = factorized.c6;
            output(4) = factorized.receiver_derivative;
            output(5) = factorized.source_derivative;
        });
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),output);
    return {
        static_cast<double>(host(0)),static_cast<double>(host(1)),
        static_cast<double>(host(2)),static_cast<double>(host(3)),
        static_cast<double>(host(4)),static_cast<double>(host(5))};
}

template<class Precision>
double D3Kokkos<Precision>::pair_energy_diagnostics(
    const int receiver_atomic_number, const int source_atomic_number,
    const double distance, const double receiver_coordination,
    const double source_coordination) const
{
    if (receiver_atomic_number < 1
            || receiver_atomic_number > d3_max_atomic_number
            || source_atomic_number < 1
            || source_atomic_number > d3_max_atomic_number)
        throw std::invalid_argument(
            "D3 pair-energy atomic number is outside the parameter table.");
    if (!std::isfinite(distance) || !(distance > 0.0))
        throw std::invalid_argument(
            "D3 pair-energy distance must be finite and positive.");
    Kokkos::View<Precision> output("D3::pair_energy_diagnostics");
    const auto c6 = c6_;
    const auto r0 = r0_;
    const auto r2r4 = r2r4_;
    const D3DampingForm damping_form = damping_form_;
    const Precision device_distance = static_cast<Precision>(distance);
    const Precision receiver =
        static_cast<Precision>(receiver_coordination);
    const Precision source = static_cast<Precision>(source_coordination);
    const Precision s6 = static_cast<Precision>(parameters_.s6);
    const Precision rs6 = static_cast<Precision>(parameters_.rs6);
    const Precision s8 = static_cast<Precision>(parameters_.s18);
    const Precision rs8 = static_cast<Precision>(parameters_.rs18);
    const Precision alpha = static_cast<Precision>(parameters_.alp);
    Kokkos::parallel_for(
        "D3::pair_energy_diagnostics",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1),
        KOKKOS_LAMBDA (const int) {
            const auto interpolation = d3_detail::interpolate_c6(
                receiver_atomic_number,source_atomic_number,
                receiver,source,c6);
            const std::size_t pair =
                static_cast<std::size_t>(receiver_atomic_number)
                *d3_detail::element_extent
                +static_cast<std::size_t>(source_atomic_number);
            const auto factor = d3_detail::pair_energy_factor(
                damping_form,device_distance,r0(pair),
                r2r4(receiver_atomic_number)*r2r4(source_atomic_number),
                s6,rs6,s8,rs8,alpha);
            output() = Precision(2)*factor.value*interpolation.c6;
        });
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),output);
    return static_cast<double>(host());
}

template<class Precision>
const DispersionResultKokkos& D3Kokkos<Precision>::result() const
{
    require_computed();
    return result_;
}

} // namespace symmetrix::dispersion
