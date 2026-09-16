#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

template <typename Precision>
class RadialFunctionSetKokkos
{
public:

    struct EvaluationPoint {
        int interval;
        double x;
        double xx;
        double xxx;
    };

    struct CachedEvaluationPoint {
        int interval;
        Precision x;
        Precision xx;
        Precision xxx;
    };

    RadialFunctionSetKokkos();
    RadialFunctionSetKokkos(
        double h,
        std::vector<std::vector<std::vector<double>>> node_values,
        std::vector<std::vector<std::vector<double>>> node_derivatives,
        double x0 = 0.0,
        double cutoff = std::numeric_limits<double>::quiet_NaN());
    void evaluate(
        const int num_nodes,
        Kokkos::View<const int*> node_types,
        Kokkos::View<const int*> num_neigh,
        Kokkos::View<const int*> neigh_types,
        Kokkos::View<const int*> type_to_active,
        int num_active_types,
        Kokkos::View<const double*> r,
        Kokkos::View<Precision**,Kokkos::LayoutRight> R,
        Kokkos::View<Precision**,Kokkos::LayoutRight> R_deriv,
        bool evaluate_derivatives = true) const;

    KOKKOS_INLINE_FUNCTION
    EvaluationPoint evaluation_point(double radius) const
    {
        int interval = static_cast<int>(Kokkos::floor((radius-x0)/h));
        double x = radius-x0-h*interval;
        const int num_intervals = num_nodes-1;
        if (interval < 0) {
            interval = 0;
            x = 0.0;
        } else if (interval >= num_intervals) {
            interval = num_intervals-1;
            x = h;
        }
        const double xx = x*x;
        return {interval, x, xx, xx*x};
    }

    KOKKOS_INLINE_FUNCTION
    CachedEvaluationPoint cached_evaluation_point(
        int interval, Precision x) const
    {
        const Precision xx=x*x;
        return {interval, x, xx, xx*x};
    }

    void prepare_evaluation_points(
        Kokkos::View<const double*> radii,
        Kokkos::View<int*> intervals,
        Kokkos::View<Precision*> coordinates) const
    {
        if(intervals.extent(0)!=radii.extent(0)
            ||coordinates.extent(0)!=radii.extent(0))
            throw std::invalid_argument(
                "Kokkos spline evaluation-point dimensions are inconsistent.");
        const double local_h=h;
        const double local_x0=x0;
        const int num_intervals=num_nodes-1;
        Kokkos::parallel_for(
            "RadialFunctionSetKokkos::prepare_evaluation_points",radii.extent(0),
            KOKKOS_LAMBDA(const std::size_t edge) {
                int interval=static_cast<int>(
                    Kokkos::floor((radii(edge)-local_x0)/local_h));
                double x=radii(edge)-local_x0-local_h*interval;
                if(interval<0) {
                    interval=0;
                    x=0.0;
                } else if(interval>=num_intervals) {
                    interval=num_intervals-1;
                    x=local_h;
                }
                intervals(edge)=interval;
                coordinates(edge)=static_cast<Precision>(x);
            });
    }

    template<typename Point>
    KOKKOS_INLINE_FUNCTION
    Precision evaluate_function(
        int edge_type,
        const Point& point,
        int function) const
    {
        const Precision c0 = coefficients(edge_type,point.interval,0,function);
        const Precision c1 = coefficients(edge_type,point.interval,1,function);
        const Precision c2 = coefficients(edge_type,point.interval,2,function);
        const Precision c3 = coefficients(edge_type,point.interval,3,function);
        return c0+c1*static_cast<Precision>(point.x)
            +c2*static_cast<Precision>(point.xx)
            +c3*static_cast<Precision>(point.xxx);
    }

    KOKKOS_INLINE_FUNCTION
    Precision evaluate_function(
        int edge_type,
        double radius,
        int function) const
    {
        return evaluate_function(edge_type, evaluation_point(radius), function);
    }

    template<typename Point>
    KOKKOS_INLINE_FUNCTION
    void evaluate_function(
        int edge_type,
        const Point& point,
        int function,
        Precision& value,
        Precision& derivative) const
    {
        const Precision c0 = coefficients(edge_type,point.interval,0,function);
        const Precision c1 = coefficients(edge_type,point.interval,1,function);
        const Precision c2 = coefficients(edge_type,point.interval,2,function);
        const Precision c3 = coefficients(edge_type,point.interval,3,function);
        value = c0+c1*static_cast<Precision>(point.x)
            +c2*static_cast<Precision>(point.xx)
            +c3*static_cast<Precision>(point.xxx);
        derivative = c1+c2*static_cast<Precision>(2.0*point.x)
            +c3*static_cast<Precision>(3.0*point.xx);
    }

    KOKKOS_INLINE_FUNCTION
    void evaluate_function(
        int edge_type,
        double radius,
        int function,
        Precision& value,
        Precision& derivative) const
    {
        evaluate_function(
            edge_type, evaluation_point(radius), function, value, derivative);
    }

    double spline_h() const noexcept { return h; }
    double spline_x0() const noexcept { return x0; }
    double spline_cutoff() const noexcept { return cutoff; }
    int edge_type_count() const noexcept { return num_edge_types; }
    int function_count() const noexcept { return num_functions; }
    int interval_count() const noexcept { return num_nodes-1; }
    const Precision* coefficient_data() const noexcept {
        return coefficients.data();
    }

private:

    double h = 0.0;
    double x0 = 0.0;
    double cutoff = std::numeric_limits<double>::quiet_NaN();
    int num_edge_types = 0;
    int num_functions = 0;
    int num_nodes = 0;
    Kokkos::View<const Precision****,Kokkos::LayoutRight> coefficients;
};
