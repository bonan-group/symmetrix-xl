#pragma once

#include <tuple>
#include <vector>

template <typename Precision>
class CubicSplineT {

public:

CubicSplineT(Precision h,
             std::vector<Precision> nodal_values,
             std::vector<Precision> nodal_derivs,
             Precision x0 = 0.0);

auto evaluate(Precision r) -> Precision;
auto evaluate_deriv(Precision r) -> std::tuple<Precision,Precision>;
auto evaluate_deriv_divided(Precision r) -> std::tuple<Precision,Precision>;

private:

Precision h;
Precision x0;
std::vector<Precision> c;

auto generate_coefficients(
    Precision h,
    std::vector<Precision> nodal_values,
    std::vector<Precision> nodal_derivs)
    -> std::vector<Precision>;
};

using CubicSpline = CubicSplineT<double>;

extern template class CubicSplineT<float>;
extern template class CubicSplineT<double>;
