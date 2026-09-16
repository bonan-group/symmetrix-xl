#pragma once

#include <span>
#include <vector>

template <typename Precision>
class CubicSplineSetT {

public:

struct EvaluationPoint {
    int interval;
    Precision x;
    Precision xx;
    Precision xxx;
};

CubicSplineSetT(Precision h,
                std::vector<std::vector<Precision>> nodal_values,
                std::vector<std::vector<Precision>> nodal_derivs,
                Precision x0 = 0.0);

void evaluate(Precision r, std::span<Precision> values);
void evaluate_derivs(Precision r, std::span<Precision> values, std::span<Precision> derivs);
EvaluationPoint evaluation_point(Precision r) const;
Precision evaluate_function(const EvaluationPoint& point, int function) const;
void evaluate_function_derivs(
    const EvaluationPoint& point,
    int function,
    Precision& value,
    Precision& derivative) const;

// TODO: protect this with accessor
int num_splines;

private:

Precision h;
Precision x0;
int num_nodes;
std::vector<Precision> c;

};

using CubicSplineSet = CubicSplineSetT<double>;

extern template class CubicSplineSetT<float>;
extern template class CubicSplineSetT<double>;
