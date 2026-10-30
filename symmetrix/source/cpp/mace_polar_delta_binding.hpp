#pragma once

#include <pybind11/pybind11.h>

// The PolarMACE backbone deltas' Python surface (mace_polar_deltas.cpp): one
// binder per evaluator class, so the family's two classes carry the same
// block-level and chain-level entry points and the family's evaluator keeps a
// single class per precision.
namespace symmetrix::mace_polar {

template<typename Evaluator>
void bind_polar_delta_surface(pybind11::class_<Evaluator>& cls);

} // namespace symmetrix::mace_polar
