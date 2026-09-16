#pragma once

#include <string>
#include <utility>

// Kept separate from evaluator constructors so front ends can reject a model
// family before trying to deserialize it through the wrong evaluator.
std::pair<std::string, bool> symmetrix_model_metadata(const std::string& filename);
std::string symmetrix_model_type(const std::string& filename);
bool symmetrix_is_nonlinear_mace_model(const std::string& filename);
