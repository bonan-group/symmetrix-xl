#pragma once

#include <string>

#include "mace_polar_program.hpp"
#include "nlohmann/json.hpp"

// The PolarMACE family's schema over the shared nonlinear backbone JSON.
//
// The family's models are MH-1 backbones carrying the polar deltas, so they are
// validated by the shared backbone validator under the polar rule row
// (``MaceBackboneSchema::polar``: the family's model type, and the two shape
// spellings its released checkpoints use -- ``scale_shift`` as a leading
// size-1 tensor and ``atomic_energies`` as a vector).  On top of that, this
// module owns the *polar* surface the family's files must carry:
//
//   * the ``polar`` backbone-delta block: the readout invariant
//     (``keep_last_layer_irreps``), the recursion depth, and the per-layer
//     spin-source heads, feature mixer, and field-update blocks the extractor
//     serializes -- each count checked against the backbone it belongs to, so a
//     malformed file cannot pose as a polar model;
//   * the embedded ``POLAR_BACKBONE`` identity contract, whose long-range slice
//     configures the evaluator's kind group. Checkpoint extraction may use
//     Python to build and embed that identity. JSON evaluation is native: the
//     loader checks the key, tag, dependent counts, and long-range slice
//     without a Python scientific stage.
//
// Pinned cross-language by ``test_mace_polar_schema.py``: the model type and
// the contract tag/version below are the Python ``POLAR_MODEL_TYPE`` and
// ``POLAR_CONTRACT_TAG``.
namespace symmetrix::mace_polar {

//: The model type a PolarMACE model file declares.  It is the family
//: registry's detection key for the family.
inline constexpr const char* model_type = "MACE_Polar";

//: The contract the family's models embed, and its schema/version.  Version 2
//: is the version that carries the unrolled chain as contract data; version 1
//: is rejected by name (nothing shipped it).
inline constexpr const char* contract_key = "POLAR_BACKBONE";
inline constexpr const char* contract_schema =
    "symmetrix.execution.polar_backbone";
inline constexpr int contract_version = 2;
inline constexpr const char* contract_tag =
    "symmetrix.execution.polar_backbone/2";

//: Validate a polar model file: the shared backbone rules under the polar row,
//: then the family's own polar surface.
void validate_mace_polar_schema(const nlohmann::json& data);

//: The long-range slice of the file's embedded POLAR_BACKBONE contract, which
//: the family's evaluator binds through its validated configuration setter.
//: Throws when the contract is absent or its slice is malformed; the slice's
//: admission rules are the kind group's own.
execution::polar::PolarLongRangeContract long_range_contract_from_model(
    const nlohmann::json& data);

} // namespace symmetrix::mace_polar
