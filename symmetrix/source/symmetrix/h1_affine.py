"""Inverse-free H1 affine representation of standard MACE JSON (format version 4).

The first readout and the H1->H2 skip of a two-interaction MACE act on the input
of ``interactions[1]``, i.e. on the features before ``interactions[1].linear_up``
(U).  Format versions 1 and 2 folded U into ``H1_weights`` (C = P U) and stored
the readout and skip after U, which requires ``inv(U)``:

    readout_1_weights      r   = U^-1 r0 * scale
    H2_weights_for_H1[z]   S_z = U^-1 S0_z

A nearly singular U makes r and S_z large although the native weights are flat,
and the fp32 evaluators then lose accuracy.  Format version 4 never forms the
inverse.  It stores the readout and skip in one of two bases (``h1_affine_mode``):

``m0_composite`` (models whose pre-linear_up features are P M0: no
first-interaction residual, no field coupling; single-layer models with zero
maps)::

    readout_1_weights_for_M0     q   = P0 r0 * scale = C r     (K values)
    H2_weights_for_M0[z]         T_z = P0 S0_z       = C S_z   (K*K values, row-major)

``pre_linear_up`` (two-interaction models whose JSON keeps ``H1_product_weights``
and ``H1_linear_up_weights`` split: first-interaction residual, MACEField)::

    readout_1_weights_for_H1_pre_linear_up    r0 * scale = U0 r
    H2_weights_for_H1_pre_linear_up[z]        S0_z       = U0 S_z

i.e. the checkpoint's own maps on the scalar block of the pre-linear_up features
(P M0 + residual + field term).  ``H1_weights`` is unchanged and still feeds the
second interaction.  The old ``readout_1_weights`` / ``H2_weights_for_H1`` keys
are omitted so that no reader can interpret them in the wrong basis.  Both
bases cover multi-head JSON (each head payload carries the basis' readout key)
and both radial representations (compact and pair-splines).

Format-1/2 files are still evaluated with the old arithmetic; the native
loaders print a warning with ``legacy_cancellation`` (computed from the file)
and recommend re-export.  ``recompose_legacy_payload`` converts every
format-1/2 kind without the checkpoint: the products C r, C S_z (or U0 r,
U0 S_z) undo the exporter's inverse up to the rounding already present in the
old file.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

H1_AFFINE_MODE_LEGACY = "legacy"
H1_AFFINE_MODE_M0_COMPOSITE = "m0_composite"
H1_AFFINE_MODE_PRE_LINEAR_UP = "pre_linear_up"
H1_AFFINE_MODES = (H1_AFFINE_MODE_M0_COMPOSITE, H1_AFFINE_MODE_PRE_LINEAR_UP)
H1_AFFINE_VERSION = 1
FORMAT_VERSION = 4
M0_COMPOSITE_FORMAT_VERSION = FORMAT_VERSION  # kept for existing callers
H1_AFFINE_TOOL_VERSION = "symmetrix-h1-affine-2"

LEGACY_KEYS = ("readout_1_weights", "H2_weights_for_H1")
BASIS_KEYS = {
    H1_AFFINE_MODE_M0_COMPOSITE: ("readout_1_weights_for_M0", "H2_weights_for_M0"),
    H1_AFFINE_MODE_PRE_LINEAR_UP: (
        "readout_1_weights_for_H1_pre_linear_up",
        "H2_weights_for_H1_pre_linear_up",
    ),
}
M0_COMPOSITE_KEYS = BASIS_KEYS[H1_AFFINE_MODE_M0_COMPOSITE]
PRE_LINEAR_UP_KEYS = BASIS_KEYS[H1_AFFINE_MODE_PRE_LINEAR_UP]


class H1AffineError(ValueError):
    """Raised when a payload cannot be represented or validated in format 4."""


def legacy_cancellation(data: dict) -> dict:
    """fp32 cancellation estimate of a format-1/2 payload, from the file alone.

    The old runtime forms the scalar M0 (or, with field coupling, pre-linear_up)
    adjoint as r A^T and S_z g A^T, A = C = H1_weights[l=0] (A = U0 for MACEField),
    so the rounding of those sums scales with |A| |r| while the result is
    A r = P0 r0 (or r0).  Returns max_j (|A||r|)_j / max_j |A r|_j (readout) and
    the same ratio of the largest entries of |A||S_z| and A S_z, worst over the
    elements (skip); 1 for zero maps.  Mirrors libsymmetrix legacy_cancellation.
    """
    K, _, _ = _dims(data)
    key = "H1_linear_up_weights" if data.get("has_field_coupling") else "H1_weights"
    A = _scalar_block(data, key)
    r, S = legacy_maps(data)

    def ratio(terms, result):
        if result > 0.0:
            return float(terms / result)
        return float("inf") if terms > 0.0 else 1.0

    readout = ratio(np.max(np.abs(A) @ np.abs(r)), np.max(np.abs(A @ r)))
    skip = 1.0
    for S_z in S:
        skip = max(skip, ratio(np.max(np.abs(A) @ np.abs(S_z)), np.max(np.abs(A @ S_z))))
    return {"readout": readout, "skip": skip}


def _bounded(value):
    """JSON-safe diagnostic: None for an unbounded (inf/nan) value."""
    return float(value) if value is not None and math.isfinite(value) else None


def _estimate_text(value) -> str:
    return f"{value:.3g}" if value is not None and math.isfinite(value) else "unbounded"


def legacy_format_message(version: int, cancellation: dict | None = None) -> str:
    """Warning of a format-1/2 load (same text as the native loaders)."""
    estimate = ""
    if cancellation is not None:
        estimate = (
            f" (fp32 cancellation estimate from this file: readout "
            f"{_estimate_text(cancellation['readout'])}, skip "
            f"{_estimate_text(cancellation['skip'])})"
        )
    return (
        f"Symmetrix MACE JSON format version {version} stores readout_1 and the "
        f"H1->H2 skip multiplied by inv(linear_up){estimate}; it is evaluated with "
        "the old arithmetic. Re-export the checkpoint with `symmetrix_extract_mace` "
        "or convert the file with `symmetrix_recompose_mace_json --input OLD.json "
        "--output NEW.json` (format version 4). Re-export to v4 is recommended "
        "but optional."
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def basis_for(data: dict) -> str:
    """Basis a model needs: pre_linear_up when its pre-linear_up features are not
    linear in M0 (first-interaction residual or field coupling)."""
    two_interactions = int(data.get("num_interactions", 2)) == 2 and not bool(
        data.get("single_layer_readout", False)
    )
    split = bool(data.get("first_interaction_residual", False)) or bool(
        data.get("has_field_coupling", False)
    )
    return (
        H1_AFFINE_MODE_PRE_LINEAR_UP
        if two_interactions and split
        else H1_AFFINE_MODE_M0_COMPOSITE
    )


def _finite_array(values: Any, name: str, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size != size:
        raise H1AffineError(f"{name} must hold {size} values, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise H1AffineError(f"{name} contains non-finite values.")
    return array


def _dims(data: dict) -> tuple[int, int, int]:
    try:
        K = int(data["num_channels"])
        L_max = int(data["L_max"])
        num_elements = int(data["num_elements"])
    except (KeyError, TypeError, ValueError) as exc:
        raise H1AffineError(f"missing/invalid model dimension: {exc}") from exc
    if K <= 0 or L_max < 0 or num_elements <= 0:
        raise H1AffineError("invalid model dimensions")
    if len(data.get("atomic_numbers", ())) != num_elements:
        raise H1AffineError("atomic_numbers length does not match num_elements")
    return K, L_max, num_elements


def _scalar_block(data: dict, key: str) -> np.ndarray:
    K, L_max, _ = _dims(data)
    if key not in data:
        raise H1AffineError(f"payload is missing {key}")
    values = _finite_array(data[key], key, (L_max + 1) * K * K)
    return values[: K * K].reshape(K, K)


def scalar_h1_map(data: dict) -> np.ndarray:
    """C = H1_weights[l=0] as a (K, K) fp64 matrix (row = M0 channel, col = H1 channel)."""
    return _scalar_block(data, "H1_weights")


def scalar_linear_up_map(data: dict) -> np.ndarray:
    """U0 = H1_linear_up_weights[l=0] (row = pre-linear_up channel, col = H1 channel)."""
    return _scalar_block(data, "H1_linear_up_weights")


def _readout_and_skip(data: dict, readout_key: str, skip_key: str) -> tuple[np.ndarray, np.ndarray]:
    K, _, num_elements = _dims(data)
    for key in (readout_key, skip_key):
        if key not in data:
            raise H1AffineError(f"payload is missing {key}")
    r = _finite_array(data[readout_key], readout_key, K)
    rows = data[skip_key]
    if not isinstance(rows, list) or len(rows) != num_elements:
        raise H1AffineError(f"{skip_key} must have one row per element")
    S = np.stack(
        [_finite_array(row, f"{skip_key}[{z}]", K * K) for z, row in enumerate(rows)]
    ).reshape(num_elements, K, K)
    return r, S


def legacy_maps(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """(r, S) of a format-1/2 payload: r (K,), S (num_elements, K, K)."""
    return _readout_and_skip(data, *LEGACY_KEYS)


def affine_maps(data: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(readout, skip) of a format-4 payload in its basis."""
    return _readout_and_skip(data, *BASIS_KEYS[mode])


def m0_composite_maps(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """(q, T) of an m0_composite payload: q (K,), T (num_elements, K, K)."""
    return affine_maps(data, H1_AFFINE_MODE_M0_COMPOSITE)


def _check_head_payloads(data: dict, readout_key: str) -> None:
    section = data.get("prediction_heads")
    if section is None:
        return
    K, _, _ = _dims(data)
    for name, payload in section.get("parameters", {}).items():
        if not isinstance(payload, dict) or readout_key not in payload:
            raise H1AffineError(f"prediction head {name!r} is missing {readout_key}")
        stale = [key for key in LEGACY_KEYS if key in payload]
        if stale:
            raise H1AffineError(f"prediction head {name!r} carries old-basis keys {stale}")
        _finite_array(payload[readout_key], f"{name}.{readout_key}", K)


def h1_affine_mode_of(data: dict) -> str:
    """Strict format-4 check (mirror of libsymmetrix h1_affine_schema.hpp).

    Returns the basis, or "legacy" for a format-1/2 payload (loaded with a
    warning by the native loaders); nonlinear payloads are not standard MACE.
    """
    if data.get("model_type") == "MACE_Nonlinear":
        raise H1AffineError(
            "MACE_Nonlinear (format 3) keeps linear_up as its own layer and has no "
            "H1 affine basis; it needs no conversion"
        )
    version = int(data.get("symmetrix_format_version", 1))
    if version in (1, 2):
        declared = [key for key in ("h1_affine_mode", "h1_affine_version") if key in data]
        declared += [key for keys in BASIS_KEYS.values() for key in keys if key in data]
        if declared:
            raise H1AffineError(
                f"format version {version} must not declare h1_affine fields {declared}")
        return H1_AFFINE_MODE_LEGACY
    if version != FORMAT_VERSION:
        raise H1AffineError(f"unsupported symmetrix_format_version {version} for MACE")
    mode = data.get("h1_affine_mode")
    if mode not in H1_AFFINE_MODES:
        raise H1AffineError(
            f"format version 4 requires h1_affine_mode 'm0_composite' or "
            f"'pre_linear_up', got {mode!r}"
        )
    if data.get("h1_affine_version") != H1_AFFINE_VERSION:
        raise H1AffineError(
            f"unsupported h1_affine_version {data.get('h1_affine_version')!r} "
            f"(supported: {H1_AFFINE_VERSION})"
        )
    present_old = [key for key in LEGACY_KEYS if key in data]
    if present_old:
        raise H1AffineError(f"format-4 payload must not contain legacy keys {present_old}")
    other = next(m for m in H1_AFFINE_MODES if m != mode)
    present_other = [key for key in BASIS_KEYS[other] if key in data]
    if present_other:
        raise H1AffineError(f"a {mode!r} payload must not contain {present_other}")
    if data.get("model_type", "MACE") not in ("MACE", "MACEField"):
        raise H1AffineError(f"model_type {data.get('model_type')!r} is not standard MACE")
    if data.get("radial_representation") not in ("compact", "pair-splines"):
        raise H1AffineError(
            "format version 4 requires radial_representation 'compact' or "
            f"'pair-splines', got {data.get('radial_representation')!r}"
        )
    needed = basis_for(data)
    if mode == H1_AFFINE_MODE_M0_COMPOSITE and needed != mode:
        raise H1AffineError(
            "the m0_composite basis cannot represent first-interaction residual or "
            "field-coupled features; such models use 'pre_linear_up'"
        )
    if mode == H1_AFFINE_MODE_PRE_LINEAR_UP:
        if int(data.get("num_interactions", 2)) != 2 or data.get("single_layer_readout"):
            raise H1AffineError("the pre_linear_up basis applies to two-interaction models only")
        if "H1_product_weights" not in data:
            raise H1AffineError("the pre_linear_up basis requires H1_product_weights")
    missing = [key for key in BASIS_KEYS[mode] if key not in data]
    if missing:
        raise H1AffineError(f"{mode} payload is missing {missing}")
    affine_maps(data, mode)
    _check_head_payloads(data, BASIS_KEYS[mode][0])
    return mode


def validate_payload(data: dict) -> dict:
    """Strict validation; returns a small summary (format 1/2: the warning and
    its cancellation estimate)."""
    mode = h1_affine_mode_of(data)
    K, L_max, num_elements = _dims(data)
    if mode == H1_AFFINE_MODE_LEGACY:
        version = int(data.get("symmetrix_format_version", 1))
        cancellation = legacy_cancellation(data)
        warning = legacy_format_message(version, cancellation)
        cancellation = {key: _bounded(value) for key, value in cancellation.items()}
        return {"mode": mode, "format_version": version, "K": K, "L_max": L_max,
                "num_elements": num_elements, "cancellation": cancellation,
                "warning": warning}
    readout, skip = affine_maps(data, mode)
    return {
        "mode": mode,
        "K": K,
        "L_max": L_max,
        "num_elements": num_elements,
        "readout_1_l2": float(np.linalg.norm(readout)),
        "skip_fro": float(np.linalg.norm(skip)),
    }


def conditioning(A: np.ndarray, name: str = "C") -> dict:
    sv = np.linalg.svd(A, compute_uv=False)
    smin = float(sv[-1])
    return {
        f"sigma_min_{name}": smin,
        f"sigma_max_{name}": float(sv[0]),
        f"cond_{name}": (float(sv[0] / smin) if smin > 0 else None),  # None: unbounded
    }


def identity_check(A: np.ndarray, r: np.ndarray, S: np.ndarray, q: np.ndarray,
                   T: np.ndarray, *, num_probe_rows: int = 64, seed: int = 20261017) -> dict:
    """fp64 identities (m A) r == m q and (m A) S_z == m T_z for every element z.

    A is the map in front of the old basis (C for m0_composite, U0 for
    pre_linear_up).  ``*_rel`` divides by the result scale (informative; grows
    with the old basis' cancellation).  ``*_rel_terms`` divides by the magnitude
    of the summed terms, a pure round-off measure (~K eps for a correct identity
    at any conditioning); the conversion gate uses it.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((num_probe_rows, A.shape[0]))
    H1 = m @ A
    e_old = H1 @ r
    e_new = m @ q
    e_terms = np.abs(H1) @ np.abs(r)
    q_terms = np.abs(A) @ np.abs(r)
    out = {
        "probe_rows": num_probe_rows,
        "seed": seed,
        "q_vs_Cr_max_rel": float(np.max(np.abs(q - A @ r)) / max(np.max(np.abs(q)), 1e-300)),
        "q_vs_Cr_max_rel_terms": float(np.max(np.abs(q - A @ r) / np.maximum(q_terms, 1e-300))),
        "E1_max_abs_diff": float(np.max(np.abs(e_old - e_new))),
        "E1_scale": float(np.max(np.abs(e_new))),
        "E1_max_rel_terms": float(np.max(np.abs(e_old - e_new) / np.maximum(e_terms, 1e-300))),
        "per_element": [],
    }
    for z in range(S.shape[0]):
        h2_old = H1 @ S[z]
        h2_new = m @ T[z]
        h2_terms = np.abs(H1) @ np.abs(S[z])
        t_terms = np.abs(A) @ np.abs(S[z])
        out["per_element"].append({
            "z_index": z,
            "T_vs_CS_max_rel": float(
                np.max(np.abs(T[z] - A @ S[z])) / max(np.max(np.abs(T[z])), 1e-300)),
            "T_vs_CS_max_rel_terms": float(
                np.max(np.abs(T[z] - A @ S[z]) / np.maximum(t_terms, 1e-300))),
            "H2_max_abs_diff": float(np.max(np.abs(h2_old - h2_new))),
            "H2_scale": float(np.max(np.abs(h2_new))),
            "H2_max_rel_terms": float(
                np.max(np.abs(h2_old - h2_new) / np.maximum(h2_terms, 1e-300))),
        })
    return out


def recompose_legacy_payload(data: dict, *, source_sha256: str, source_bytes: int,
                             source_path: str | None = None) -> tuple[dict, dict]:
    """Convert a format-1/2 standard MACE / MACEField payload to format 4.

    Covers every old kind: compact (v2) and pair-splines (v1) radial data,
    two-interaction models (m0_composite: q = C r, T_z = C S_z; residual-first
    and MACEField: pre_linear_up with r0 s = U0 r, S0_z = U0 S_z), single-layer
    models (zero maps) and multi-head JSON (every head's readout_1_weights is
    converted with the same map).  Only the readout/skip keys and the version
    header change; the inverse-rounding error of the source export is kept,
    nothing else is changed.
    """
    if data.get("model_type") == "MACE_Nonlinear":
        raise H1AffineError(
            "MACE_Nonlinear (format 3) stores linear_up as its own layer (no inverse); "
            "it needs no conversion"
        )
    version = int(data.get("symmetrix_format_version", 1))
    if version == FORMAT_VERSION:
        raise H1AffineError("input is already a format-version-4 payload")
    if version not in (1, 2):
        raise H1AffineError(f"unsupported symmetrix_format_version {version}")
    if any(key in data for key in ("h1_affine_mode", "h1_affine_version")) or any(
        key in data for keys in BASIS_KEYS.values() for key in keys
    ):
        raise H1AffineError(
            f"format version {version} must not declare h1_affine fields")
    radial = "compact" if version == 2 else "pair-splines"
    if version == 2 and data.get("radial_representation") != "compact":
        raise H1AffineError("format version 2 requires radial_representation 'compact'")
    mode = basis_for(data)
    if mode == H1_AFFINE_MODE_PRE_LINEAR_UP:
        A = scalar_linear_up_map(data)
        if "H1_product_weights" not in data:
            raise H1AffineError("residual-first / MACEField payload lacks H1_product_weights")
    else:
        A = scalar_h1_map(data)
    r, S = legacy_maps(data)
    q = A @ r
    T = np.einsum("ab,zbc->zac", A, S)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(T))):
        raise H1AffineError("non-finite recomposed maps")
    checks = identity_check(A, r, S, q, T)
    checks["mode"] = mode
    diag = conditioning(scalar_h1_map(data))
    diag["legacy_cancellation"] = {
        key: _bounded(value) for key, value in legacy_cancellation(data).items()}
    diag.update({
        "legacy_readout_1_l2": float(np.linalg.norm(r)),
        "legacy_skip_fro": float(np.linalg.norm(S)),
        "readout_1_l2": float(np.linalg.norm(q)),
        "skip_fro": float(np.linalg.norm(T)),
    })
    if mode == H1_AFFINE_MODE_PRE_LINEAR_UP:
        diag.update(conditioning(A, "U0"))
    readout_key, skip_key = BASIS_KEYS[mode]
    out = {}
    for key, value in data.items():
        if key in LEGACY_KEYS:
            continue
        out[key] = value
    out["symmetrix_format_version"] = FORMAT_VERSION
    out["h1_affine_mode"] = mode
    out["h1_affine_version"] = H1_AFFINE_VERSION
    out["radial_representation"] = radial
    out[readout_key] = q.tolist()
    out[skip_key] = [T[z].reshape(-1).tolist() for z in range(T.shape[0])]
    section = data.get("prediction_heads")
    if section is not None:
        K = q.size
        parameters = {}
        for name, payload in section.get("parameters", {}).items():
            payload = dict(payload)
            if "readout_1_weights" not in payload:
                raise H1AffineError(f"prediction head {name!r} lacks readout_1_weights")
            head_r = _finite_array(payload.pop("readout_1_weights"), f"{name}.readout_1_weights", K)
            payload[readout_key] = (A @ head_r).tolist()
            parameters[name] = payload
        out["prediction_heads"] = dict(section, parameters=parameters)
    out["h1_affine_provenance"] = {
        "source": "json_migration",
        "origin": f"converted_from_v{version}",
        "source_json_sha256": source_sha256,
        "source_json_bytes": int(source_bytes),
        "source_json_path": source_path,
        "source_format_version": version,
        "tool": "symmetrix.cli.recompose_mace_json",
        "tool_version": H1_AFFINE_TOOL_VERSION,
        "numpy_version": np.__version__,
        "conditioning": diag,
    }
    h1_affine_mode_of(out)
    return out, checks


def native_composite_maps(P0: np.ndarray, r0_scaled: np.ndarray,
                          S0: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fresh export from native weights: q = P0 r0*scale, T_z = P0 S0_z (no inverse)."""
    q = P0 @ r0_scaled
    T = [P0 @ S0_z for S0_z in S0]
    return q, T
