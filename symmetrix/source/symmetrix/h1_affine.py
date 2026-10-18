"""Inverse-free H1 affine representation (``h1_affine_mode = "m0_composite"``).

Legacy MACE exports fold ``interactions[1].linear_up`` (U) into ``H1_weights``
(C = P U) and store the first readout and the H1->H2 skip in the post-U basis,
which requires ``inv(U)``:

    readout_1_weights      r   = U^-1 r0 * scale
    H2_weights_for_H1[z]   S_z = U^-1 S0_z

A nearly singular U makes r and S_z large although the native weights are
flat; the runtime then stores H1 = M0 C and the H1 adjoint r + S_z g2 in the
working precision before the C^T contraction, which amplifies fp32 rounding.

Format version 4 with ``h1_affine_mode = "m0_composite"`` stores instead the
composite maps that act on the scalar M0 block directly:

    readout_1_weights_for_M0     q   = C r   = P0 r0 * scale   (K values)
    H2_weights_for_M0[z]         T_z = C S_z = P0 S0_z         (K*K values, row-major)

so that E1 = m.q, H2 = m T_z + M1 W, and the scalar M0 adjoint is
a C^T + q + T_z g2 where a is the message-passing (R1) part of the H1 adjoint
only.  ``H1_weights`` is unchanged and still feeds the second interaction.
The legacy ``readout_1_weights`` / ``H2_weights_for_H1`` keys are omitted from a
v4 payload so that no reader can interpret them in the wrong basis.

Scope (``h1_affine_version`` 1): single-head, two-interaction MACE with compact
radial data, no field coupling, no first-interaction residual, no single-layer
readout.  Everything else is refused.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

H1_AFFINE_MODE_LEGACY = "legacy"
H1_AFFINE_MODE_M0_COMPOSITE = "m0_composite"
H1_AFFINE_MODES = (H1_AFFINE_MODE_LEGACY, H1_AFFINE_MODE_M0_COMPOSITE)
H1_AFFINE_VERSION = 1
M0_COMPOSITE_FORMAT_VERSION = 4
H1_AFFINE_TOOL_VERSION = "symmetrix-h1-affine-m0c-1"

LEGACY_KEYS = ("readout_1_weights", "H2_weights_for_H1")
M0_COMPOSITE_KEYS = ("readout_1_weights_for_M0", "H2_weights_for_M0")


class H1AffineError(ValueError):
    """Raised when a model cannot be represented or validated in m0_composite form."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_array(values: Any, name: str, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size != size:
        raise H1AffineError(f"{name} must hold {size} values, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise H1AffineError(f"{name} contains non-finite values.")
    return array


def refuse_unsupported_model(data: dict, *, context: str) -> None:
    """Refuse every model family the m0_composite contract does not cover."""
    reasons = []
    if data.get("model_type", "MACE") != "MACE":
        reasons.append(f"model_type={data.get('model_type')!r} (only 'MACE')")
    if int(data.get("num_interactions", 2)) != 2:
        reasons.append(f"num_interactions={data.get('num_interactions')}")
    if bool(data.get("single_layer_readout", False)):
        reasons.append("single_layer_readout")
    if bool(data.get("has_field_coupling", False)) or data.get("field_couplings"):
        reasons.append("field coupling")
    if bool(data.get("first_interaction_residual", False)):
        reasons.append("first_interaction_residual")
    if "prediction_heads" in data:
        reasons.append("multi-head payload (prediction_heads)")
    if data.get("radial_representation") != "compact":
        reasons.append(
            f"radial_representation={data.get('radial_representation')!r} (compact required)"
        )
    if reasons:
        raise H1AffineError(
            f"{context}: h1_affine_mode 'm0_composite' supports only single-head "
            "two-interaction compact MACE without field coupling, first-interaction "
            "residual or single-layer readout; refused because: " + "; ".join(reasons)
        )


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


def scalar_h1_map(data: dict) -> np.ndarray:
    """C = H1_weights[l=0] as a (K, K) fp64 matrix (row = M0 channel, col = H1 channel)."""
    K, L_max, _ = _dims(data)
    H1 = _finite_array(data["H1_weights"], "H1_weights", (L_max + 1) * K * K)
    return H1[: K * K].reshape(K, K)


def legacy_maps(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """(r, S) of a legacy payload: r (K,), S (num_elements, K, K)."""
    K, _, num_elements = _dims(data)
    for key in LEGACY_KEYS:
        if key not in data:
            raise H1AffineError(f"legacy payload is missing {key}")
    r = _finite_array(data["readout_1_weights"], "readout_1_weights", K)
    rows = data["H2_weights_for_H1"]
    if len(rows) != num_elements:
        raise H1AffineError("H2_weights_for_H1 must have one row per element")
    S = np.stack(
        [_finite_array(row, f"H2_weights_for_H1[{z}]", K * K) for z, row in enumerate(rows)]
    ).reshape(num_elements, K, K)
    return r, S


def m0_composite_maps(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """(q, T) of a v4 payload: q (K,), T (num_elements, K, K)."""
    K, _, num_elements = _dims(data)
    q = _finite_array(data["readout_1_weights_for_M0"], "readout_1_weights_for_M0", K)
    rows = data["H2_weights_for_M0"]
    if len(rows) != num_elements:
        raise H1AffineError("H2_weights_for_M0 must have one row per element")
    T = np.stack(
        [_finite_array(row, f"H2_weights_for_M0[{z}]", K * K) for z, row in enumerate(rows)]
    ).reshape(num_elements, K, K)
    return q, T


def h1_affine_mode_of(data: dict) -> str:
    """Return the declared basis after a strict schema check (legacy or m0_composite)."""
    version = int(data.get("symmetrix_format_version", 1))
    mode = data.get("h1_affine_mode")
    present_new = [key for key in M0_COMPOSITE_KEYS if key in data]
    present_old = [key for key in LEGACY_KEYS if key in data]
    if version in (1, 2):
        if mode is not None or "h1_affine_version" in data or present_new:
            raise H1AffineError(
                f"format version {version} must not declare h1_affine fields "
                f"(mode={mode!r}, keys={present_new})"
            )
        return H1_AFFINE_MODE_LEGACY
    if version != M0_COMPOSITE_FORMAT_VERSION:
        raise H1AffineError(f"unsupported symmetrix_format_version {version} for MACE")
    if mode != H1_AFFINE_MODE_M0_COMPOSITE:
        raise H1AffineError(f"format version 4 requires h1_affine_mode 'm0_composite', got {mode!r}")
    if data.get("h1_affine_version") != H1_AFFINE_VERSION:
        raise H1AffineError(
            f"unsupported h1_affine_version {data.get('h1_affine_version')!r} "
            f"(supported: {H1_AFFINE_VERSION})"
        )
    if present_old:
        raise H1AffineError(f"m0_composite payload must not contain legacy keys {present_old}")
    missing = [key for key in M0_COMPOSITE_KEYS if key not in data]
    if missing:
        raise H1AffineError(f"m0_composite payload is missing {missing}")
    refuse_unsupported_model(data, context="format version 4")
    return H1_AFFINE_MODE_M0_COMPOSITE


def validate_payload(data: dict) -> dict:
    """Strict validation of either basis; returns a small summary."""
    mode = h1_affine_mode_of(data)
    K, L_max, num_elements = _dims(data)
    C = scalar_h1_map(data)
    if mode == H1_AFFINE_MODE_LEGACY:
        r, S = legacy_maps(data)
        return {"mode": mode, "K": K, "L_max": L_max, "num_elements": num_elements,
                "readout_l2": float(np.linalg.norm(r)), "skip_fro": float(np.linalg.norm(S))}
    q, T = m0_composite_maps(data)
    return {"mode": mode, "K": K, "L_max": L_max, "num_elements": num_elements,
            "readout_for_M0_l2": float(np.linalg.norm(q)),
            "skip_for_M0_fro": float(np.linalg.norm(T)),
            "C_shape": list(C.shape)}


def conditioning(C: np.ndarray) -> dict:
    sv = np.linalg.svd(C, compute_uv=False)
    smin = float(sv[-1])
    return {
        "sigma_min_C": smin,
        "sigma_max_C": float(sv[0]),
        "cond_C": (float(sv[0] / smin) if smin > 0 else math.inf),
    }


def identity_check(C: np.ndarray, r: np.ndarray, S: np.ndarray, q: np.ndarray,
                   T: np.ndarray, *, num_probe_rows: int = 64, seed: int = 20261017) -> dict:
    """fp64 identities (m C) r == m q and (m C) S_z == m T_z for every element z.

    ``*_rel`` divides by the result scale (informative; grows with the legacy
    cancellation ratio).  ``*_rel_terms`` divides by the magnitude of the summed
    terms (|h|.|r|, |h|.|S| and |C|.|r|, |C|.|S|), i.e. a pure round-off measure
    that is ~K eps for a correct identity whatever the conditioning; the
    migration gate uses it.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((num_probe_rows, C.shape[0]))
    H1 = m @ C
    e_old = H1 @ r
    e_new = m @ q
    e_terms = np.abs(H1) @ np.abs(r)
    q_terms = np.abs(C) @ np.abs(r)
    out = {
        "probe_rows": num_probe_rows,
        "seed": seed,
        "q_vs_Cr_max_rel": float(np.max(np.abs(q - C @ r)) / max(np.max(np.abs(q)), 1e-300)),
        "q_vs_Cr_max_rel_terms": float(np.max(np.abs(q - C @ r) / np.maximum(q_terms, 1e-300))),
        "E1_max_abs_diff": float(np.max(np.abs(e_old - e_new))),
        "E1_scale": float(np.max(np.abs(e_new))),
        "E1_max_rel_terms": float(np.max(np.abs(e_old - e_new) / np.maximum(e_terms, 1e-300))),
        "per_element": [],
    }
    for z in range(S.shape[0]):
        h2_old = H1 @ S[z]
        h2_new = m @ T[z]
        h2_terms = np.abs(H1) @ np.abs(S[z])
        t_terms = np.abs(C) @ np.abs(S[z])
        out["per_element"].append({
            "z_index": z,
            "T_vs_CS_max_rel": float(
                np.max(np.abs(T[z] - C @ S[z])) / max(np.max(np.abs(T[z])), 1e-300)),
            "T_vs_CS_max_rel_terms": float(
                np.max(np.abs(T[z] - C @ S[z]) / np.maximum(t_terms, 1e-300))),
            "H2_max_abs_diff": float(np.max(np.abs(h2_old - h2_new))),
            "H2_scale": float(np.max(np.abs(h2_new))),
            "H2_max_rel_terms": float(
                np.max(np.abs(h2_old - h2_new) / np.maximum(h2_terms, 1e-300))),
        })
    return out


def recompose_legacy_payload(data: dict, *, source_sha256: str, source_bytes: int,
                             source_path: str | None = None) -> tuple[dict, dict]:
    """Build a new v4 m0_composite payload from a validated legacy v2 payload.

    The recomposition q = C r and T_z = C S_z is exact up to the inverse already
    applied by the legacy exporter (the inverse-rounding error of the source
    export is preserved, nothing else is changed).
    """
    if h1_affine_mode_of(data) != H1_AFFINE_MODE_LEGACY:
        raise H1AffineError("input is already an m0_composite payload")
    if int(data.get("symmetrix_format_version", 1)) != 2:
        raise H1AffineError(
            "JSON migration requires a compact format-version-2 MACE export "
            f"(got version {data.get('symmetrix_format_version', 1)})"
        )
    refuse_unsupported_model(data, context="recompose_mace_json")
    C = scalar_h1_map(data)
    r, S = legacy_maps(data)
    q = C @ r
    T = np.einsum("ab,zbc->zac", C, S)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(T))):
        raise H1AffineError("non-finite composite maps")
    checks = identity_check(C, r, S, q, T)
    diag = conditioning(C)
    diag.update({
        "sigma_min_U0": None,
        "sigma_min_U0_status": "UNAVAILABLE_FROM_LEGACY_JSON",
        "legacy_readout_1_l2": float(np.linalg.norm(r)),
        "legacy_skip_fro": float(np.linalg.norm(S)),
        "readout_1_for_M0_l2": float(np.linalg.norm(q)),
        "skip_for_M0_fro": float(np.linalg.norm(T)),
    })
    out = {}
    for key, value in data.items():
        if key == "symmetrix_format_version":
            out[key] = M0_COMPOSITE_FORMAT_VERSION
            out["h1_affine_mode"] = H1_AFFINE_MODE_M0_COMPOSITE
            out["h1_affine_version"] = H1_AFFINE_VERSION
        elif key == "readout_1_weights":
            out["readout_1_weights_for_M0"] = q.tolist()
        elif key == "H2_weights_for_H1":
            out["H2_weights_for_M0"] = [T[z].reshape(-1).tolist() for z in range(T.shape[0])]
        else:
            out[key] = value
    out["h1_affine_provenance"] = {
        "source": "json_migration",
        "source_json_sha256": source_sha256,
        "source_json_bytes": int(source_bytes),
        "source_json_path": source_path,
        "source_format_version": int(data.get("symmetrix_format_version", 1)),
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
