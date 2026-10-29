"""Inverse-free H1 affine basis (format v4, ``h1_affine_mode = "m0_composite"``).

Pure-python tests (schema, migration, exporter algebra, adjoint algebra) always run.
Native tests build a synthetic compact two-interaction model (``compact_r1_model``)
whose second-interaction ``linear_up`` U has a controlled smallest singular value,
store it once in the legacy basis (readout/skip multiplied by inv(U), exactly as the
legacy exporter does) and once in the m0_composite basis (P0 r0, P0 S0_z, no
inverse), and compare both with the serial and the Kokkos evaluators.  They skip
when the native extension predates format-v4 support.
"""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

import numpy as np
import pytest

from symmetrix.h1_affine import (
    H1AffineError,
    h1_affine_mode_of,
    identity_check,
    legacy_v2_reexport_guidance,
    recompose_legacy_payload,
    validate_payload,
)

RNG = np.random.default_rng(20261017)


# --------------------------------------------------------------------------- helpers


def _synthetic_legacy_payload(K=6, num_elements=6, L_max=1, sigma_min=1e-6, seed=3):
    rng = np.random.default_rng(seed)
    P = [rng.standard_normal((K, K)) / np.sqrt(K) for _ in range(L_max + 1)]
    U = []
    for l in range(L_max + 1):
        u, _, vt = np.linalg.svd(rng.standard_normal((K, K)))
        s = np.geomspace(1.0, sigma_min if l == 0 else 0.5, K)
        U.append((u * s) @ vt)
    C = np.stack([P[l] @ U[l] for l in range(L_max + 1)])
    r0 = rng.standard_normal(K) / np.sqrt(K)
    S0 = [
        rng.standard_normal((K, K)) / np.sqrt(K * num_elements)
        for _ in range(num_elements)
    ]
    r = np.linalg.solve(U[0], r0)
    S = [np.linalg.solve(U[0], s) for s in S0]
    payload = {
        "num_channels": K,
        "L_max": L_max,
        "num_interactions": 2,
        "model_type": "MACE",
        "has_field_coupling": False,
        "first_interaction_residual": False,
        "field_couplings": [],
        "atomic_numbers": list(range(1, num_elements + 1)),
        "num_elements": num_elements,
        "symmetrix_format_version": 2,
        "radial_representation": "compact",
        "H1_weights": C.reshape(-1).tolist(),
        "H2_weights_for_H1": [s.reshape(-1).tolist() for s in S],
        "H2_weights_for_M1": rng.standard_normal(K * K).tolist(),
        "readout_1_weights": r.tolist(),
    }
    native = {"P0": P[0], "U0": U[0], "r0": r0, "S0": S0, "C": C}
    return payload, native


def _assert_refused(payload, match=None):
    with pytest.raises(H1AffineError, match=match):
        recompose_legacy_payload(payload, source_sha256="0" * 64, source_bytes=1)


# --------------------------------------------------------------------------- schema / migration


def test_recompose_identities_and_v4_layout():
    payload, native = _synthetic_legacy_payload(sigma_min=1e-4)
    new, checks = recompose_legacy_payload(
        payload, source_sha256="a" * 64, source_bytes=10
    )
    assert new["symmetrix_format_version"] == 4
    assert new["h1_affine_mode"] == "m0_composite" and new["h1_affine_version"] == 1
    assert "readout_1_weights" not in new and "H2_weights_for_H1" not in new
    assert new["H1_weights"] == payload["H1_weights"]
    assert new["H2_weights_for_M1"] == payload["H2_weights_for_M1"]
    assert h1_affine_mode_of(new) == "m0_composite"
    q = np.asarray(new["readout_1_weights_for_M0"])
    T = np.asarray(new["H2_weights_for_M0"]).reshape(6, 6, 6)
    np.testing.assert_allclose(q, native["P0"] @ native["r0"], rtol=0, atol=1e-10)
    for z in range(6):
        np.testing.assert_allclose(
            T[z], native["P0"] @ native["S0"][z], rtol=0, atol=1e-10
        )
    assert checks["E1_max_abs_diff"] <= 1e-10 * max(1.0, checks["E1_scale"])
    prov = new["h1_affine_provenance"]
    assert prov["source"] == "json_migration" and prov["source_json_sha256"] == "a" * 64
    assert prov["origin"] == "converted_from_v2"
    assert prov["source_format_version"] == 2
    assert prov["conditioning"]["sigma_min_U0"] is None
    assert new["symmetrix_format_provenance"] == {
        "origin": "converted_from_v2",
        "source_format_version": 2,
        "source_json_sha256": "a" * 64,
        "tool": "symmetrix.cli.recompose_mace_json",
        "tool_version": "symmetrix-h1-affine-m0c-1",
    }


def test_v4_nonstandard_family_does_not_require_h1_affine_fields():
    payload, _ = _synthetic_legacy_payload(sigma_min=1.0)
    payload.update(
        symmetrix_format_version=4,
        model_type="MACEField",
        has_field_coupling=True,
    )

    assert h1_affine_mode_of(payload) == "legacy"
    assert validate_payload(payload)["mode"] == "legacy"


def test_v4_single_layer_standard_model_does_not_require_h1_affine_fields():
    assert (
        h1_affine_mode_of(
            {
                "model_type": "MACE",
                "symmetrix_format_version": 4,
                "num_interactions": 1,
                "single_layer_readout": True,
            }
        )
        == "legacy"
    )


def test_legacy_v2_guidance_distinguishes_map_amplification_risk():
    low_risk, _ = _synthetic_legacy_payload(K=2, num_elements=1, L_max=0, sigma_min=1.0)
    low_risk["H1_weights"] = np.eye(2).reshape(-1).tolist()
    low_risk["readout_1_weights"] = [1.0, 1.0]
    low_risk["H2_weights_for_H1"] = [np.eye(2).reshape(-1).tolist()]

    elevated = copy.deepcopy(low_risk)
    elevated["H1_weights"] = np.diag([1.0, 1e-8]).reshape(-1).tolist()
    elevated["readout_1_weights"] = [1.0, 1e8]

    low = legacy_v2_reexport_guidance(low_risk)
    high = legacy_v2_reexport_guidance(elevated)

    assert low["risk"] == "low"
    assert low["diagnostic_source"] == "exported_maps"
    assert "Re-export to v4 is recommended but optional." in low["message"]
    assert high["risk"] == "elevated"
    assert high["readout_amplification"] > low["readout_amplification"]
    assert "elevated float32 cancellation risk" in high["message"]


def test_legacy_v2_guidance_prefers_checkpoint_conditioning():
    payload, _ = _synthetic_legacy_payload(K=2, num_elements=1, L_max=0, sigma_min=1.0)
    payload["h1_affine_provenance"] = {
        "origin": "native_checkpoint",
        "conditioning": {"cond_U0": 2e7},
    }

    guidance = legacy_v2_reexport_guidance(payload)

    assert guidance["risk"] == "elevated"
    assert guidance["diagnostic_source"] == "checkpoint_linear_up"
    assert guidance["cond_U0"] == 2e7


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda d: d.update(prediction_heads={"names": ["a", "b"]}), "multi-head"),
        (lambda d: d.update(has_field_coupling=True), "field"),
        (
            lambda d: d.update(first_interaction_residual=True),
            "first_interaction_residual",
        ),
        (
            lambda d: d.update(single_layer_readout=True, num_interactions=1),
            "single_layer",
        ),
        (lambda d: d.update(model_type="MACE_Nonlinear"), "model_type"),
        (
            lambda d: d.update(symmetrix_format_version=1, radial_representation=None),
            "version-2",
        ),
        (lambda d: d.update(H2_weights_for_M0=[[0.0]]), "must not declare"),
        (lambda d: d.update(readout_1_weights=[1.0, 2.0]), "readout_1_weights"),
        (lambda d: d.update(readout_1_weights=[float("nan")] * 6), "non-finite"),
    ],
)
def test_recompose_refusals(mutation, match):
    payload, _ = _synthetic_legacy_payload()
    mutation(payload)
    _assert_refused(payload, match)


def test_validator_rejects_mixed_and_partial_v4():
    payload, _ = _synthetic_legacy_payload()
    new, _ = recompose_legacy_payload(payload, source_sha256="b" * 64, source_bytes=1)
    validate_payload(new)
    for mutate, match in (
        (lambda d: d.__setitem__("readout_1_weights", [0.0] * 6), "legacy keys"),
        (
            lambda d: d.__setitem__("H2_weights_for_H1", d["H2_weights_for_M0"]),
            "legacy keys",
        ),
        (lambda d: d.pop("H2_weights_for_M0"), "missing"),
        (lambda d: d.__setitem__("h1_affine_mode", "pre_linear_up"), "m0_composite"),
        (lambda d: d.__setitem__("h1_affine_version", 2), "h1_affine_version"),
        (lambda d: d.__setitem__("symmetrix_format_version", 3), "unsupported"),
        (lambda d: d["H2_weights_for_M0"].pop(), "one row per element"),
        (lambda d: d.__setitem__("readout_1_weights_for_M0", [0.0] * 5), "6 values"),
    ):
        bad = copy.deepcopy(new)
        mutate(bad)
        with pytest.raises(H1AffineError, match=match):
            validate_payload(bad)


def test_migration_cli_never_overwrites(tmp_path):
    from symmetrix.cli.recompose_mace_json import main

    payload, _ = _synthetic_legacy_payload()
    src = tmp_path / "old.json"
    src.write_text(json.dumps(payload))
    dst = tmp_path / "new" / "m0c.json"
    assert (
        main(["-i", str(src), "-o", str(dst), "--receipt", str(tmp_path / "r.json")])
        == 0
    )
    first = dst.read_bytes()
    assert main(["-i", str(src), "-o", str(dst)]) == 2
    assert dst.read_bytes() == first
    assert main(["-i", str(dst), "-o", str(tmp_path / "again.json")]) == 3
    assert main(["--validate-only", str(src), str(dst)]) == 0
    receipt = json.loads((tmp_path / "r.json").read_text())
    assert receipt["origin"] == "converted_from_v2"
    assert receipt["source_format_version"] == 2
    assert receipt["identity_fp64"]["detail"]["worst_rel_terms"] <= 1e-12


# --------------------------------------------------------------------------- native algebra


@pytest.mark.parametrize("sigma_min", [1e-2, 1e-6, 1e-10, 0.0])
def test_native_composite_needs_no_inverse(sigma_min):
    """q = P0 r0 and T = P0 S0 exist for ill-conditioned and singular U; the legacy
    inverse either amplifies (|r| ~ 1/sigma_min) or does not exist."""
    K = 8
    rng = np.random.default_rng(11)
    P0 = rng.standard_normal((K, K)) / np.sqrt(K)
    u, _, vt = np.linalg.svd(rng.standard_normal((K, K)))
    s = np.geomspace(1.0, max(sigma_min, 1e-300), K)
    if sigma_min == 0.0:
        s[-1] = 0.0
    U0 = (u * s) @ vt
    r0 = rng.standard_normal(K)
    S0 = rng.standard_normal((K, K))
    q = P0 @ r0
    T = P0 @ S0
    m = rng.standard_normal((32, K))
    h_pre = m @ P0  # native pre-linear_up features
    np.testing.assert_allclose(m @ q, h_pre @ r0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(m @ T, h_pre @ S0, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(q) <= np.linalg.norm(P0, 2) * np.linalg.norm(r0) * (1 + 1e-12)
    if sigma_min > 0.0:
        r = np.linalg.solve(U0, r0)
        assert np.linalg.norm(r) >= 0.1 * abs(u[:, -1] @ r0) / sigma_min
    else:
        assert np.linalg.matrix_rank(U0) < K


def _toy_energy(m, C, msg_w, q=None, T=None, r=None, S=None, W2=None, M1=None):
    """Toy two-branch energy with a message branch on h = m C (sin), the first
    readout and the skip feeding a nonlinear second readout (softplus)."""
    h = m @ C
    e_msg = np.sum(np.sin(h @ msg_w))
    if q is not None:
        e1 = np.sum(m @ q)
        h2 = np.einsum("nk,nkc->nc", m, T) + M1 @ W2
    else:
        e1 = np.sum(h @ r)
        h2 = np.einsum("nk,nkc->nc", h, S) + M1 @ W2
    e2 = np.sum(np.logaddexp(0.0, h2))
    return e_msg + e1 + e2


def test_reverse_algebra_matches_finite_differences():
    """Legacy reverse (a + r + S g2) C^T and composite reverse a C^T + q + T g2
    agree and match central finite differences (six species, ill-conditioned U)."""
    payload, native = _synthetic_legacy_payload(K=6, sigma_min=1e-5, seed=9)
    new, _ = recompose_legacy_payload(payload, source_sha256="c" * 64, source_bytes=1)
    K, nz, N = 6, 6, 12
    C = np.asarray(payload["H1_weights"]).reshape(2, K, K)[0]
    r = np.asarray(payload["readout_1_weights"])
    S = np.asarray(payload["H2_weights_for_H1"]).reshape(nz, K, K)
    q = np.asarray(new["readout_1_weights_for_M0"])
    T = np.asarray(new["H2_weights_for_M0"]).reshape(nz, K, K)
    rng = np.random.default_rng(5)
    types = np.arange(N) % nz  # all six species
    m = rng.standard_normal((N, K))
    msg_w = rng.standard_normal(K)
    W2 = rng.standard_normal((K, K)) / K
    M1 = rng.standard_normal((N, K))
    Sz, Tz = S[types], T[types]
    h = m @ C
    a = np.cos(h @ msg_w)[:, None] * msg_w[None, :]  # message (R1) adjoint only
    h2 = np.einsum("nk,nkc->nc", m, Tz) + M1 @ W2
    g2 = 1.0 / (1.0 + np.exp(-h2))
    g_legacy = (a + r[None, :] + np.einsum("nkc,nc->nk", Sz, g2)) @ C.T
    g_m0c = a @ C.T + q[None, :] + np.einsum("nkc,nc->nk", Tz, g2)
    scale = np.max(np.abs(g_m0c))
    assert np.max(np.abs(g_legacy - g_m0c)) <= 1e-8 * max(scale, 1.0)
    E = lambda mm: _toy_energy(mm, C, msg_w, q=q, T=Tz, W2=W2, M1=M1)  # noqa: E731
    E_leg = lambda mm: _toy_energy(mm, C, msg_w, r=r, S=Sz, W2=W2, M1=M1)  # noqa: E731
    assert abs(E(m) - E_leg(m)) <= 1e-9 * max(abs(E(m)), 1.0)
    eps = 1e-6
    for i, k in [(0, 0), (1, 3), (2, 5), (3, 1), (4, 2), (5, 4)]:
        dm = np.zeros_like(m)
        dm[i, k] = eps
        fd = (E(m + dm) - E(m - dm)) / (2 * eps)
        assert abs(fd - g_m0c[i, k]) <= 1e-6 * max(1.0, abs(fd))
    d = rng.standard_normal(m.shape)
    fd_dir = (E(m + eps * d) - E(m - eps * d)) / (2 * eps)
    assert abs(fd_dir - np.sum(g_m0c * d)) <= 1e-6 * max(1.0, abs(fd_dir))


def test_identity_check_reports_per_element():
    payload, _ = _synthetic_legacy_payload()
    new, checks = recompose_legacy_payload(
        payload, source_sha256="d" * 64, source_bytes=1
    )
    assert len(checks["per_element"]) == 6
    C = np.asarray(payload["H1_weights"]).reshape(2, 6, 6)[0]
    r = np.asarray(payload["readout_1_weights"])
    S = np.asarray(payload["H2_weights_for_H1"]).reshape(6, 6, 6)
    q = np.asarray(new["readout_1_weights_for_M0"])
    T = np.asarray(new["H2_weights_for_M0"]).reshape(6, 6, 6)
    again = identity_check(C, r, S, q, T)
    assert all(row["T_vs_CS_max_rel_terms"] <= 1e-12 for row in again["per_element"])
    assert all(row["H2_max_rel_terms"] <= 1e-12 for row in again["per_element"])
    assert again["E1_max_rel_terms"] <= 1e-12


# --------------------------------------------------------------------------- native library (synthetic model)

_CONTRACT = (
    Path(__file__).resolve().parent
    / "data"
    / "execution_contracts"
    / "jit_r1_mace_off23_small_contract.json"
)


def _native():
    nc = pytest.importorskip("symmetrix._native_cpu")
    if not hasattr(nc.MACE, "h1_affine_mode"):
        pytest.skip("native extension without format-v4 (h1_affine_mode) support")
    return nc


def synthetic_model_pair(sigma_min, seed=5):
    """Legacy (v2) and m0_composite (v4) payloads of one synthetic compact model.

    U0 = Q1 diag(geomspace(1, sigma_min)) Q2^T; C = P0 U0 is the fused H1 map.
    Legacy: readout_1 = inv(U0) r0, H2_weights_for_H1[z] = inv(U0) S0_z (the legacy
    exporter's algebra).  v4: readout_1_weights_for_M0 = P0 r0 and
    H2_weights_for_M0[z] = P0 S0_z, built from the "native" factors without any
    inverse; sigma_min = 0 (singular U0) returns legacy = None.
    """
    from compact_r1_model import foundation_r1_test_model

    model = foundation_r1_test_model(
        json.loads(_CONTRACT.read_text()), atomic_numbers=(1, 8)
    )
    rng = np.random.default_rng(seed)
    K, nz, L = model["num_channels"], model["num_elements"], model["L_max"]

    def randomize(values, scale):
        if isinstance(values, dict):
            return {name: randomize(value, scale) for name, value in values.items()}
        values = np.asarray(values, dtype=float)
        return (scale * rng.standard_normal(values.shape)).tolist()

    for key in ("H0_weights", "A0_weights", "A1_weights", "H2_weights_for_M1"):
        model[key] = randomize(model[key], 1.0 / np.sqrt(K))
    for key in ("M0_weights", "M1_weights"):
        model[key] = randomize(model[key], 0.5)
    P0 = rng.standard_normal((K, K)) / np.sqrt(K)
    q1, _, q2 = np.linalg.svd(rng.standard_normal((K, K)))
    s = np.geomspace(1.0, max(sigma_min, 1e-300), K)
    if sigma_min == 0.0:
        s[-1] = 0.0
    U0 = (q1 * s) @ q2
    r0 = rng.standard_normal(K) / np.sqrt(K)
    S0 = rng.standard_normal((nz, K, K)) / np.sqrt(K * nz)
    H1 = rng.standard_normal((L + 1, K, K)) / np.sqrt(K)
    H1[0] = P0 @ U0
    model["H1_weights"] = H1.reshape(-1).tolist()
    v4 = copy.deepcopy(model)
    for key in ("readout_1_weights", "H2_weights_for_H1"):
        v4.pop(key)
    v4.update(
        symmetrix_format_version=4,
        h1_affine_mode="m0_composite",
        h1_affine_version=1,
        readout_1_weights_for_M0=(P0 @ r0).tolist(),
        H2_weights_for_M0=[(P0 @ S0[z]).reshape(-1).tolist() for z in range(nz)],
        h1_affine_provenance={"source": "synthetic_test"},
    )
    legacy = None
    if sigma_min > 0.0:
        U_inv = np.linalg.inv(U0)
        legacy = copy.deepcopy(model)
        legacy["readout_1_weights"] = (U_inv @ r0).tolist()
        legacy["H2_weights_for_H1"] = [
            (U_inv @ S0[z]).reshape(-1).tolist() for z in range(nz)
        ]
    return legacy, v4


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _structure(seed=11, count=12):
    from ase import Atoms

    rng = np.random.default_rng(seed)
    return Atoms(
        ["H", "O", "O"] * (count // 3),
        positions=rng.uniform(0.0, 6.0, size=(count, 3)),
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )


def _efs(model_path, dtype, use_kokkos, displacement=None):
    """Energy, forces and stress of the synthetic structure; the evaluator is
    released before returning so Kokkos can be finalised at session end."""
    from symmetrix import Symmetrix

    kwargs = {"streamed_edges": "generic"} if use_kokkos else {}
    calculator = Symmetrix(
        str(model_path), use_kokkos=use_kokkos, dtype=dtype, **kwargs
    )
    atoms = _structure()
    if displacement is not None:
        atoms.positions = atoms.positions + displacement
    atoms.calc = calculator
    try:
        return (
            float(atoms.get_potential_energy()),
            np.array(atoms.get_forces(), copy=True),
            np.array(atoms.get_stress(), copy=True),
        )
    finally:
        atoms.calc = None
        del calculator
        gc.collect()


def test_synthetic_dispatch_and_payload_agreement(tmp_path):
    nc = _native()
    legacy, v4 = synthetic_model_pair(1e-4)
    validate_payload(v4)
    migrated, _ = recompose_legacy_payload(
        legacy, source_sha256="0" * 64, source_bytes=1
    )
    # migration from the legacy JSON reproduces the inverse-free maps up to the
    # inverse rounding already present in the legacy payload
    np.testing.assert_allclose(
        migrated["readout_1_weights_for_M0"],
        v4["readout_1_weights_for_M0"],
        rtol=0,
        atol=1e-9,
    )
    legacy_path = _write(tmp_path, "legacy.json", legacy)
    v4_path = _write(tmp_path, "v4.json", v4)
    for cls in (nc.MACE, nc.MACEFloat):
        assert cls(str(legacy_path), "").h1_affine_mode == "legacy"
        assert cls(str(v4_path), "").h1_affine_mode == "m0_composite"


def test_native_v4_nonstandard_family_uses_its_existing_payload(tmp_path):
    native = _native()
    legacy, _ = synthetic_model_pair(0.5)
    legacy["symmetrix_format_version"] = 4
    legacy["model_type"] = "MACEField"
    path = _write(tmp_path, "field-container-v4.json", legacy)

    assert native.MACE(str(path), "").h1_affine_mode == "legacy"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.__setitem__("readout_1_weights", d["readout_1_weights_for_M0"]),
        lambda d: d.pop("h1_affine_mode"),
        lambda d: d.__setitem__("h1_affine_version", 2),
        lambda d: d.__setitem__("symmetrix_format_version", 2),
        lambda d: d["H2_weights_for_M0"][0].pop(),
        lambda d: d.__setitem__("has_field_coupling", True),
    ],
)
def test_synthetic_loader_refuses_malformed_v4(tmp_path, mutation):
    nc = _native()
    _, v4 = synthetic_model_pair(1e-2)
    mutation(v4)
    path = _write(tmp_path, "bad.json", v4)
    with pytest.raises((ValueError, RuntimeError)):
        nc.MACE(str(path), "")


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_synthetic_fp64_equivalence(tmp_path, use_kokkos):
    _native()
    legacy, v4 = synthetic_model_pair(1e-4)
    E_old, F_old, S_old = _efs(
        _write(tmp_path, "legacy.json", legacy), "float64", use_kokkos
    )
    E_new, F_new, S_new = _efs(_write(tmp_path, "v4.json", v4), "float64", use_kokkos)
    assert np.isfinite(E_new) and np.all(np.isfinite(F_new))
    assert abs(E_new - E_old) <= 1e-9 * max(1.0, abs(E_old))
    assert np.max(np.abs(F_new - F_old)) <= 1e-10 * max(1.0, np.max(np.abs(F_old)))
    assert np.max(np.abs(S_new - S_old)) <= 1e-10 * max(1.0, np.max(np.abs(S_old)))


# Measured on CPU, one thread (synthetic K = 96 model, sigma_min 0.5 = "well", 1e-4 = "ill"; max fp32 |dF| vs
# legacy fp64): serial legacy 2.08e-7 -> 2.56e-6, v4 3.18e-7 -> 2.99e-7; host Kokkos legacy 7.05e-8 -> 4.43e-6,
# v4 4.31e-8 -> 6.68e-8.  The serial path evaluates everything in float, so its ill/well legacy ratio (12x) is
# bounded by its own float floor; the Kokkos path keeps the readout/skip sums in double (63x).  Thresholds are
# the measured ratios with a factor-2 margin; CUDA has no CPU measurement here and gets the generic bounds.
_FP32_RATIO_BOUNDS = {
    # path: (min legacy_ill/legacy_well, min legacy_ill/v4_ill, max v4_ill/v4_well)
    "serial": (6.0, 4.0, 2.0),
    "kokkos-host": (30.0, 30.0, 3.0),
    "kokkos-cuda": (None, 4.0, 3.0),
}


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_synthetic_fp32_amplification_is_removed(tmp_path, use_kokkos):
    """The legacy fp32 force error grows with the inverse-scaled coefficients; the
    m0_composite error of the same ill-conditioned model stays at the floor of the
    well-conditioned m0_composite model."""
    _native()
    errors = {}
    for label, sigma_min in (("well", 0.5), ("ill", 1e-4)):
        legacy, v4 = synthetic_model_pair(sigma_min)
        legacy_path = _write(tmp_path, f"legacy_{label}.json", legacy)
        v4_path = _write(tmp_path, f"v4_{label}.json", v4)
        _, F64, _ = _efs(legacy_path, "float64", use_kokkos)
        _, F32_legacy, _ = _efs(legacy_path, "float32", use_kokkos)
        _, F32_v4, _ = _efs(v4_path, "float32", use_kokkos)
        errors[label] = (
            float(np.max(np.abs(F32_legacy - F64))),
            float(np.max(np.abs(F32_v4 - F64))),
        )
    path = "serial"
    if use_kokkos:
        import symmetrix as native

        if not native._kokkos_is_initialized():
            native._init_kokkos()
        path = (
            "kokkos-cuda"
            if native._kokkos_default_execution_space() == "Cuda"
            else "kokkos-host"
        )
    print(f"H1_AFFINE_FP32 path={path} {errors}")
    (legacy_well, v4_well), (legacy_ill, v4_ill) = errors["well"], errors["ill"]
    min_amplification, min_removal, max_floor = _FP32_RATIO_BOUNDS[path]
    if min_amplification is not None:
        assert (
            legacy_ill >= min_amplification * legacy_well
        )  # amplification present (legacy basis)
    assert legacy_ill >= min_removal * v4_ill  # removed in the m0_composite basis
    assert v4_ill <= max_floor * v4_well  # v4 stays at its own well-conditioned floor


def test_synthetic_singular_linear_up_evaluates(tmp_path):
    _native()
    legacy, v4 = synthetic_model_pair(0.0)
    assert legacy is None  # the legacy representation does not exist
    E, F, S = _efs(_write(tmp_path, "v4_singular.json", v4), "float64", False)
    assert np.isfinite(E) and np.all(np.isfinite(F)) and np.all(np.isfinite(S))


def test_synthetic_v4_forces_match_finite_differences(tmp_path):
    _native()
    _, v4 = synthetic_model_pair(1e-4)
    path = _write(tmp_path, "v4.json", v4)
    _, F, _ = _efs(path, "float64", False)
    h = 1e-3
    count = len(_structure())
    for atom in (0, 1, 5):
        for component in range(3):
            d = np.zeros((count, 3))
            d[atom, component] = 1.0
            energies = [
                _efs(path, "float64", False, k * h * d)[0] for k in (2, 1, -1, -2)
            ]
            fd = (-energies[0] + 8 * energies[1] - 8 * energies[2] + energies[3]) / (
                12 * h
            )
            assert abs(-fd - F[atom, component]) <= 1e-6 * max(
                1.0, abs(F[atom, component])
            )
