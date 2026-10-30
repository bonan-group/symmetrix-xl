"""Format 4: no inverse of interactions[1].linear_up for any standard-MACE variant.

Pure tests (schema, conversion of every format-1/2 kind, exporter/adjoint algebra)
always run.  Native tests need an extension with format-4 support:

* synthetic compact models (``compact_r1_model``) with a controlled smallest
  singular value of U: the natively built v4 file and the v4 file converted from
  the inverse-carrying old file agree in fp64, and the fp32 error of v4 does not
  grow with the conditioning of U (the reproducer of PR #9);
* small mace-torch models of every two-interaction variant available in
  mace-torch (plain, first-interaction residual, density, density residual,
  pair-spline radial, multi-head) exported to v4 and compared with the fp64
  torch model, serial and host Kokkos;
* a MACEField payload built from a residual export plus an e3nn field coupling:
  zero field equals the field-free evaluation; forces, polarization,
  polarizability and BECs agree with finite differences;
* format-1/2 files still load and evaluate with the old arithmetic, with a
  warning that names the converter and carries a cancellation estimate
  computed from the file; the estimate is compared with the measured fp32
  force error of the old files over sigma_min(U).
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
    legacy_cancellation,
    recompose_legacy_payload,
    validate_payload,
)
from symmetrix.prediction_heads import select_prediction_head

CONVERTER = "symmetrix_recompose_mace_json"


def _recompose(payload):
    return recompose_legacy_payload(payload, source_sha256="0" * 64, source_bytes=1)


# --------------------------------------------------------------------------- synthetic payloads

def _factors(K=6, num_elements=6, L_max=1, sigma_min=1e-6, seed=3):
    rng = np.random.default_rng(seed)
    P = [rng.standard_normal((K, K)) / np.sqrt(K) for _ in range(L_max + 1)]
    U = []
    for l in range(L_max + 1):
        u, _, vt = np.linalg.svd(rng.standard_normal((K, K)))
        s = np.geomspace(1.0, sigma_min if l == 0 else 0.5, K)
        U.append((u * s) @ vt)
    r0 = rng.standard_normal(K) / np.sqrt(K)
    S0 = [rng.standard_normal((K, K)) / np.sqrt(K * num_elements) for _ in range(num_elements)]
    b = rng.standard_normal((num_elements, (L_max + 1) ** 2, K)) * 0.1
    return {"P": P, "U": U, "r0": r0, "S0": S0, "b": b, "K": K,
            "num_elements": num_elements, "L_max": L_max}


def _legacy_payload(f, *, split=False, field=False, version=2):
    """Old-basis payload: readout/skip multiplied by inv(U0) as the old exporter did."""
    K, nz, L = f["K"], f["num_elements"], f["L_max"]
    C = np.stack([f["P"][l] @ f["U"][l] for l in range(L + 1)])
    payload = {
        "num_channels": K, "L_max": L, "num_interactions": 2,
        "model_type": "MACEField" if field else "MACE",
        "has_field_coupling": field, "first_interaction_residual": split and not field,
        "field_couplings": [],
        "atomic_numbers": list(range(1, nz + 1)), "num_elements": nz,
        "H1_weights": C.reshape(-1).tolist(),
        "H2_weights_for_H1": [np.linalg.solve(f["U"][0], s).reshape(-1).tolist() for s in f["S0"]],
        "H2_weights_for_M1": np.eye(K).reshape(-1).tolist(),
        "readout_1_weights": np.linalg.solve(f["U"][0], f["r0"]).tolist(),
    }
    if version == 2:
        payload.update(symmetrix_format_version=2, radial_representation="compact")
    else:
        payload.update(radial_spline_h=0.1, radial_spline_values_0=[[[0.0]]])
    if split or field:
        payload["H1_product_weights"] = np.stack(f["P"]).reshape(-1).tolist()
        payload["H1_linear_up_weights"] = np.stack(f["U"]).reshape(-1).tolist()
    if split and not field:
        payload["H1_first_residual_weights"] = f["b"].reshape(-1).tolist()
    return payload


# --------------------------------------------------------------------------- conversion

def test_plain_v2_converts_to_m0_composite():
    f = _factors(sigma_min=1e-4)
    old = _legacy_payload(f)
    new, checks = _recompose(old)
    assert new["symmetrix_format_version"] == 4
    assert new["h1_affine_mode"] == "m0_composite" and new["h1_affine_version"] == 1
    assert new["radial_representation"] == "compact"
    assert "readout_1_weights" not in new and "H2_weights_for_H1" not in new
    assert new["H1_weights"] == old["H1_weights"]
    q = np.asarray(new["readout_1_weights_for_M0"])
    T = np.asarray(new["H2_weights_for_M0"]).reshape(6, 6, 6)
    np.testing.assert_allclose(q, f["P"][0] @ f["r0"], rtol=0, atol=1e-10)
    for z in range(6):
        np.testing.assert_allclose(T[z], f["P"][0] @ f["S0"][z], rtol=0, atol=1e-10)
    assert checks["mode"] == "m0_composite"
    assert h1_affine_mode_of(new) == "m0_composite"
    assert new["h1_affine_provenance"]["origin"] == "converted_from_v2"


@pytest.mark.parametrize("field", [False, True])
def test_split_v2_converts_to_pre_linear_up(field):
    """Residual-first and MACEField files keep P and U split: the converted maps
    are the checkpoint's own r0, S0 on the pre-linear_up features (r0 = U0 r)."""
    f = _factors(sigma_min=1e-5, seed=4)
    old = _legacy_payload(f, split=True, field=field)
    new, checks = _recompose(old)
    assert new["h1_affine_mode"] == "pre_linear_up"
    r0 = np.asarray(new["readout_1_weights_for_H1_pre_linear_up"])
    S0 = np.asarray(new["H2_weights_for_H1_pre_linear_up"]).reshape(6, 6, 6)
    np.testing.assert_allclose(r0, f["r0"], rtol=0, atol=1e-9)
    for z in range(6):
        np.testing.assert_allclose(S0[z], f["S0"][z], rtol=0, atol=1e-9)
    assert "readout_1_weights_for_M0" not in new
    assert checks["E1_max_rel_terms"] <= 1e-12
    assert new["H1_product_weights"] == old["H1_product_weights"]
    validate_payload(new)


def test_multihead_v2_converts_every_head():
    f = _factors(sigma_min=1e-3, seed=5)
    old = _legacy_payload(f)
    r_head = np.linalg.solve(f["U"][0], 0.5 * f["r0"])
    old["prediction_heads"] = {
        "schema_version": 1, "names": ["a", "b"], "default": "a",
        "parameters": {
            "a": {"atomic_energies": [0.0] * 6, "readout_1_weights": old["readout_1_weights"],
                  "readout_2_weights_1": [1.0], "readout_2_weights_2": [1.0]},
            "b": {"atomic_energies": [1.0] * 6, "readout_1_weights": r_head.tolist(),
                  "readout_2_weights_1": [2.0], "readout_2_weights_2": [2.0]},
        },
    }
    new, _ = _recompose(old)
    heads = new["prediction_heads"]["parameters"]
    assert all("readout_1_weights" not in payload for payload in heads.values())
    np.testing.assert_allclose(heads["b"]["readout_1_weights_for_M0"],
                               f["P"][0] @ (0.5 * f["r0"]), rtol=0, atol=1e-10)
    selected = select_prediction_head(new, "b")
    assert selected["readout_1_weights_for_M0"] == heads["b"]["readout_1_weights_for_M0"]
    validate_payload(new)


def test_v1_pair_splines_converts_and_keeps_tables():
    f = _factors(sigma_min=1e-3, seed=6)
    old = _legacy_payload(f, version=1)
    assert "symmetrix_format_version" not in old
    new, _ = _recompose(old)
    assert new["symmetrix_format_version"] == 4
    assert new["radial_representation"] == "pair-splines"
    assert new["radial_spline_values_0"] == old["radial_spline_values_0"]
    assert new["h1_affine_provenance"]["origin"] == "converted_from_v1"
    assert new["h1_affine_provenance"]["source_format_version"] == 1
    validate_payload(new)


def test_single_layer_converts_to_zero_maps():
    f = _factors(sigma_min=0.5, seed=7)
    old = _legacy_payload(f)
    old.update(num_interactions=1, single_layer_readout=True,
               readout_1_weights=[0.0] * 6,
               H2_weights_for_H1=[[0.0] * 36 for _ in range(6)])
    new, _ = _recompose(old)
    assert new["h1_affine_mode"] == "m0_composite"
    assert new["readout_1_weights_for_M0"] == [0.0] * 6
    assert all(row == [0.0] * 36 for row in new["H2_weights_for_M0"])


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda d: d.update(model_type="MACE_Nonlinear"), "needs no conversion"),
        (lambda d: d.update(symmetrix_format_version=3), "unsupported"),
        (lambda d: d.update(symmetrix_format_version=4), "already"),
        (lambda d: d.update(H2_weights_for_M0=[[0.0]]), "must not declare"),
        (lambda d: d.update(readout_1_weights=[1.0, 2.0]), "readout_1_weights"),
        (lambda d: d.update(readout_1_weights=[float("nan")] * 6), "non-finite"),
        (lambda d: d.update(radial_representation="pair-splines"), "compact"),
        (lambda d: d.update(first_interaction_residual=True), "H1_linear_up_weights"),
    ],
)
def test_conversion_refusals(mutation, match):
    payload = _legacy_payload(_factors())
    mutation(payload)
    with pytest.raises(H1AffineError, match=match):
        _recompose(payload)


def test_old_formats_validate_as_legacy_with_warning():
    for version in (1, 2):
        payload = _legacy_payload(_factors(sigma_min=1e-4), version=version)
        assert h1_affine_mode_of(payload) == "legacy"
        summary = validate_payload(payload)
        assert summary["mode"] == "legacy" and CONVERTER in summary["warning"]
        assert summary["cancellation"]["readout"] > 10.0
    bad = _legacy_payload(_factors())
    bad["h1_affine_mode"] = "m0_composite"
    with pytest.raises(H1AffineError, match="must not declare"):
        h1_affine_mode_of(bad)


def test_cancellation_estimate_grows_with_inverse_amplification():
    """The estimate is ~1 for zero maps, moderate for a well-conditioned U and
    grows like 1/sigma_min(U) (pure algebra; fp32 behaviour is measured in the
    native tests)."""
    values = []
    for sigma_min in (0.5, 1e-2, 1e-4, 1e-6):
        values.append(legacy_cancellation(_legacy_payload(_factors(sigma_min=sigma_min, seed=12))))
    print(f"H1_AFFINE_ESTIMATE_PURE {values}")
    readout = [value["readout"] for value in values]
    assert readout == sorted(readout)
    assert readout[-1] >= 1e3 * readout[0]
    zero = _legacy_payload(_factors())
    zero["readout_1_weights"] = [0.0] * 6
    zero["H2_weights_for_H1"] = [[0.0] * 36 for _ in range(6)]
    assert legacy_cancellation(zero) == {"readout": 1.0, "skip": 1.0}


def _as_m0_composite(d):
    d["h1_affine_mode"] = "m0_composite"
    d["readout_1_weights_for_M0"] = d.pop("readout_1_weights_for_H1_pre_linear_up")
    d["H2_weights_for_M0"] = d.pop("H2_weights_for_H1_pre_linear_up")


def test_validator_rejects_mixed_and_partial_v4():
    new, _ = _recompose(_legacy_payload(_factors()))
    split, _ = _recompose(_legacy_payload(_factors(), split=True))
    validate_payload(new)
    for base, mutate, match in (
        (new, lambda d: d.__setitem__("readout_1_weights", [0.0] * 6), "legacy keys"),
        (new, lambda d: d.__setitem__("H2_weights_for_H1", d["H2_weights_for_M0"]), "legacy keys"),
        (new, lambda d: d.pop("H2_weights_for_M0"), "missing"),
        (new, lambda d: d.__setitem__("h1_affine_mode", "pre_linear"), "h1_affine_mode"),
        (new, lambda d: d.__setitem__("h1_affine_mode", "pre_linear_up"), "must not contain"),
        (new, lambda d: d.__setitem__("h1_affine_version", 2), "h1_affine_version"),
        (new, lambda d: d.__setitem__("symmetrix_format_version", 3), "unsupported"),
        (new, lambda d: d["H2_weights_for_M0"].pop(), "one row per element"),
        (new, lambda d: d.__setitem__("readout_1_weights_for_M0", [0.0] * 5), "6 values"),
        (new, lambda d: d.__setitem__("readout_1_weights_for_H1_pre_linear_up", [0.0] * 6),
         "must not contain"),
        (split, _as_m0_composite, "cannot represent"),
    ):
        bad = copy.deepcopy(base)
        mutate(bad)
        with pytest.raises(H1AffineError, match=match):
            validate_payload(bad)


def test_conversion_cli_never_overwrites(tmp_path):
    from symmetrix.cli.recompose_mace_json import main

    src = tmp_path / "old.json"
    src.write_text(json.dumps(_legacy_payload(_factors(), split=True)))
    dst = tmp_path / "new" / "v4.json"
    assert main(["-i", str(src), "-o", str(dst), "--receipt", str(tmp_path / "r.json")]) == 0
    first = dst.read_bytes()
    assert main(["-i", str(src), "-o", str(dst)]) == 2
    assert dst.read_bytes() == first
    assert main(["-i", str(dst), "-o", str(tmp_path / "again.json")]) == 3
    assert main(["--validate-only", str(dst)]) == 0
    assert main(["--validate-only", str(src)]) == 0  # old file: valid, with the warning
    receipt = json.loads((tmp_path / "r.json").read_text())
    assert receipt["h1_affine_mode"] == "pre_linear_up"
    assert receipt["origin"] == "converted_from_v2" and receipt["source_format_version"] == 2
    assert receipt["identity_fp64"]["detail"]["worst_rel_terms"] <= 1e-12


# --------------------------------------------------------------------------- algebra

@pytest.mark.parametrize("sigma_min", [1e-2, 1e-6, 1e-10, 0.0])
def test_native_maps_need_no_inverse(sigma_min):
    """q = P0 r0 and T = P0 S0 exist for ill-conditioned and singular U; the old
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
    q, T = P0 @ r0, P0 @ S0
    m = rng.standard_normal((32, K))
    h_pre = m @ P0
    np.testing.assert_allclose(m @ q, h_pre @ r0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(m @ T, h_pre @ S0, rtol=1e-12, atol=1e-12)
    if sigma_min > 0.0:
        r = np.linalg.solve(U0, r0)
        assert np.linalg.norm(r) >= 0.1 * abs(u[:, -1] @ r0) / sigma_min
    else:
        assert np.linalg.matrix_rank(U0) < K


def test_reverse_algebra_matches_finite_differences():
    """Composite reverse a C^T + q + T g2 equals the old (a + r + S g2) C^T and
    central finite differences (six species, ill-conditioned U)."""
    f = _factors(K=6, sigma_min=1e-5, seed=9)
    old = _legacy_payload(f)
    new, _ = _recompose(old)
    K, nz, N = 6, 6, 12
    C = np.asarray(old["H1_weights"]).reshape(2, K, K)[0]
    r = np.asarray(old["readout_1_weights"])
    S = np.asarray(old["H2_weights_for_H1"]).reshape(nz, K, K)
    q = np.asarray(new["readout_1_weights_for_M0"])
    T = np.asarray(new["H2_weights_for_M0"]).reshape(nz, K, K)
    rng = np.random.default_rng(5)
    types = np.arange(N) % nz
    m = rng.standard_normal((N, K))
    msg_w = rng.standard_normal(K)
    W2 = rng.standard_normal((K, K)) / K
    M1 = rng.standard_normal((N, K))
    Tz = T[types]

    def energy(mm):
        h = mm @ C
        h2 = np.einsum("nk,nkc->nc", mm, Tz) + M1 @ W2
        return np.sum(np.sin(h @ msg_w)) + np.sum(mm @ q) + np.sum(np.logaddexp(0.0, h2))

    h = m @ C
    a = np.cos(h @ msg_w)[:, None] * msg_w[None, :]
    h2 = np.einsum("nk,nkc->nc", m, Tz) + M1 @ W2
    g2 = 1.0 / (1.0 + np.exp(-h2))
    g_old = (a + r[None, :] + np.einsum("nkc,nc->nk", S[types], g2)) @ C.T
    g_new = a @ C.T + q[None, :] + np.einsum("nkc,nc->nk", Tz, g2)
    assert np.max(np.abs(g_old - g_new)) <= 1e-8 * max(np.max(np.abs(g_new)), 1.0)
    eps = 1e-6
    for (i, k) in [(0, 0), (1, 3), (2, 5), (3, 1), (4, 2), (5, 4)]:
        dm = np.zeros_like(m)
        dm[i, k] = eps
        fd = (energy(m + dm) - energy(m - dm)) / (2 * eps)
        assert abs(fd - g_new[i, k]) <= 1e-6 * max(1.0, abs(fd))


def test_identity_check_reports_per_element():
    f = _factors()
    old = _legacy_payload(f)
    new, checks = _recompose(old)
    assert len(checks["per_element"]) == 6
    C = np.asarray(old["H1_weights"]).reshape(2, 6, 6)[0]
    again = identity_check(
        C, np.asarray(old["readout_1_weights"]),
        np.asarray(old["H2_weights_for_H1"]).reshape(6, 6, 6),
        np.asarray(new["readout_1_weights_for_M0"]),
        np.asarray(new["H2_weights_for_M0"]).reshape(6, 6, 6))
    assert all(row["H2_max_rel_terms"] <= 1e-12 for row in again["per_element"])
    assert again["E1_max_rel_terms"] <= 1e-12


# --------------------------------------------------------------------------- native helpers

_CONTRACT = (
    Path(__file__).resolve().parent / "data" / "execution_contracts"
    / "jit_r1_mace_off23_small_contract.json"
)


def _native():
    nc = pytest.importorskip("symmetrix._native_cpu")
    if not hasattr(nc.MACE, "h1_affine_mode"):
        pytest.skip("native extension without format-4 (h1_affine_mode) support")
    return nc


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _structure(symbols=("H", "O", "O"), seed=11, count=12, box=6.0):
    from ase import Atoms

    rng = np.random.default_rng(seed)
    return Atoms(
        [symbols[i % len(symbols)] for i in range(count)],
        positions=rng.uniform(0.0, box, size=(count, 3)),
        cell=[box, box, box],
        pbc=True,
    )


def _evaluate(model_path, dtype, use_kokkos, atoms, electric_field=None, head=None,
              properties=("energy", "forces", "stress"), streamed_edges=None):
    """Results of one fresh calculator; the evaluator is released afterwards so
    Kokkos can be finalized at session end."""
    from symmetrix import Symmetrix

    kwargs = {}
    if use_kokkos:
        kwargs["streamed_edges"] = streamed_edges or "generic"
    if electric_field is not None:
        kwargs["electric_field"] = np.asarray(electric_field, dtype=float)
    if head is not None:
        kwargs["head"] = head
    calculator = Symmetrix(str(model_path), use_kokkos=use_kokkos, dtype=dtype, **kwargs)
    atoms = atoms.copy()
    try:
        calculator.calculate(atoms, properties=list(properties))
        return {key: np.array(calculator.results[key], copy=True) for key in properties}
    finally:
        del calculator
        gc.collect()


# --------------------------------------------------------------------------- synthetic compact model

def synthetic_model_pair(sigma_min, seed=5):
    """(old-basis, v4) payloads of one synthetic compact model, and the v4 file
    converted from the old one.  U0 = Q1 diag(geomspace(1, sigma_min)) Q2^T,
    C = P0 U0; old: inv(U0) r0, inv(U0) S0_z; v4: P0 r0, P0 S0_z (no inverse).
    sigma_min = 0 (singular U0) returns old = converted = None."""
    from compact_r1_model import foundation_r1_test_model

    model = foundation_r1_test_model(json.loads(_CONTRACT.read_text()), atomic_numbers=(1, 8))
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
    v4.update(
        readout_1_weights_for_M0=(P0 @ r0).tolist(),
        H2_weights_for_M0=[(P0 @ S0[z]).reshape(-1).tolist() for z in range(nz)],
    )
    if sigma_min == 0.0:
        return None, v4, None
    U_inv = np.linalg.inv(U0)  # test only: the old exporter's arithmetic
    old = copy.deepcopy(model)
    for key in ("h1_affine_mode", "h1_affine_version",
                "readout_1_weights_for_M0", "H2_weights_for_M0"):
        old.pop(key)
    old.update(
        symmetrix_format_version=2,
        readout_1_weights=(U_inv @ r0).tolist(),
        H2_weights_for_H1=[(U_inv @ S0[z]).reshape(-1).tolist() for z in range(nz)],
    )
    converted, _ = _recompose(old)
    return old, v4, converted


def test_loaders_warn_and_evaluate_old_formats(tmp_path, capfd):
    """Format 2 loads with the old arithmetic: one warning per load naming the
    converter, with the same estimate as legacy_cancellation; the old file and
    its converted v4 file evaluate the same model."""
    nc = _native()
    old, v4, converted = synthetic_model_pair(1e-3)
    path = _write(tmp_path, "old.json", old)
    estimate = legacy_cancellation(old)
    expected = f"readout {estimate['readout']:.3g}, skip {estimate['skip']:.3g}"
    for cls in (nc.MACE, nc.MACEFloat):
        assert cls(str(path), "").h1_affine_mode == "legacy"
        err = capfd.readouterr().err
        assert "Warning:" in err and CONVERTER in err and expected in err, err
        assert "Re-export to v4 is recommended but optional." in err, err
    atoms = _structure()
    for use_kokkos in (False, True):
        a = _evaluate(path, "float64", use_kokkos, atoms)
        b = _evaluate(_write(tmp_path, "v4.json", v4), "float64", use_kokkos, atoms)
        err = capfd.readouterr().err
        assert CONVERTER in err
        with capfd.disabled():
            print(f"H1_AFFINE_LEGACY_VS_V4 kokkos={use_kokkos} "
                  f"dE={abs(a['energy'] - b['energy']):.3e} "
                  f"dF={np.max(np.abs(a['forces'] - b['forces'])):.3e}")
        assert abs(a["energy"] - b["energy"]) <= 1e-9 * max(1.0, abs(a["energy"]))
        assert np.max(np.abs(a["forces"] - b["forces"])) <= 1e-9


def test_synthetic_dispatch(tmp_path):
    nc = _native()
    _, v4, converted = synthetic_model_pair(1e-4)
    validate_payload(v4)
    np.testing.assert_allclose(
        converted["readout_1_weights_for_M0"], v4["readout_1_weights_for_M0"], rtol=0, atol=1e-9)
    path = _write(tmp_path, "v4.json", v4)
    for cls in (nc.MACE, nc.MACEFloat):
        assert cls(str(path), "").h1_affine_mode == "m0_composite"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.__setitem__("readout_1_weights", d["readout_1_weights_for_M0"]),
        lambda d: d.pop("h1_affine_mode"),
        lambda d: d.__setitem__("h1_affine_version", 2),
        lambda d: d.__setitem__("symmetrix_format_version", 3),
        lambda d: d["H2_weights_for_M0"][0].pop(),
        lambda d: d.__setitem__("has_field_coupling", True),
        lambda d: d.__setitem__("h1_affine_mode", "pre_linear_up"),
    ],
)
def test_synthetic_loader_refuses_malformed_v4(tmp_path, mutation):
    nc = _native()
    _, v4, _ = synthetic_model_pair(1e-2)
    mutation(v4)
    path = _write(tmp_path, "bad.json", v4)
    with pytest.raises((ValueError, RuntimeError)):
        nc.MACE(str(path), "")


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_synthetic_converted_file_matches_native_v4(tmp_path, use_kokkos):
    _native()
    _, v4, converted = synthetic_model_pair(1e-4)
    atoms = _structure()
    new = _evaluate(_write(tmp_path, "v4.json", v4), "float64", use_kokkos, atoms)
    conv = _evaluate(_write(tmp_path, "conv.json", converted), "float64", use_kokkos, atoms)
    assert np.isfinite(new["energy"]) and np.all(np.isfinite(new["forces"]))
    print(f"H1_AFFINE_CONVERTED kokkos={use_kokkos} "
          f"dE={abs(new['energy'] - conv['energy']):.3e} "
          f"dF={np.max(np.abs(new['forces'] - conv['forces'])):.3e}")
    assert abs(new["energy"] - conv["energy"]) <= 1e-9 * max(1.0, abs(new["energy"]))
    assert np.max(np.abs(new["forces"] - conv["forces"])) <= 1e-9
    assert np.max(np.abs(new["stress"] - conv["stress"])) <= 1e-9


_SWEEP = (0.5, 1e-1, 1e-2, 1e-3, 1e-4)


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_synthetic_fp32_error_and_cancellation_estimate(tmp_path, use_kokkos):
    """PR #9 reproducer over sigma_min(U): fp32 force error of the old file and
    of the v4 file against the v4 fp64 result, and the old file's cancellation
    estimate.  The v4 error stays at its floor; the old error grows with the
    estimate."""
    _native()
    atoms = _structure()
    rows = []
    for sigma_min in _SWEEP:
        old, v4, _ = synthetic_model_pair(sigma_min)
        old_path = _write(tmp_path, f"old_{sigma_min:g}.json", old)
        v4_path = _write(tmp_path, f"v4_{sigma_min:g}.json", v4)
        F64 = _evaluate(v4_path, "float64", use_kokkos, atoms)["forces"]
        old32 = _evaluate(old_path, "float32", use_kokkos, atoms)["forces"]
        v432 = _evaluate(v4_path, "float32", use_kokkos, atoms)["forces"]
        estimate = legacy_cancellation(old)
        rows.append((sigma_min, estimate["readout"], estimate["skip"],
                     float(np.max(np.abs(old32 - F64))), float(np.max(np.abs(v432 - F64))),
                     float(np.linalg.norm(old["readout_1_weights"]))))
    for row in rows:
        print(f"H1_AFFINE_SWEEP kokkos={use_kokkos} sigma_min={row[0]:.0e} "
              f"estimate readout={row[1]:.3e} skip={row[2]:.3e} |r_old|={row[5]:.3e} "
              f"fp32 max|dF| old={row[3]:.3e} v4={row[4]:.3e}")
    estimates = [max(row[1], row[2]) for row in rows]
    assert estimates == sorted(estimates)
    old_err = [row[3] for row in rows]
    v4_err = [row[4] for row in rows]
    assert old_err[-1] >= 5.0 * old_err[0]
    assert max(v4_err) <= 3.0 * max(min(v4_err), 1e-7)


def test_synthetic_singular_linear_up_evaluates(tmp_path):
    _native()
    _, v4, _ = synthetic_model_pair(0.0)
    result = _evaluate(_write(tmp_path, "v4_singular.json", v4), "float64", False, _structure())
    assert all(np.all(np.isfinite(value)) for value in result.values())


def test_synthetic_v4_forces_match_finite_differences(tmp_path):
    _native()
    _, v4, _ = synthetic_model_pair(1e-4)
    path = _write(tmp_path, "v4.json", v4)
    atoms = _structure()
    F = _evaluate(path, "float64", False, atoms)["forces"]
    h = 1e-3
    for atom in (0, 1, 5):
        for component in range(3):
            energies = []
            for k in (2, 1, -1, -2):
                displaced = atoms.copy()
                displaced.positions[atom, component] += k * h
                energies.append(_evaluate(path, "float64", False, displaced)["energy"])
            fd = (-energies[0] + 8 * energies[1] - 8 * energies[2] + energies[3]) / (12 * h)
            assert abs(-fd - F[atom, component]) <= 1e-6 * max(1.0, abs(F[atom, component]))


# --------------------------------------------------------------------------- mace-torch variants

def _mace_modules():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mace")
    from e3nn import o3
    from mace import modules
    from mace.calculators import MACECalculator

    return torch, o3, modules, MACECalculator


def _ill_condition_linear_up(torch, model, sigma_min, seed):
    """Replace the scalar block of interactions[1].linear_up by Q1 diag(s) Q2^T
    (the --from-mace variant of the PR #9 reproducer)."""
    linear_up = model.interactions[1].linear_up
    rng = np.random.default_rng(seed)
    K = model.node_embedding.linear.irreps_out.count("0e")
    q1, _, q2 = np.linalg.svd(rng.standard_normal((K, K)))
    if sigma_min == 0.0:
        block = np.zeros((K, K))  # exactly singular (sigma_min_U0 = 0)
    else:
        block = (q1 * np.geomspace(1.0, sigma_min, K)) @ q2 * np.sqrt(K)
    with torch.no_grad():
        weight = linear_up.weight_view_for_instruction(0)
        weight.copy_(torch.as_tensor(block, dtype=weight.dtype).reshape(weight.shape))


def _mace_model(variant, sigma_min=1e-3, heads=None, seed=0):
    torch, o3, modules, _ = _mace_modules()
    torch.manual_seed(seed)
    blocks = modules.interaction_classes
    first, second = {
        "plain": ("RealAgnosticInteractionBlock", "RealAgnosticResidualInteractionBlock"),
        "residual": ("RealAgnosticResidualInteractionBlock", "RealAgnosticResidualInteractionBlock"),
        "density": ("RealAgnosticDensityInteractionBlock",
                    "RealAgnosticDensityResidualInteractionBlock"),
        "density_residual": ("RealAgnosticDensityResidualInteractionBlock",
                             "RealAgnosticDensityResidualInteractionBlock"),
    }[variant]
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        num_heads = 1 if heads is None else len(heads)
        energies = np.array([[-1.0, -2.5, -3.0]] * num_heads) + 0.1 * np.arange(num_heads)[:, None]
        model = modules.ScaleShiftMACE(
            r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=3,
            interaction_cls=blocks[second], interaction_cls_first=blocks[first],
            num_interactions=2, num_elements=3,
            hidden_irreps=o3.Irreps("8x0e + 8x1o"), MLP_irreps=o3.Irreps("8x0e"),
            atomic_energies=energies if heads is not None else energies[0],
            avg_num_neighbors=6.0, atomic_numbers=[1, 6, 8], correlation=3,
            gate=torch.nn.functional.silu, radial_MLP=[16, 16],
            atomic_inter_scale=[1.3, 0.8][:num_heads] if heads else 1.3,
            atomic_inter_shift=[0.1, -0.2][:num_heads] if heads else 0.1,
            heads=heads, use_reduced_cg=False,
        )
    finally:
        torch.set_default_dtype(previous)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    _ill_condition_linear_up(torch, model, sigma_min, seed + 1)
    return model.double().eval()


def _torch_reference(model, atoms, head=None):
    _, _, _, MACECalculator = _mace_modules()
    kwargs = {"head": head} if head is not None else {}
    calculator = MACECalculator(models=[model], device="cpu", default_dtype="float64", **kwargs)
    atoms = atoms.copy()
    atoms.calc = calculator
    return float(atoms.get_potential_energy()), np.array(atoms.get_forces(), copy=True)


def _variant_atoms():
    return _structure(symbols=("H", "C", "O"), count=9, box=4.5, seed=21)


def _assert_matches_torch(result, reference, label):
    E_ref, F_ref = reference
    dE = abs(result["energy"] - E_ref)
    dF = float(np.max(np.abs(result["forces"] - F_ref)))
    print(f"H1_AFFINE_TORCH {label} |dE|={dE:.3e} max|dF|={dF:.3e} |F|max={np.max(np.abs(F_ref)):.3e}")
    assert dE <= 1e-5 * max(1.0, abs(E_ref))
    assert dF <= 1e-4 * max(1.0, float(np.max(np.abs(F_ref))))


@pytest.mark.parametrize("radial_format", ["compact", "pair-splines"])
@pytest.mark.parametrize("variant", ["plain", "residual", "density", "density_residual"])
def test_mace_variants_match_torch_fp64(tmp_path, variant, radial_format):
    _native()
    from symmetrix.extract_mace_data import _extract_single_head_data

    model = _mace_model(variant, sigma_min=1e-3)
    data = _extract_single_head_data(model, num_spline_points=1000, radial_format=radial_format)
    expected = "m0_composite" if variant in ("plain", "density") else "pre_linear_up"
    assert data["symmetrix_format_version"] == 4 and data["h1_affine_mode"] == expected
    assert data["radial_representation"] == radial_format
    assert "readout_1_weights" not in data and "H2_weights_for_H1" not in data
    provenance = data["h1_affine_provenance"]
    assert provenance["origin"] == "native_checkpoint"
    cond = provenance["conditioning"]
    assert cond["cond_U0"] == pytest.approx(cond["sigma_max_U0"] / cond["sigma_min_U0"])
    assert cond["cond_U0"] == pytest.approx(1e3, rel=1e-6)
    validate_payload(data)
    path = _write(tmp_path, f"{variant}.json", data)
    atoms = _variant_atoms()
    reference = _torch_reference(model, atoms)
    for use_kokkos in (False, True):
        if radial_format == "pair-splines" and use_kokkos:
            result = _evaluate(path, "float64", True, atoms, streamed_edges="materialized")
        else:
            result = _evaluate(path, "float64", use_kokkos, atoms)
        _assert_matches_torch(result, reference, f"{variant}/{radial_format}/kokkos={use_kokkos}")


@pytest.mark.parametrize("variant", ["plain", "residual"])
def test_mace_singular_linear_up_exports_and_matches_torch(tmp_path, variant):
    """A singular scalar block of linear_up has no inverse; the v4 export needs
    none (cond_U0 is recorded as null) and still evaluates the model."""
    _native()
    from symmetrix.extract_mace_data import _extract_single_head_data

    model = _mace_model(variant, sigma_min=0.0)
    data = _extract_single_head_data(model, num_spline_points=1000)
    cond = data["h1_affine_provenance"]["conditioning"]
    assert cond["sigma_min_U0"] == 0.0 and cond["cond_U0"] is None
    json.dumps(data, allow_nan=False)
    validate_payload(data)
    path = _write(tmp_path, f"{variant}_singular.json", data)
    atoms = _variant_atoms()
    reference = _torch_reference(model, atoms)
    for use_kokkos in (False, True):
        result = _evaluate(path, "float64", use_kokkos, atoms)
        _assert_matches_torch(result, reference, f"{variant}/singular/kokkos={use_kokkos}")


def test_mace_multihead_matches_torch_fp64(tmp_path):
    _native()
    torch, _, _, _ = _mace_modules()
    from mace.tools.scripts_utils import remove_pt_head
    from symmetrix.extract_mace_data import extract_mace_data

    model = _mace_model("plain", sigma_min=1e-3, heads=["first", "second"])
    checkpoint = tmp_path / "multihead.model"
    torch.save(model, checkpoint)
    data = extract_mace_data(str(checkpoint), head="first", num_spline_points=1000)
    assert data["h1_affine_mode"] == "m0_composite"
    for payload in data["prediction_heads"]["parameters"].values():
        assert "readout_1_weights_for_M0" in payload and "readout_1_weights" not in payload
    validate_payload(data)
    path = _write(tmp_path, "multihead.json", data)
    atoms = _variant_atoms()
    for head in ("first", "second"):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            single = remove_pt_head(copy.deepcopy(model), head).double().eval()
        finally:
            torch.set_default_dtype(previous)
        reference = _torch_reference(single, atoms)
        for use_kokkos in (False, True):
            result = _evaluate(path, "float64", use_kokkos, atoms, head=head)
            _assert_matches_torch(result, reference, f"multihead/{head}/kokkos={use_kokkos}")


@pytest.mark.parametrize("variant", ["plain", "residual"])
def test_mace_variants_fp32_error_independent_of_linear_up_conditioning(tmp_path, variant):
    """--from-mace form of the reproducer: fp32 vs fp64 of the same v4 file for
    sigma_min(U) = 0.5 and 1e-5."""
    _native()
    from symmetrix.extract_mace_data import _extract_single_head_data

    atoms = _variant_atoms()
    for use_kokkos in (False, True):
        errors = {}
        for label, sigma_min in (("well", 0.5), ("ill", 1e-5)):
            data = _extract_single_head_data(_mace_model(variant, sigma_min=sigma_min),
                                             num_spline_points=256)
            path = _write(tmp_path, f"{variant}_{label}.json", data)
            F64 = _evaluate(path, "float64", use_kokkos, atoms)["forces"]
            F32 = _evaluate(path, "float32", use_kokkos, atoms)["forces"]
            errors[label] = float(np.max(np.abs(F32 - F64)))
        print(f"H1_AFFINE_TORCH_FP32 {variant} kokkos={use_kokkos} max|dF| {errors}")
        assert errors["ill"] <= 3.0 * max(errors["well"], 1e-7)


# --------------------------------------------------------------------------- MACEField (pre_linear_up + field)

def _field_coupling(K, seed=3):
    """Compact MACEField coupling (schema 1) from e3nn field_feats/field_linear
    modules, serialized like extract_mace_data does for a MACEField checkpoint."""
    torch, o3, _, _ = _mace_modules()
    torch.manual_seed(seed)
    hidden = o3.Irreps(f"{K}x0e + {K}x1o")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        field_feats = o3.FullyConnectedTensorProduct(hidden, o3.Irreps("1x1o"), hidden)
        field_linear = o3.Linear(hidden, hidden)
    finally:
        torch.set_default_dtype(previous)
    with torch.no_grad():
        field_feats.weight.mul_(0.5)

    def instruction(ins):
        out = {}
        for name in ("i_in", "i_in1", "i_in2", "i_out", "connection_mode",
                     "has_weight", "path_weight", "path_shape"):
            if hasattr(ins, name):
                value = getattr(ins, name)
                out[name] = list(value) if isinstance(value, tuple) else value
        return out

    feats = []
    for ins in field_feats.instructions:
        serialized = instruction(ins)
        l_in = field_feats.irreps_in1[ins.i_in1][1].l
        l_field = field_feats.irreps_in2[ins.i_in2][1].l
        l_out = field_feats.irreps_out[ins.i_out][1].l
        wigner = o3.wigner_3j(l_in, l_field, l_out).to(dtype=torch.float64)
        serialized["wigner_3j_shape"] = list(wigner.shape)
        serialized["wigner_3j"] = wigner.flatten().tolist()
        feats.append(serialized)
    return {
        "schema_version": 1,
        "field_feats_irreps_in1": str(field_feats.irreps_in1),
        "field_feats_irreps_in2": str(field_feats.irreps_in2),
        "field_feats_irreps_out": str(field_feats.irreps_out),
        "field_feats_instructions": feats,
        "field_feats_weight": field_feats.weight.detach().numpy().tolist(),
        "field_feats_output_mask": field_feats.output_mask.numpy().tolist(),
        "field_linear_irreps_in": str(field_linear.irreps_in),
        "field_linear_irreps_out": str(field_linear.irreps_out),
        "field_linear_instructions": [instruction(ins) for ins in field_linear.instructions],
        "field_linear_weight": field_linear.weight.detach().numpy().tolist(),
        "field_linear_bias": field_linear.bias.detach().numpy().tolist(),
        "field_linear_output_mask": field_linear.output_mask.numpy().tolist(),
    }


@pytest.fixture(scope="module")
def field_models(tmp_path_factory):
    """(field-free residual v4 JSON, MACEField v4 JSON) sharing every weight."""
    pytest.importorskip("symmetrix._native_cpu")
    from symmetrix.extract_mace_data import _extract_single_head_data

    data = _extract_single_head_data(_mace_model("residual", sigma_min=1e-3, seed=7),
                                     num_spline_points=500)
    assert data["h1_affine_mode"] == "pre_linear_up"
    field = copy.deepcopy(data)
    field.update(model_type="MACEField", has_field_coupling=True,
                 field_couplings=[_field_coupling(data["num_channels"])])
    validate_payload(field)
    root = tmp_path_factory.mktemp("h1-affine-field")
    plain_path = root / "residual.json"
    field_path = root / "field.json"
    plain_path.write_text(json.dumps(data))
    field_path.write_text(json.dumps(field))
    return plain_path, field_path


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_field_zero_field_matches_field_free_model(field_models, use_kokkos):
    _native()
    plain_path, field_path = field_models
    atoms = _variant_atoms()
    plain = _evaluate(plain_path, "float64", use_kokkos, atoms)
    field = _evaluate(field_path, "float64", use_kokkos, atoms, electric_field=[0.0, 0.0, 0.0])
    print(f"H1_AFFINE_FIELD_ZERO kokkos={use_kokkos} dE={abs(plain['energy'] - field['energy']):.3e} "
          f"dF={np.max(np.abs(plain['forces'] - field['forces'])):.3e}")
    assert abs(plain["energy"] - field["energy"]) <= 1e-10 * max(1.0, abs(plain["energy"]))
    assert np.max(np.abs(plain["forces"] - field["forces"])) <= 1e-10


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_field_free_h1_entry_refuses_field_model(field_models, use_kokkos):
    """compute_H1 (and so the field-free whole evaluator) forms H1 = M0 U without
    the field coupling and in the wrong basis for the pre_linear_up maps; a
    MACEField model must take compute_H1_product -> field -> compute_H1_linear_up."""
    nc = _native()
    import symmetrix as native

    if use_kokkos:
        if not native._kokkos_is_initialized():
            native._init_kokkos()
        cls = native.MACEKokkos
    else:
        cls = nc.MACE
    plain_path, field_path = field_models
    data = json.loads(Path(field_path).read_text())
    K, LM = data["num_channels"], (data["L_max"] + 1) ** 2
    atoms = _variant_atoms()
    graph = _cluster_graph(atoms, data["r_cut"], data["atomic_numbers"])
    n = len(atoms)
    rng = np.random.default_rng(41)
    for path, refuses in ((field_path, True), (plain_path, False)):
        evaluator = cls(str(path), "")
        try:
            evaluator.M0 = rng.normal(size=n * LM * K)
            if refuses:
                with pytest.raises(ValueError, match="compute_H1_product"):
                    evaluator.compute_H1(n)
            else:
                evaluator.compute_H1(n)
            evaluator.set_streamed_edges("materialized")
            args = (n, graph["node_types"], graph["num_neigh"], graph["j"],
                    graph["neigh_types"], graph["xyz"], graph["r"])
            if refuses:
                with pytest.raises(ValueError, match="compute_H1_product"):
                    evaluator.compute_node_energies_forces(*args)
            else:
                evaluator.compute_node_energies_forces(*args)
                assert np.isfinite(np.sum(evaluator.node_energies))
        finally:
            del evaluator
            gc.collect()


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_field_derivatives_match_finite_differences(field_models, use_kokkos):
    """With a field: forces = -dE/dx, polarization = -dE/dE_field / V,
    polarizability = d(polarization)/dE_field / eps0 and BECs = dF/dE_field."""
    _native()
    from symmetrix.calculator import Symmetrix

    _, field_path = field_models
    atoms = _variant_atoms()
    E0 = np.array([0.05, -0.03, 0.08])
    props = ("energy", "forces", "polarization", "polarizability", "becs")
    base = _evaluate(field_path, "float64", use_kokkos, atoms, electric_field=E0, properties=props)
    h = 1e-4
    volume = atoms.get_volume()
    worst = {}
    for component in range(3):
        dE = np.zeros(3)
        dE[component] = h
        plus = _evaluate(field_path, "float64", use_kokkos, atoms, electric_field=E0 + dE,
                         properties=("energy", "forces", "polarization"))
        minus = _evaluate(field_path, "float64", use_kokkos, atoms, electric_field=E0 - dE,
                          properties=("energy", "forces", "polarization"))
        fd_energy = (plus["energy"] - minus["energy"]) / (2 * h)
        worst["polarization"] = max(worst.get("polarization", 0.0),
                                    abs(-fd_energy / volume - base["polarization"][component]))
        fd_pol = (plus["polarization"] - minus["polarization"]) / (2 * h)
        alpha = base["polarizability"].reshape(3, 3)
        worst["polarizability"] = max(worst.get("polarizability", 0.0), float(np.max(np.abs(
            fd_pol / Symmetrix._macefield_eps0 - alpha[:, component]))) / max(
                1.0, float(np.max(np.abs(alpha)))))
        fd_force = (plus["forces"] - minus["forces"]) / (2 * h)
        becs = base["becs"].reshape(len(atoms), 3, 3)
        worst["becs"] = max(worst.get("becs", 0.0), float(np.max(np.abs(
            fd_force - becs[:, component, :]))))
    eps = 1e-4
    for atom in (0, 4):
        for cart in range(3):
            plus_atoms, minus_atoms = atoms.copy(), atoms.copy()
            plus_atoms.positions[atom, cart] += eps
            minus_atoms.positions[atom, cart] -= eps
            fd = (_evaluate(field_path, "float64", use_kokkos, plus_atoms, electric_field=E0)["energy"]
                  - _evaluate(field_path, "float64", use_kokkos, minus_atoms,
                              electric_field=E0)["energy"]) / (2 * eps)
            worst["forces"] = max(worst.get("forces", 0.0), abs(-fd - base["forces"][atom, cart]))
    print(f"H1_AFFINE_FIELD_FD kokkos={use_kokkos} {worst}")
    assert worst["polarization"] <= 1e-7
    assert worst["polarizability"] <= 1e-5
    assert worst["becs"] <= 1e-6
    assert worst["forces"] <= 1e-6


# --------------------------------------------------------------------------- LAMMPS-style ghost rows

def _cluster_graph(atoms, cutoff, atomic_numbers):
    from ase.neighborlist import neighbor_list

    i, j, d, D = neighbor_list("ijdD", atoms, cutoff)
    type_of = {z: t for t, z in enumerate(atomic_numbers)}
    node_types = np.array([type_of[z] for z in atoms.numbers], dtype=np.int32)
    return {
        "i": i, "j": j.astype(np.int32), "r": np.ascontiguousarray(d, dtype=float),
        "xyz": np.ascontiguousarray(D.reshape(-1), dtype=float),
        "node_types": node_types,
        "num_neigh": np.bincount(i, minlength=len(atoms)).astype(np.int32),
        "neigh_types": node_types[j].copy(),
    }


def _ghost_sequence(evaluator, graph, num_local):
    """Stage sequence of the serial / Kokkos pair styles without MPI message
    passing: H1 for local + ghost nodes, readouts for the local nodes only."""
    g = graph
    num_all = len(g["node_types"])
    args = (g["node_types"], g["num_neigh"], g["neigh_types"])
    evaluator.set_streamed_edges("materialized")
    evaluator.prepare_active_types(g["node_types"])
    evaluator.node_energies = np.zeros(num_local)
    evaluator.node_forces = np.zeros(g["xyz"].size)
    evaluator.compute_Y(g["xyz"])
    evaluator.compute_R0(num_all, *args, g["r"])
    evaluator.compute_A0(num_all, *args)
    evaluator.compute_A0_scaled(num_all, *args, g["r"])
    evaluator.compute_M0(num_all, g["node_types"])
    evaluator.compute_H1(num_all)
    evaluator.add_H1_first_residual(num_all, g["node_types"], True)
    evaluator.compute_R1(num_local, *args, g["r"])
    evaluator.compute_Phi1(num_local, g["num_neigh"], g["j"])
    evaluator.compute_A1(num_local)
    evaluator.compute_A1_scaled(num_local, *args, g["r"])
    evaluator.compute_M1(num_local, g["node_types"])
    evaluator.compute_H2(num_local, g["node_types"])
    evaluator.compute_readouts(num_local, g["node_types"])
    evaluator.reverse_H2(num_local, g["node_types"], False)
    evaluator.reverse_M1(num_local, g["node_types"])
    evaluator.reverse_A1_scaled(num_local, *args, g["xyz"], g["r"])
    evaluator.reverse_A1(num_local)
    evaluator.reverse_Phi1(num_local, g["num_neigh"], g["j"], g["xyz"], g["r"], False, False)
    evaluator.reverse_H1(num_all)
    evaluator.reverse_M0(num_all, g["node_types"])
    evaluator.reverse_A0_scaled(num_all, *args, g["xyz"], g["r"])
    evaluator.reverse_A0(num_all, *args, g["xyz"], g["r"])
    pair = np.asarray(evaluator.node_forces, dtype=float).reshape(-1, 3)[: len(g["i"])]
    forces = np.zeros((num_all, 3))
    for c in range(3):
        forces[:, c] = (np.bincount(g["j"], weights=pair[:, c], minlength=num_all)
                        - np.bincount(g["i"], weights=pair[:, c], minlength=num_all))
    return float(np.sum(evaluator.node_energies)), forces


def _local_energy(evaluator, atoms, cutoff, atomic_numbers, num_local):
    g = _cluster_graph(atoms, cutoff, atomic_numbers)
    evaluator.set_streamed_edges("materialized")
    evaluator.compute_node_energies_forces(
        len(atoms), g["node_types"], g["num_neigh"], g["j"], g["neigh_types"], g["xyz"], g["r"])
    return float(np.sum(np.asarray(evaluator.node_energies)[:num_local]))


@pytest.mark.parametrize("use_kokkos", [False, True])
@pytest.mark.parametrize("variant", ["synthetic_m0_composite", "residual_pre_linear_up",
                                     "residual_old_format"])
def test_ghost_rows_receive_no_direct_adjoint(tmp_path, variant, use_kokkos):
    """The pair-style stage sequence (as fixed: the first-interaction residual is
    added after compute_H1): the H1 reverse covers ghost rows after the local
    ones and only the local rows carry the readout_1 / skip adjoint.  The local
    energy and its exact gradient (finite differences of the sum of local node
    energies) must follow, for a residual-first model with its residual b in
    v4 and in the old format."""
    nc = _native()
    if variant == "synthetic_m0_composite":
        _, data, _ = synthetic_model_pair(1e-3)
        symbols = ("H", "O", "O")
    else:
        from symmetrix.extract_mace_data import _extract_single_head_data

        data = _extract_single_head_data(_mace_model("residual", sigma_min=1e-3, seed=3),
                                         num_spline_points=500)
        assert data["h1_affine_mode"] == "pre_linear_up"
        assert np.max(np.abs(data["H1_first_residual_weights"])) > 1e-3
        if variant == "residual_old_format":
            data = _to_old_basis(data)
        symbols = ("H", "C", "O")
    path = _write(tmp_path, "model.json", data)
    if use_kokkos:
        import symmetrix as native

        if not native._kokkos_is_initialized():
            native._init_kokkos()
        cls = native.MACEKokkos
    else:
        cls = nc.MACE
    atoms = _structure(symbols=symbols, count=10, box=5.0, seed=31)
    atoms.pbc = False
    atoms.cell = None
    cutoff, numbers, num_local = data["r_cut"], data["atomic_numbers"], 6
    evaluator = cls(str(path), "")
    reference = cls(str(path), "")
    try:
        energy, forces = _ghost_sequence(
            evaluator, _cluster_graph(atoms, cutoff, numbers), num_local)
        local = _local_energy(reference, atoms, cutoff, numbers, num_local)
        h = 1e-4
        worst = 0.0
        for atom in (1, 8):  # one local and one ghost atom
            for cart in range(3):
                plus, minus = atoms.copy(), atoms.copy()
                plus.positions[atom, cart] += h
                minus.positions[atom, cart] -= h
                fd = (_local_energy(reference, plus, cutoff, numbers, num_local)
                      - _local_energy(reference, minus, cutoff, numbers, num_local)) / (2 * h)
                worst = max(worst, abs(-fd - forces[atom, cart]))
    finally:
        del evaluator, reference
        gc.collect()
    print(f"H1_AFFINE_GHOST {variant} kokkos={use_kokkos} dE={abs(energy - local):.3e} "
          f"max|F+dE/dx|={worst:.3e}")
    assert abs(energy - local) <= 1e-10
    assert worst <= 1e-6


def _to_old_basis(data):
    """Rebuild the format-2 payload the old exporter wrote for a split
    (pre_linear_up) model: readout and skip multiplied by inv(U0) (test only)."""
    K = data["num_channels"]
    U0 = np.asarray(data["H1_linear_up_weights"][: K * K]).reshape(K, K)
    old = copy.deepcopy(data)
    r0 = np.asarray(old.pop("readout_1_weights_for_H1_pre_linear_up"))
    S0 = np.asarray(old.pop("H2_weights_for_H1_pre_linear_up")).reshape(-1, K, K)
    for key in ("h1_affine_mode", "h1_affine_version", "h1_affine_provenance"):
        old.pop(key, None)
    old["symmetrix_format_version"] = 2
    old["readout_1_weights"] = np.linalg.solve(U0, r0).tolist()
    old["H2_weights_for_H1"] = [np.linalg.solve(U0, s).reshape(-1).tolist() for s in S0]
    return old


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_converted_split_files_match_native_exports(field_models, use_kokkos):
    """Residual-first and MACEField: the old file (inverse; evaluated with the
    old arithmetic, incl. the field response) and its converted v4 file agree
    with the fresh export, with and without a field."""
    _native()
    plain_path, field_path = field_models
    atoms = _variant_atoms()
    for path, field in ((plain_path, None), (field_path, [0.05, -0.03, 0.08])):
        native = json.loads(Path(path).read_text())
        old = _to_old_basis(native)
        converted, _ = _recompose(old)
        assert converted["h1_affine_mode"] == "pre_linear_up"
        old_path = Path(path).with_name("old_" + Path(path).name)
        conv_path = Path(path).with_name("converted_" + Path(path).name)
        old_path.write_text(json.dumps(old))
        conv_path.write_text(json.dumps(converted))
        props = ("energy", "forces") if field is None else (
            "energy", "forces", "polarization", "polarizability", "becs")
        a = _evaluate(path, "float64", use_kokkos, atoms, electric_field=field, properties=props)
        for label, other in (("converted", conv_path), ("old", old_path)):
            b = _evaluate(other, "float64", use_kokkos, atoms, electric_field=field,
                          properties=props)
            worst = {key: float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key]))))
                     for key in props}
            print(f"H1_AFFINE_SPLIT_{label.upper()} {Path(path).stem} kokkos={use_kokkos} {worst}")
            assert worst["energy"] <= 1e-9 * max(1.0, abs(a["energy"]))
            for key, value in worst.items():
                if key != "energy":
                    assert value <= 1e-8 * max(1.0, float(np.max(np.abs(a[key])))), key


# --------------------------------------------------------------------------- review fixes (check 1)

@pytest.mark.parametrize("mode", ["m0_composite", "pre_linear_up"])
def test_reverse_H2_overwrites_M1_adjoint(tmp_path, mode):
    """dE/dM1 = H2_adj W^T is assigned, not accumulated: M1_adj may alias M1 or
    hold a previous adjoint (Kokkos MH0 adjoint reuse)."""
    nc = _native()
    import symmetrix as native

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    _, data, _ = synthetic_model_pair(0.5)
    if mode == "pre_linear_up":
        data["h1_affine_mode"] = mode
        data["H1_product_weights"] = data["H1_weights"]
        data["readout_1_weights_for_H1_pre_linear_up"] = data.pop("readout_1_weights_for_M0")
        data["H2_weights_for_H1_pre_linear_up"] = data.pop("H2_weights_for_M0")
    K, LM = data["num_channels"], (data["L_max"] + 1) ** 2
    rng = np.random.default_rng(2207)
    path = _write(tmp_path, "model.json", data)
    weight = np.asarray(data["H2_weights_for_M1"], dtype=np.float64).reshape(K, K)
    for cls, scalar in ((nc.MACE, np.float64), (nc.MACEFloat, np.float32),
                        (native.MACEKokkos, np.float64), (native.MACEKokkosFloat, np.float32)):
        evaluator = cls(str(path), "")
        try:
            n = 3
            types = np.array([0, 1, 0], dtype=np.int32)
            evaluator.M0 = np.asarray(rng.normal(size=n * LM * K), dtype=scalar)
            evaluator.compute_H1(n)
            evaluator.M1 = np.asarray(rng.normal(size=n * K), dtype=scalar)
            evaluator.M1_adj = np.full(n * K, 17.0, dtype=scalar)
            evaluator.H1_adj = np.zeros(n * LM * K, dtype=scalar)
            adjoint = rng.normal(size=(n, K))
            evaluator.H2_adj = np.asarray(adjoint.reshape(-1), dtype=np.float64
                                          if "Kokkos" in cls.__name__ else scalar)
            expected = adjoint @ weight.T
            evaluator.reverse_H2(n, types, False)
            first = np.asarray(evaluator.M1_adj, dtype=np.float64).reshape(n, K)
            evaluator.reverse_H2(n, types, False)
            second = np.asarray(evaluator.M1_adj, dtype=np.float64).reshape(n, K)
            tolerance = 5e-6 if scalar is np.float32 else 1e-12
            np.testing.assert_allclose(first, expected, rtol=tolerance, atol=tolerance)
            np.testing.assert_array_equal(second, first)
        finally:
            del evaluator
            gc.collect()


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_field_rank_without_owned_readouts(field_models, use_kokkos):
    """Field MPI split stages on a rank that owns no readout rows: reverse_H2(0)
    then the linear_up reverse over the communicated features must not throw and
    must add no direct readout / skip adjoint."""
    nc = _native()
    import symmetrix as native

    if use_kokkos:
        if not native._kokkos_is_initialized():
            native._init_kokkos()
        cls = native.MACEKokkos
    else:
        cls = nc.MACE
    _, field_path = field_models
    data = json.loads(Path(field_path).read_text())
    K, LM = data["num_channels"], (data["L_max"] + 1) ** 2
    rng = np.random.default_rng(31)
    evaluator = cls(str(field_path), "")
    try:
        features = 3
        evaluator.H1 = rng.normal(size=features * LM * K)
        evaluator.compute_field_H1(features, np.array([0.03, -0.02, 0.07]))
        evaluator.compute_H1_linear_up(features)
        evaluator.H1_adj = np.zeros(features * LM * K)
        evaluator.M1 = np.empty(0)
        evaluator.H2_adj = np.empty(0)
        evaluator.reverse_H2(0, np.empty(0, dtype=np.int32), False)
        evaluator.reverse_H1_linear_up(features)
        assert float(np.max(np.abs(np.asarray(evaluator.H1_adj)))) == 0.0
    finally:
        del evaluator
        gc.collect()


def _cli(*args):
    import shutil
    import subprocess
    import sys

    command = shutil.which("symmetrix_recompose_mace_json")
    prefix = [command] if command else [sys.executable, "-m", "symmetrix.cli.recompose_mace_json"]
    return subprocess.run(prefix + [str(arg) for arg in args], capture_output=True, text=True)


@pytest.mark.parametrize("case", ["zero_P", "exact_cancellation"])
def test_converter_accepts_unbounded_diagnostics(tmp_path, case, capfd):
    """A singular fused map (zero P: cond_C unbounded) or an exactly cancelling
    readout / skip (unbounded cancellation estimate) is a valid old model; the
    installed converter writes standard JSON (null for unbounded diagnostics)
    and the v4 output loads and evaluates."""
    nc = _native()
    old, _, _ = synthetic_model_pair(0.5)
    K = old["num_channels"]
    C = np.asarray(old["H1_weights"]).reshape(-1, K, K)
    if case == "zero_P":
        C[:] = 0.0
    else:
        C[0] = 1.0
        readout = np.zeros(K)
        readout[:2] = [1.0, -1.0]
        skip = np.zeros((K, K))
        skip[:2, 0] = [1.0, -1.0]
        old["readout_1_weights"] = readout.tolist()
        old["H2_weights_for_H1"] = [skip.reshape(-1).tolist() for _ in old["atomic_numbers"]]
    old["H1_weights"] = C.reshape(-1).tolist()
    src = _write(tmp_path, "old.json", old)
    dst = tmp_path / "v4.json"
    result = _cli("--input", src, "--output", dst, "--receipt", tmp_path / "receipt.json")
    assert result.returncode == 0, result.stderr
    new = json.loads(dst.read_text())  # strict JSON (no Infinity / NaN tokens)
    for text in (dst.read_text(), (tmp_path / "receipt.json").read_text()):
        assert "Infinity" not in text and "NaN" not in text
    diag = new["h1_affine_provenance"]["conditioning"]
    if case == "zero_P":
        assert diag["cond_C"] is None
    else:
        assert diag["legacy_cancellation"]["readout"] is None
    validate_payload(new)
    summary = _cli("--validate-only", src)
    assert summary.returncode == 0 and "Infinity" not in summary.stdout
    result = _evaluate(dst, "float64", False, _structure())
    assert all(np.all(np.isfinite(value)) for value in result.values())
    nc.MACE(str(src), "")
    if case == "exact_cancellation":
        assert "readout unbounded" in capfd.readouterr().err
