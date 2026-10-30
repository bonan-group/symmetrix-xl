"""Family naming contract for the Symmetrix source trees.

Family-owned translation units carry their family prefix so ownership is
visible from the filename alone, and framework files stay inside the recorded
neutral allowlist.  The allowlists below are the explicit contract: adding a
file to either tree requires either a family prefix or a deliberate entry here.
The prefix itself is not listed here at all: it is derived from the declared
family entries, so declaring a family admits its own files without editing this
module.

The retired-basename check is the companion invariant: a rename sweep must not
leave the previous spelling behind in any tree, so a re-introduced legacy
filename fails immediately.
"""

import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest

from symmetrix import family_registry

ROOT = Path(__file__).resolve().parents[2]

NATIVE_ROOT = ROOT / "libsymmetrix" / "source"
PYTHON_ROOT = ROOT / "symmetrix" / "source" / "symmetrix"
BINDINGS_ROOT = ROOT / "symmetrix" / "source" / "cpp"


def _family_token(key: str) -> str:
    """The filename token a declared family key spells.

    The token is the key's leading segment, lowercased: the family root a
    declaration names, which its own files carry before the ``_`` that opens
    the rest of the basename.
    """

    return key.split("_", 1)[0].lower()


def _family_prefix(entries) -> re.Pattern[str]:
    """Compile the prefix admitting every declared family's own files."""

    tokens = sorted({_family_token(entry.key) for entry in entries})
    return re.compile(r"^(?:" + "|".join(f"{re.escape(t)}[._]" for t in tokens) + r")")


# Family files spell the family token followed by ``_`` or ``.``.  The bare
# family-root basenames (``mace.cpp``, ``oam.hpp``) belonged to the serial
# evaluators, which retired, and are recorded in ``NATIVE_RETIRED`` so a
# reappearance fails there.
FAMILY_PREFIX = _family_prefix(family_registry.declared_entries())

# Framework/neutral files permitted to stay unprefixed in ``libsymmetrix/source``.
NATIVE_UNPREFIXED = frozenset(
    {
        "affine_mlp.cpp",
        "affine_mlp.hpp",
        "affine_mlp_kokkos.cpp",
        "affine_mlp_kokkos.hpp",
        "cblas.hpp",
        "compact_radial.cpp",
        "compact_radial.hpp",
        "fourier_d3_coordination.cpp",
        "fourier_d3_coordination.hpp",
        "fourier_d3_evaluator.cpp",
        "fourier_d3_evaluator.hpp",
        "fourier_d3_reciprocal.cpp",
        "fourier_d3_reciprocal.hpp",
        "fourier_d3_reverse.cpp",
        "fourier_d3_reverse.hpp",
        "fourier_d3_storage.cpp",
        "fourier_d3_storage.hpp",
        "fourier_d3_timing.hpp",
        "cubic_spline.cpp",
        "cubic_spline.hpp",
        "cubic_spline_kokkos.cpp",
        "cubic_spline_kokkos.hpp",
        "cubic_spline_set.cpp",
        "cubic_spline_set.hpp",
        "cubic_spline_set_kokkos.cpp",
        "cubic_spline_set_kokkos.hpp",
        "cuda_r1_identity.hpp",
        "device_backend.cpp",
        "device_backend.hpp",
        "d2_kokkos.cpp",
        "d2_kokkos.hpp",
        "d2_kokkos_impl.tpp",
        "d3_kokkos.cpp",
        "d3_kokkos.hpp",
        "d3_kokkos_impl.tpp",
        "dispersion_common.hpp",
        "dispersion_graph_kokkos.cpp",
        "dispersion_graph_kokkos.hpp",
        "dispersion_parameter_provenance.hpp",
        "dispersion_parameters.cpp",
        "dispersion_parameters.hpp",
        "e3nn.cpp",
        "e3nn.hpp",
        "e3nn_kokkos.cpp",
        "e3nn_kokkos.hpp",
        "e3nn_product.cpp",
        "e3nn_product.hpp",
        "e3nn_product_kokkos.cpp",
        "e3nn_product_kokkos.hpp",
        "e3nn_product_monomial_key.hpp",
        "execution_mode.hpp",
        "execution_plan.hpp",
        "factorized_blas.hpp",
        "family_interface.hpp",
        "fourier_d3_coordination.cpp",
        "fourier_d3_coordination.hpp",
        "fourier_d3_evaluator.cpp",
        "fourier_d3_evaluator.hpp",
        "fourier_d3_reciprocal.cpp",
        "fourier_d3_reciprocal.hpp",
        "fourier_d3_reverse.cpp",
        "fourier_d3_reverse.hpp",
        "fourier_d3_storage.cpp",
        "fourier_d3_storage.hpp",
        "fourier_d3_timing.hpp",
        "host_batched_readout.hpp",
        "host_dense_kernels.hpp",
        "host_worker_blas.hpp",
        "implementation_registry.hpp",
        "jit_device_plugin_common.hpp",
        "jit_generation_version.hpp",
        "jit_hiprtc.cpp",
        "jit_hiprtc.hpp",
        "jit_manifest_identity.hpp",
        "jit_nvrtc.cpp",
        "jit_nvrtc.hpp",
        "kernel_launch_profile.hpp",
        "long_range_kernels.cpp",
        "long_range_kernels.hpp",
        "long_range_operator_kinds.hpp",
        "long_range_solver_policy.hpp",
        "les_particle_mesh.cpp",
        "les_particle_mesh.hpp",
        "les_monopole.cpp",
        "les_monopole.hpp",
        "metal_view_registry.hpp",
        "model_metadata.cpp",
        "model_metadata.hpp",
        "multilayer_perceptron.cpp",
        "multilayer_perceptron.hpp",
        "multilayer_perceptron_kokkos.cpp",
        "multilayer_perceptron_kokkos.hpp",
        "multivariate_polynomial.cpp",
        "multivariate_polynomial.hpp",
        "multivariate_polynomial_kokkos.cpp",
        "multivariate_polynomial_kokkos.hpp",
        "neighbor_graph_kokkos.cpp",
        "neighbor_graph_kokkos.hpp",
        "particle_mesh_geometry.hpp",
        "particle_mesh_stencil.hpp",
        "particle_mesh_storage.cpp",
        "particle_mesh_storage.hpp",
        "particle_mesh_strain.hpp",
        "particle_mesh_transport.cpp",
        "particle_mesh_transport.hpp",
        "plan_candidate_space.hpp",
        "radial_function_set_kokkos.cpp",
        "radial_function_set_kokkos.hpp",
        "sha256.hpp",
        "spherical_harmonic.cpp",
        "spherical_harmonic.hpp",
        "spherical_harmonic_device.hpp",
        "tools.cpp",
        "tools.hpp",
        "tools_kokkos.cpp",
        "tools_kokkos.hpp",
        "zbl.cpp",
        "zbl.hpp",
        "zbl_kokkos.cpp",
        "zbl_kokkos.hpp",
    }
)

# Framework/neutral files permitted to stay unprefixed in the Python package.
# ``extract_mace_*``, ``prediction_heads.py``, and the ``nequip_*`` modules are
# extraction/validation front-ends whose registry fate belongs to Stage 8, not
# to the Task 6 loader/module rename sweep.
PYTHON_UNPREFIXED = frozenset(
    {
        "__init__.py",
        "backend_loader.py",
        "calculator.py",
        "corrections.py",
        "cuda_codegen_common.py",
        "ensemble.py",
        "extract_mace_data.py",
        "extract_mace_nonlinear.py",
        "extract_mace_polar.py",
        "extract_nequip_data.py",
        "factorized_r1_operator_fixture.py",
        "family_registry.py",
        "fourier_d3_capability.py",
        "fourier_d3_factors.py",
        "foundation_m0_topologies.json",
        "foundation_oam_architectures.json",
        "foundation_operator_manifest.json",
        "foundation_operator_manifest.py",
        "generated_operator_codegen.py",
        "generated_operator_core.py",
        "generated_operator_version.py",
        "installed_operator_artifact.py",
        "jit.py",
        "jit_codegen_common.py",
        "kernel_launch_calibrate.py",
        "kernel_launch_tuning.py",
        "metal_codegen.py",
        "neighbor_graph.py",
        "nequip_calculator.py",
        "nequip_oam_graph.py",
        "nequip_probe.py",
        "nequip_schema.py",
        "polar_long_range_kinds.py",
        "prediction_heads.py",
        "r0_reporting.py",
        "runtime_diagnostics.py",
        "symmetrix.py",
    }
)

# Basenames retired by the family-alignment, neutralization, and direct-execution
# sweeps, per tree.  The serial-evaluator translation units (``mace``,
# ``mace_nonlinear``, ``oam``) retired with their evaluators, so their basenames
# must not reappear beside the retained Kokkos ones.
NATIVE_RETIRED = frozenset(
    {
        "mace",
        "mace_nonlinear",
        "oam",
        "jit_host_plugin",
        "jit_cuda_plugin",
        "jit_hip_plugin",
        "jit_m0_host_plugin",
        "jit_mh1_host_plugin",
        "jit_mh1_cuda_plugin",
        "mh1_stage",
        "jit_operator_module",
        "jit_oam_host_plugin",
        "mace_streamed_edges",
        "standard_m0",
        "standard_m1",
        "standard_r0",
        "prediction_heads",
        "field_coupling",
        "nequip_oam",
        "nequip_oam_kokkos",
        "nequip_oam_kokkos_forward",
        "nequip_oam_kokkos_reverse",
        "nequip_oam_radial",
        "nequip_schema",
        "nequip_validation",
    }
)

PYTHON_RETIRED = frozenset(
    {
        "jit_codegen",
        "mh1_jit_codegen",
        "jit_operator_codegen",
        "jit_host_artifact",
        "jit_device_artifact",
        "jit_operator_artifact",
        "execution_contract",
        "execution_mh1_contract",
        "jit_host_plugin_abi",
        "jit_cuda_plugin_abi",
        "jit_hip_plugin_abi",
        "jit_mh1_host_plugin_abi",
        "jit_mh1_cuda_plugin_abi",
        "jit_m0_host_plugin_abi",
        "jit_operator_module_abi",
        "jit_oam_host_plugin_abi",
    }
)

TREES = (
    pytest.param(NATIVE_ROOT, NATIVE_UNPREFIXED, NATIVE_RETIRED, id="native"),
    pytest.param(PYTHON_ROOT, PYTHON_UNPREFIXED, PYTHON_RETIRED, id="python"),
)


def _filenames(root: Path) -> set[str]:
    return {path.name for path in root.iterdir() if path.is_file()}


def _naming_violations(prefix, names, unprefixed):
    """The two ways a name set can break the contract under ``prefix``.

    A name that is neither family-prefixed nor allowlisted is unclassified, and
    an allowlisted name that gained a family prefix is mislabelled.
    """

    unclassified = sorted(
        name for name in names if not prefix.match(name) and name not in unprefixed
    )
    mislabelled = sorted(
        name for name in names if prefix.match(name) and name in unprefixed
    )
    return unclassified, mislabelled


@pytest.mark.parametrize(("root", "unprefixed", "retired"), TREES)
def test_every_file_is_family_prefixed_or_allowlisted(root, unprefixed, retired):
    unclassified, mislabelled = _naming_violations(
        FAMILY_PREFIX, _filenames(root), unprefixed
    )

    assert not unclassified, (
        f"unprefixed files without a naming decision in {root}: {unclassified}. "
        "Give family-owned files their family prefix or record them in the allowlist."
    )

    assert (
        not mislabelled
    ), f"allowlisted framework files gained a family prefix in {root}: {mislabelled}"


def test_every_declared_family_token_admits_its_own_files():
    """The prefix is the declared families' tokens, read from the registry."""

    entries = family_registry.declared_entries()

    assert FAMILY_PREFIX.pattern == _family_prefix(entries).pattern

    for entry in entries:
        token = _family_token(entry.key)
        assert FAMILY_PREFIX.match(f"{token}_module.py"), entry.key
        assert FAMILY_PREFIX.match(f"{token}.hpp"), entry.key


def test_a_newly_declared_family_admits_its_files_without_editing_this_module():
    """A synthetic declaration extends the prefix; no list here is consulted."""

    declared = family_registry.declared_entries()
    synthetic = replace(declared[0], key="SYNTH_Fam")
    prefix = _family_prefix((*declared, synthetic))

    assert prefix.match("synth_declared_module.py")
    assert prefix.match("synth.module.cpp")
    # The token is the key's leading segment, so the synthetic family's own
    # files pass while a name no declaration claims still does not.
    unclassified, _ = _naming_violations(
        prefix, {"synth_declared_module.py", "phantom_module.py"}, frozenset()
    )
    assert unclassified == ["phantom_module.py"]


def test_unregistered_and_misprefixed_names_stay_unclassified():
    """A name no declared family admits is still a naming violation."""

    unclassified, _ = _naming_violations(
        FAMILY_PREFIX,
        {
            "mace_jit_codegen.py",
            "oam.module.cpp",
            "recorded_neutral.py",
            "phantom_family.py",
            "macfoo.py",
            "mace2_family.py",
            "mace-operator.hpp",
        },
        frozenset({"recorded_neutral.py"}),
    )

    assert unclassified == [
        "mace-operator.hpp",
        "mace2_family.py",
        "macfoo.py",
        "phantom_family.py",
    ]

    _, mislabelled = _naming_violations(
        FAMILY_PREFIX,
        {"mace_jit_codegen.py"},
        frozenset({"mace_jit_codegen.py"}),
    )
    assert mislabelled == ["mace_jit_codegen.py"]


@pytest.mark.parametrize(("root", "unprefixed", "retired"), TREES)
def test_retired_family_basenames_are_absent(root, unprefixed, retired):
    present = sorted(
        path.name for path in root.iterdir() if path.is_file() and path.stem in retired
    )
    assert not present, f"retired family basenames reappeared in {root}: {present}"


def test_binding_sources_do_not_reuse_retired_family_basenames():
    present = sorted(
        path.name
        for path in BINDINGS_ROOT.iterdir()
        if path.is_file() and path.stem in NATIVE_RETIRED
    )
    assert (
        not present
    ), f"retired family basenames reappeared in {BINDINGS_ROOT}: {present}"


# Shared-layer plumbing: the neutral allowlist entries that more than one family
# or a family-neutral translation unit consumes.  A family token here leaks one
# family into every consumer, so these files carry none.  Stage 3 Task 8 removed
# the term-plan include, the MH-1 product layout names, the spherical-harmonic
# normalization name, the MACE device-packet/plugin traits, the error strings,
# and the comments that named a family.  Stage 6 Task 11 renamed the
# streamed-edge mode header into the neutral execution mode and retyped the
# execution plan report onto it, which retired the ``execution_plan.hpp``
# deferral.
SHARED_NEUTRAL_HEADERS = (
    "e3nn.hpp",
    "e3nn_kokkos.hpp",
    "e3nn_product.hpp",
    "e3nn_product_kokkos.hpp",
    "e3nn_product_monomial_key.hpp",
    "spherical_harmonic.hpp",
    "spherical_harmonic_device.hpp",
    "compact_radial.hpp",
    "cubic_spline.hpp",
    "cubic_spline_kokkos.hpp",
    "cubic_spline_set.hpp",
    "cubic_spline_set_kokkos.hpp",
    "affine_mlp.hpp",
    "affine_mlp_kokkos.hpp",
    "multilayer_perceptron.hpp",
    "multilayer_perceptron_kokkos.hpp",
    "multivariate_polynomial.hpp",
    "multivariate_polynomial_kokkos.hpp",
    "radial_function_set_kokkos.hpp",
    "tools.hpp",
    "tools_kokkos.hpp",
    "sha256.hpp",
    "device_backend.hpp",
    "neighbor_graph_kokkos.hpp",
    "jit_device_plugin_common.hpp",
    "jit_generation_version.hpp",
    "jit_manifest_identity.hpp",
    "jit_nvrtc.hpp",
    "jit_hiprtc.hpp",
    "model_metadata.hpp",
    "zbl.hpp",
    "zbl_kokkos.hpp",
    "cuda_r1_identity.hpp",
    "execution_mode.hpp",
    "execution_plan.hpp",
    "plan_candidate_space.hpp",
    "family_interface.hpp",
    "implementation_registry.hpp",
)

# The shared-layer owners whose sources carry the same contract: the neutral
# translation units that implement the headers above.
SHARED_NEUTRAL_SOURCES = (
    "e3nn.cpp",
    "e3nn_kokkos.cpp",
    "e3nn_product.cpp",
    "e3nn_product_kokkos.cpp",
    "spherical_harmonic.cpp",
    "compact_radial.cpp",
    "device_backend.cpp",
)

# Neutral-named files that still carry a family token.  Each entry records why
# the token stays or which owner resolves it; the companion test fails once the
# file becomes clean, so this list cannot go stale.
SHARED_FAMILY_DEFERRED = {
    "long_range_kernels.hpp": (
        "the polar long-range kernels for the tiered registry; the "
        "design-note filename it cites is provenance for the physics, while "
        "the kernels themselves are family-neutral"
    ),
    "long_range_operator_kinds.hpp": (
        "the polar long-range kind vocabulary for the tiered registry; the "
        "design-note and checkpoint-facts names it cites are provenance for "
        "the physics, while the declarations themselves are family-neutral"
    ),
    "model_metadata.cpp": (
        "the serialized model_type values are wire format and stay; the "
        "file's own predicates and declarations are family-neutral"
    ),
}

# ``mace``/``mh1``/``nequip``/``oam`` as a family token, not as a substring of
# an unrelated word.
FAMILY_TOKEN = re.compile(r"mace|mh1|nequip|oam", re.IGNORECASE)


# Neutral Python modules that carry the same contract as the shared native
# headers: every family's data flows through the Python frontend registry, so a
# family token here leaks one family into every consumer.  The calculators and
# the extraction front-ends are the frontends that still name their families by
# wire string, so they are not scanned here.
NEUTRAL_PYTHON_MODULES = ("family_registry.py",)


def _family_token_lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if FAMILY_TOKEN.search(line)]


def test_native_evaluator_calls_carry_model_type_and_avoid_hardcoded_mace_classes():
    helper_calls = []
    hardcoded_constructions = []
    for path in PYTHON_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "native_evaluator":
                helper_calls.append((path.name, node.lineno))
                assert any(keyword.arg == "model_type" for keyword in node.keywords), (
                    path.name,
                    node.lineno,
                )
            if name in {"MACEKokkos", "MACEKokkosFloat"}:
                hardcoded_constructions.append((path.name, node.lineno, name))

    assert sorted(path for path, _ in helper_calls) == [
        "mace_jit_device_artifact.py",
        "mace_jit_host_artifact.py",
    ]
    assert hardcoded_constructions == []


@pytest.mark.parametrize("name", SHARED_NEUTRAL_HEADERS + SHARED_NEUTRAL_SOURCES)
def test_shared_layer_carries_no_family_token(name):
    hits = _family_token_lines(NATIVE_ROOT / name)

    assert not hits, (
        f"shared-layer file {name} names a family: {hits}. Keep shared "
        "plumbing family-neutral and put the family-owned part behind an "
        "opaque handle in the owning family layer."
    )


def test_shared_layer_scan_covers_every_allowlisted_family_token_file():
    """Every neutral-named native file with a family token is accounted for."""
    deferred = set(SHARED_FAMILY_DEFERRED)
    scanned = set(SHARED_NEUTRAL_HEADERS) | set(SHARED_NEUTRAL_SOURCES)

    assert not scanned & deferred, (
        "a shared-layer file cannot be both scanned and deferred: "
        f"{sorted(scanned & deferred)}"
    )

    unaccounted = sorted(
        path.name
        for path in NATIVE_ROOT.iterdir()
        if path.is_file()
        and path.suffix in {".hpp", ".cpp", ".tpp"}
        and path.name in NATIVE_UNPREFIXED
        and path.name not in scanned
        and path.name not in deferred
        and _family_token_lines(path)
    )
    assert not unaccounted, (
        "neutral-named files carry a family token without a decision: "
        f"{unaccounted}. Scan them or record the owner in "
        "SHARED_FAMILY_DEFERRED."
    )


@pytest.mark.parametrize("name", sorted(SHARED_FAMILY_DEFERRED))
def test_deferred_shared_file_still_carries_its_family_token(name):
    hits = _family_token_lines(NATIVE_ROOT / name)

    assert hits, (
        f"{name} is now family-neutral, so its deferral is stale: remove it "
        "from SHARED_FAMILY_DEFERRED and scan it instead."
    )


@pytest.mark.parametrize("name", NEUTRAL_PYTHON_MODULES)
def test_neutral_python_module_carries_no_family_token(name):
    hits = _family_token_lines(PYTHON_ROOT / name)

    assert not hits, (
        f"neutral Python module {name} names a family: {hits}. Keep the "
        "registry mechanism family-neutral and declare family data beside the "
        "family's own modules."
    )


def test_polar_contract_and_extraction_apis_are_family_owned():
    """The ordinary nonlinear modules must not discover or define Polar."""

    import importlib
    import importlib.util

    from symmetrix import extract_mace_nonlinear
    from symmetrix import extract_mace_polar
    from symmetrix import mace_nonlinear_execution_contract

    module_name = "symmetrix.mace_polar_execution_contract"
    assert importlib.util.find_spec(module_name) is not None
    mace_polar_execution_contract = importlib.import_module(module_name)

    contract_names = {
        "POLAR_CONTRACT_KEY",
        "POLAR_CONTRACT_TAG",
        "make_execution_polar_backbone_contract",
        "make_execution_polar_mh1_v4_contract",
        "normalize_execution_polar_backbone_contract",
        "polar_execution_schedule",
        "validate_execution_polar_backbone_contract_for_model",
        "validate_execution_polar_mh1_v4_contract_for_model",
    }
    for contract_name in contract_names:
        assert not hasattr(mace_nonlinear_execution_contract, contract_name)
        assert hasattr(mace_polar_execution_contract, contract_name)

    assert not hasattr(extract_mace_nonlinear, "_extract_polar_backbone_data")
    assert hasattr(extract_mace_polar, "_extract_polar_backbone_data")


def test_shared_operator_kind_vocabulary_contains_no_polar_tokens():
    """Family-owned program kinds use the neutral custom-kind extension."""

    shared_interface = (NATIVE_ROOT / "family_interface.hpp").read_text()
    assert "polar_" not in shared_interface


def test_polar_binding_helpers_are_defined_by_the_family_binding():
    """The shared nonlinear binding only invokes the Polar registration hook."""

    generic_binding = (BINDINGS_ROOT / "mace_nonlinear_kokkos.cpp").read_text()
    polar_binding = (BINDINGS_ROOT / "mace_polar_binding.hpp").read_text()
    for helper in ("bind_polar_graph_inputs", "bind_polar_long_range_contract"):
        assert f"void {helper}" not in generic_binding
        assert f"void {helper}" in polar_binding
    assert "bind_mace_polar_kokkos_evaluators" in polar_binding


def test_neutral_backbone_owns_no_polar_family_state_or_execution():
    """The reusable backbone must compile without any Polar family owner."""

    header = (NATIVE_ROOT / "mace_nonlinear_kokkos.hpp").read_text()
    implementation = (NATIVE_ROOT / "mace_nonlinear_kokkos_impl.tpp").read_text()
    neutral_source = header + implementation
    forbidden = (
        'include "mace_polar_',
        "MacePolarFamilyState",
        "polar_state",
        "polar_long_range_contract",
        "polar_deltas",
        "polar_graph_inputs",
        "pack_polar_public_outputs",
        "collect_polar_public_outputs",
        "MaceBackboneSchema::polar",
    )
    leaked = [token for token in forbidden if token in neutral_source]
    assert not leaked, f"neutral backbone still owns Polar behavior: {leaked}"

    family_source = (NATIVE_ROOT / "mace_polar_kokkos.hpp").read_text() + (
        NATIVE_ROOT / "mace_polar_family_state.hpp"
    ).read_text()
    required = (
        "MacePolarFamilyState",
        "polar_long_range_contract",
        "polar_deltas",
        "set_polar_graph_inputs",
        "pack_polar_public_outputs",
    )
    missing = [token for token in required if token not in family_source]
    assert not missing, f"Polar evaluator does not own its complete state: {missing}"


def test_neutral_long_range_contract_excludes_polar_program_fields():
    """Electrostatic primitives must not encode the Polar program schedule."""

    neutral = (NATIVE_ROOT / "long_range_operator_kinds.hpp").read_text()
    family = (NATIVE_ROOT / "mace_polar_long_range_contract.hpp").read_text()
    assert (
        "struct PolarLongRangeContract" in family
    ), "the Polar family has no owned aggregate for its program fields"
    contract_start = neutral.index("struct LongRangeContract")
    contract_end = neutral.index("};", contract_start)
    neutral_contract = neutral[contract_start:contract_end]
    family_start = family.index("struct PolarLongRangeContract")
    family_end = family.index("};", family_start)
    family_contract = family[family_start:family_end]

    family_fields = (
        "recursion_step_count",
        "spin_channel_count",
        "include_electrostatic_self_interaction",
        "field_si",
    )
    assert not [field for field in family_fields if field in neutral_contract]
    assert not [field for field in family_fields if field not in family_contract]
