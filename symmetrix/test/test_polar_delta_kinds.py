"""The PolarMACE backbone deltas as declared program data, torch-free.

The delta group is the family's node-local execution layer declared in the same
style as the long-range operator group: one kind per region, buffer contracts
from contract data, scalar parameters, one portable registry descriptor each,
and a producer/consumer coherence rule in the program assembly.  This module
reads that declaration from the native slice (``polar_long_range_program``
compiled with the delta contract data) and pins it against the Python mirror,
so the two sides cannot drift and the chain state's buffers cannot lose their
producers again.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import symmetrix

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

from symmetrix.polar_long_range_kinds import (
    POLAR_DELTA_KINDS,
    POLAR_DELTA_NODE_BUFFERS,
    POLAR_LONG_RANGE_KINDS,
    POLAR_PROGRAM_INPUT_BUFFERS,
    PolarLongRangeContract,
    require_delta_kind,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "polar"
FACTS_PATH = DATA_DIR / "checkpoint_facts.json"

#: The backbone's own data the delta kinds parameterize their contracts with
#: (the shipped M checkpoint's parse).
NODE_FEATURE_DIMENSION = 2048
NUM_ELEMENTS = 83
INTERACTION_COUNT = 2


@pytest.fixture(scope="module")
def contract() -> PolarLongRangeContract:
    facts = json.loads(FACTS_PATH.read_text())
    return PolarLongRangeContract.from_checkpoint_facts(facts["models"]["M"])


def program_slice(contract: PolarLongRangeContract, **overrides: object) -> dict:
    arguments = dict(
        kspace_cutoff=contract.kspace_cutoff,
        density_max_l=contract.density_max_l,
        density_smearing_width=contract.density_smearing_width,
        feature_max_l=contract.feature_max_l,
        feature_smearing_widths=list(contract.feature_smearing_widths),
        charges_dimension=contract.charges_dimension,
        field_dimension=contract.field_dimension,
        potential_dimension=contract.potential_dimension,
        field_norm_factor=contract.field_norm_factor,
        recursion_step_count=contract.recursion_step_count,
        table_dtype=contract.table_dtype,
        node_feature_dimension=NODE_FEATURE_DIMENSION,
        num_elements=NUM_ELEMENTS,
        interaction_count=INTERACTION_COUNT,
    )
    arguments.update(overrides)
    return symmetrix.polar_long_range_program(**arguments)


def test_the_delta_vocabulary_mirrors_the_native_side(contract):
    """The delta kind vocabulary is the same list on both sides of the ABI."""

    program = program_slice(contract)
    assert [node["kind"] for node in program["delta_nodes"]] == list(POLAR_DELTA_KINDS)
    assert len(POLAR_DELTA_KINDS) == len(set(POLAR_DELTA_KINDS))
    for name in POLAR_DELTA_KINDS:
        assert require_delta_kind(name) == name
    with pytest.raises(ValueError, match="polar_spin_source_heads"):
        require_delta_kind("polar-spin-source-heads")


def test_every_delta_node_declares_the_mirrored_buffers(contract):
    """Each kind's buffers are the mirror's, in order.

    The buffer names are the program's linking vocabulary: they are how the
    chain state's producers and the long-range kinds' consumers line up, so
    the pin is on names and order rather than on presence alone.
    """

    program = program_slice(contract)
    delta_nodes = {node["kind"]: node for node in program["delta_nodes"]}
    assert set(delta_nodes) == set(POLAR_DELTA_KINDS)
    for kind, expected in POLAR_DELTA_NODE_BUFFERS.items():
        node = delta_nodes[kind]
        assert (
            tuple(buffer["name"] for buffer in node["inputs"]) == (expected["inputs"])
        ), kind
        assert (
            tuple(buffer["name"] for buffer in node["outputs"]) == (expected["outputs"])
        ), kind
        for buffer in node["inputs"] + node["outputs"]:
            assert buffer["dtype"] == "float64"
            assert buffer["layout"] == "node_major"
            assert buffer["shape"], kind
            assert buffer["shape"][0] == 0, (kind, buffer["name"])


def test_the_chain_state_buffers_gain_producers(contract):
    """The eight buffers the long-range kinds consume now have producers.

    Before the delta group was declared, the compiled program consumed the
    spin-resolved multipoles, the equilibrated densities, the Fukui sources,
    and the per-channel l=0 charges with no producer at all: the nodes could
    not run.  The rule here is the coherence invariant itself, read off the
    program's own node set rather than a parallel table.
    """

    program = program_slice(contract)
    produced: dict[str, list[str]] = {}
    for node in program["nodes"] + program["delta_nodes"]:
        for buffer in node["outputs"]:
            produced.setdefault(buffer["name"], []).append(node["kind"])
    homeless = (
        "multipoles_alpha",
        "multipoles_beta",
        "fukui_sources",
        "charge_l0_alpha",
        "charge_l0_beta",
        "charge_density",
        "charge_density_alpha",
        "charge_density_beta",
    )
    for name in homeless:
        assert name in produced, name
    # The sources have the report's two producers: the pre-loop source map and
    # every unrolled update's readout.
    assert set(produced["fukui_sources"]) == {
        "polar_source_map",
        "polar_field_update",
    }
    # Every long-range node input is either produced in the program or is a
    # declared program input.
    for node in program["nodes"]:
        for buffer in node["inputs"]:
            assert (
                buffer["name"] in produced
                or buffer["name"] in POLAR_PROGRAM_INPUT_BUFFERS
            ), (node["kind"], buffer["name"])
    # And the delta group's own inputs are too.
    for node in program["delta_nodes"]:
        for buffer in node["inputs"]:
            assert (
                buffer["name"] in produced
                or buffer["name"] in POLAR_PROGRAM_INPUT_BUFFERS
            ), (node["kind"], buffer["name"])


def test_the_delta_group_binds_at_the_portable_tier(contract):
    """Every delta node carries a qualified portable implementation."""

    program = program_slice(contract)
    report = program["binding_report"]
    assert report["admitted"]
    assert report["node_count"] == len(program["nodes"]) + len(program["delta_nodes"])
    assert report["portable_node_count"] == report["node_count"]
    delta_ids = {node["id"] for node in program["delta_nodes"]}
    for binding in report["bindings"]:
        if binding["operator_id"] not in delta_ids:
            continue
        assert binding["tier"] == "portable"
        assert binding["portable_covered"]
        assert not binding["artifact_required"]
        assert binding["precision"] == {
            "compute": "float64",
            "accumulator": "float64",
        }


def test_the_program_identity_folds_the_delta_contract(contract):
    """The chain's widths and the self-interaction flag are program identity."""

    baseline = program_slice(contract)
    deeper = program_slice(contract, node_feature_dimension=1024)
    wider = program_slice(contract, interaction_count=3)
    assert deeper["fingerprint"] != baseline["fingerprint"]
    assert wider["fingerprint"] != baseline["fingerprint"]
    assert program_slice(contract)["fingerprint"] == baseline["fingerprint"]


def test_an_incoherent_delta_node_set_is_refused(contract):
    """The coherence and shape rules reject a contradicted contract.

    A charge width the kinds cannot serve is refused by name rather than
    accepted, and the vocabulary refusal is the group's no-spelling idiom: an
    unknown delta token never falls back to another kind.
    """

    with pytest.raises(ValueError, match="charges_dimension"):
        program_slice(contract, charges_dimension=contract.charges_dimension + 2)
    with pytest.raises(ValueError, match="charges_dimension"):
        program_slice(contract, charges_dimension=6)


def test_the_long_range_slice_is_unchanged_by_the_delta_group(contract):
    """The delta group is additive: the long-range half keeps its own node set.

    A consumer that reads the long-range slice (the conformance battery) sees
    exactly the nodes it always did; the delta nodes are reported beside them.
    """

    program = program_slice(contract)
    assert [node["kind"] for node in program["nodes"]] == list(POLAR_LONG_RANGE_KINDS)
    without = symmetrix.polar_long_range_program(
        kspace_cutoff=contract.kspace_cutoff,
        density_max_l=contract.density_max_l,
        density_smearing_width=contract.density_smearing_width,
        feature_max_l=contract.feature_max_l,
        feature_smearing_widths=list(contract.feature_smearing_widths),
        charges_dimension=contract.charges_dimension,
        field_dimension=contract.field_dimension,
        potential_dimension=contract.potential_dimension,
        field_norm_factor=contract.field_norm_factor,
        recursion_step_count=contract.recursion_step_count,
        table_dtype=contract.table_dtype,
    )
    assert without["delta_nodes"] == []
    assert without["nodes"] == program["nodes"]
    assert without["fingerprint"] != program["fingerprint"]


# ---------------------------------------------------------------------------
# The compiled coherence probe
# ---------------------------------------------------------------------------

#: A C++20 probe over the delta group's declarative layer.  It owns the rules
#: the Python mirror cannot own: that the node set validates, that the assembly
#: folds both groups, and that the producer/consumer pass refuses a long-range
#: node whose chain-state input no delta node produces -- the rejection the
#: data-side tests can only assert positively.
DELTA_KIND_PROBE = r"""
#include "mace_polar_delta_kinds.hpp"
#include "mace_polar_program.hpp"

#include <cstdio>
#include <string>
#include <vector>

using symmetrix::execution::Program;
using symmetrix::execution::operator_kind_name;
using LongRangeContract =
    symmetrix::execution::polar::PolarLongRangeContract;
using symmetrix::execution::long_range::find_long_range_kind;
using symmetrix::execution::polar::DeltaContract;
using symmetrix::execution::polar::compile_polar_program;
using symmetrix::execution::polar::delta_kind_specs;
using symmetrix::execution::polar::delta_operator_nodes;
using symmetrix::execution::polar::find_delta_kind;
using symmetrix::execution::polar::validate_polar_program;

static int failures = 0;

static void check(const bool condition, const std::string& what)
{
    if (!condition) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

static LongRangeContract checkpoint_m_contract()
{
    LongRangeContract contract;
    contract.kspace_cutoff = 1.8467166423797607;
    contract.density_max_l = 1;
    contract.density_smearing_width = 1.5;
    contract.feature_max_l = 1;
    contract.feature_smearing_widths = {1.5, 3.0};
    contract.charges_dimension = 8;
    contract.field_dimension = 8;
    contract.potential_dimension = 16;
    contract.field_norm_factor = 1.0;
    contract.recursion_step_count = 2;
    contract.spin_channel_count = 2;
    return contract;
}

int main()
{
    const LongRangeContract contract = checkpoint_m_contract();
    DeltaContract delta;
    delta.node_feature_dimension = 2048;
    delta.num_elements = 83;
    delta.interaction_count = 2;

    const std::vector<symmetrix::execution::OperatorNode> delta_nodes =
        delta_operator_nodes(contract, delta.node_feature_dimension,
            delta.num_elements, delta.interaction_count);
    check(delta_nodes.size() == delta_kind_specs().size(),
        "one delta node per declared kind");

    Program backbone;
    backbone.precision = "float64";
    const Program program = compile_polar_program(backbone, contract, delta);
    // The assembled program passes its own coherence rule.
    check(!program.operators.empty(), "the assembly produces a program");
    std::printf("polar delta program fingerprint %s\n",
        program.fingerprint.c_str());

    // A program whose charge state produces nothing: the node stays declared
    // (the completeness rule is satisfied) but its consumer's chain-state
    // input has no delta producer, which the pass must refuse by name.  This
    // is the coherence rule's own rejection rather than the completeness
    // rule's, which a missing node would trip first.
    Program broken = program;
    for (auto& node : broken.operators) {
        if (std::string_view(operator_kind_name(node)) != "polar_charge_state")
            continue;
        // Keep the state the delta group itself consumes, so the refusal under
        // test is the *long-range* node's missing chain state rather than the
        // delta group's own.
        std::vector<symmetrix::execution::BufferContract> kept;
        for (const auto& output : node.outputs)
            if (output.name == "spin_charge_density")
                kept.push_back(output);
        node.outputs = kept;
    }
    const std::string reason =
        validate_polar_program(broken, contract,
            symmetrix::execution::polar::polar_execution_schedule(
                contract.recursion_step_count), delta);
    check(reason.find("chain-state") != std::string::npos,
        "the missing charge state is refused as a chain-state producer: "
            + reason);
    check(reason.find("multipoles_alpha") != std::string::npos,
        "the refusal names the unproduced buffer: " + reason);

    // And a program without the delta group at all keeps the declaration-only
    // refusal rather than the chain-state one.
    Program slice;
    for (const auto& node : program.operators)
        if (find_delta_kind(operator_kind_name(node)).has_value())
            continue;
        else
            slice.operators.push_back(node);
    const std::string slice_reason = validate_polar_program(slice, contract,
        symmetrix::execution::polar::polar_execution_schedule(
            contract.recursion_step_count));
    check(slice_reason.find("chain-state") == std::string::npos,
        "the declaration-only slice keeps its own refusal: " + slice_reason);

    if (failures == 0)
        std::printf("PASS delta kind coherence\n");
    return failures == 0 ? 0 : 1;
}
"""


def _delta_probe_output(tmp_path: Path) -> str:
    """Compile and run the delta coherence probe, returning its stdout."""

    import shutil
    import subprocess

    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the delta coherence probe")
    source_root = Path(__file__).resolve().parents[2] / "libsymmetrix" / "source"
    source = tmp_path / "delta_kinds_probe.cpp"
    source.write_text(DELTA_KIND_PROBE)
    binary = tmp_path / "delta_kinds_probe"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            f"-I{source_root}",
            "-o",
            str(binary),
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    completed = subprocess.run([str(binary)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "FAIL" not in completed.stdout
    return completed.stdout


def test_the_coherence_rule_refuses_an_unproduced_chain_state(tmp_path):
    """The producer/consumer rule's rejection is reachable and named.

    The rule lives in the compiled assembly, so it is gated where it lives: a
    probe builds the coherent program, removes the charge state from a copy,
    and asserts the pass refuses the resulting dangling chain state by name --
    the negative case the data-side tests can only assert positively.
    """

    output = _delta_probe_output(tmp_path)
    assert "PASS delta kind coherence" in output
    assert (
        "polar delta program fingerprint "
        "sha256:33e78b2d688c282ab32c1be7110724e861fca23901c58d9e5170f85469797cec"
        in output
    )


def test_the_probe_needs_no_package_import():
    """The probe's source root resolves from the installed package layout."""

    source_root = Path(__file__).resolve().parents[2] / "libsymmetrix" / "source"
    assert (source_root / "mace_polar_delta_kinds.hpp").is_file()
