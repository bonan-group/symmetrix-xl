"""Prepare an MH-1 staged host plugin from a model JSON, without a calculator.

The nonlinear and polar families share the generated MH-1 backbone subprogram.
Their frontend renders its host plugin inline while it constructs the
calculator, and the shared JIT entry point cannot stand in for a declared
family adapter because it takes rendered source, not a model file. This module
reads the staged node contract either model carries, renders the plugin for that
topology, compiles it through the shared JIT cache, and returns the prepared
artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jit import (
    JIT_GENERATION_VERSION,
    prepare_jit_artifact,
    validate_artifact_envelope,
)
from .mace_jit_artifact_common import (
    read_model_json,
    require_matching_generation_version,
)
from .mace_nonlinear_execution_contract import MH1_CONTRACT_KEY
from .mace_nonlinear_jit_codegen import (
    MH1_HOST_PLUGIN_V7_ABI,
    MH1_HOST_PLUGIN_V7_ABI_VERSION,
    NONLINEAR_ARTIFACT_ENVELOPE,
    execution_mh1_node_program_metadata,
    render_jit_mh1_host_plugin_v7,
)

#: The composed-only host generator identity.  It is part of the cache key.
GENERATOR = "symmetrix.jit.mh1-host-v7-composed"

#: Compatibility alias retained for callers that imported the old name.
GENERATOR_V6 = GENERATOR

#: The interaction count the legacy ABI-v5 host plugin admits.  The current
#: frontend always publishes composed ABI v7, whose pointer-backed tables carry
#: the contract's exact interaction count.  Keep this constant for callers that
#: still inspect the retired compatibility boundary.
V5_INTERACTION_COUNT = 2

#: The artifact name the compact frontend publishes for this artifact.
ARTIFACT_NAME = "jit_mh1_host_plugin"

#: The node-state storage policy the frontend applies when a caller requests
#: none, matching the compact calculator's own default.
DEFAULT_NODE_STATE_POLICY = "full-retention-v1"

#: The model types whose declared generated subprogram is MH1_UVU.
MODEL_TYPES = ("MACE_Nonlinear", "MACE_Polar")


class MaceNonlinearHostArtifactError(RuntimeError):
    """Raised when an explicitly requested MH-1 host artifact cannot be prepared."""

    def __init__(self, reason, *, diagnostics=(), cache_key=None):
        super().__init__(str(reason))
        self.diagnostics = tuple(str(value) for value in diagnostics)
        self.cache_key = None if cache_key is None else str(cache_key)


@dataclass(frozen=True)
class MaceNonlinearHostArtifactResult:
    """A compiled and cache-published MH-1 staged host plugin."""

    status: str
    precision: str
    model_type: str
    model_path: Path
    cache_key: str
    artifact_path: Path
    manifest_path: Path
    artifact_id: str
    abi: str
    abi_version: int
    node_state_policy: str
    final_output_specializations: tuple[str, ...]
    diagnostics: tuple[str, ...]

    @property
    def available(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "available": True,
            "target": "host",
            "precision": self.precision,
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "cache_key": self.cache_key,
            "artifact_path": str(self.artifact_path),
            "manifest_path": str(self.manifest_path),
            "artifact_id": self.artifact_id,
            "abi": self.abi,
            "abi_version": self.abi_version,
            "node_state_policy": self.node_state_policy,
            "final_output_specializations": list(self.final_output_specializations),
            "diagnostics": list(self.diagnostics),
        }


def prepare_mh1_host_artifact(
    model_path,
    *,
    precision: str,
    target: str = "host",
    cache_root=None,
    node_state_policy: str = DEFAULT_NODE_STATE_POLICY,
) -> MaceNonlinearHostArtifactResult:
    """Prepare and cache the MH-1 staged host plugin for a model JSON.

    Every topology is published through the count-agnostic composed-product
    ABI v7.  Product contraction code and coefficients remain exclusively in
    canonical M0/M1 artifacts.

    ``target`` is the declared artifact adapter's execution target.  ``"host"``
    is the target this entry point serves, and any other target -- the CUDA and
    HIP targets the family's generated modules bind during calculator
    construction -- is refused by name before the model file is read, so a
    frontend that resolved the wrong target for a request fails at preparation
    instead of rendering a host plugin for a device request.

    ``node_state_policy`` is the node-state storage policy the renderer and its
    metadata are built with, so the compiled plugin and the identity published
    beside it agree on the policy the artifact implements.
    """

    if target != "host":
        raise MaceNonlinearHostArtifactError(
            "the MH-1 host artifact adapter serves the host target only, "
            f"not {target!r}"
        )
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")

    try:
        model_path, model_data = read_model_json(
            model_path, error_type=MaceNonlinearHostArtifactError, label="nonlinear"
        )
        model_type = model_data.get("model_type")
        if model_type not in MODEL_TYPES:
            raise MaceNonlinearHostArtifactError(
                "MH-1 host artifact preparation supports "
                f"{', '.join(MODEL_TYPES)} models only"
            )
        stored = model_data.get("execution_contracts")
        contract = stored.get(MH1_CONTRACT_KEY) if isinstance(stored, dict) else None
        if not isinstance(contract, dict):
            raise MaceNonlinearHostArtifactError(
                f"the model does not contain an Execution {MH1_CONTRACT_KEY} contract"
            )
        from . import family_registry

        family = family_registry.declared_registry().detect(model_type=model_type)
        route = family.generated_contract if family is not None else None
        validator_symbol = route.validator if route is not None else None
        if validator_symbol is None:
            raise MaceNonlinearHostArtifactError(
                f"the {model_type} family does not declare a generated-contract validator"
            )
        validator = family_registry.resolve_symbol(validator_symbol)
        contract = validator(model_data, contract)

        # Generation 19 admits only the composed-product host artifact.
        generator = GENERATOR
        render_function = render_jit_mh1_host_plugin_v7
        plugin_abi = (MH1_HOST_PLUGIN_V7_ABI, MH1_HOST_PLUGIN_V7_ABI_VERSION)

        # The frontend renders this plugin from the validated staged contract
        # and republishes the node-program metadata's ABI as the plugin ABI it
        # actually implements: the metadata describes the ABI-v4 node program
        # the artifact carries, while the artifact itself is the selected
        # plugin generation.
        require_matching_generation_version(error_type=MaceNonlinearHostArtifactError)
        metadata = execution_mh1_node_program_metadata(
            contract, backend="host", node_state_policy=node_state_policy
        )
        selected_node_state_policy = metadata["node_state_policy"]
        metadata = {
            **metadata,
            "abi": plugin_abi[0],
            "abi_version": plugin_abi[1],
        }
        validate_artifact_envelope(metadata, NONLINEAR_ARTIFACT_ENVELOPE)
        source = render_function(
            contract,
            precision=precision,
            node_state_policy=node_state_policy,
        )
        result = prepare_jit_artifact(
            source,
            abi={"tag": metadata["abi"], "version": metadata["abi_version"]},
            build={
                "generator": generator,
                "precision": precision,
                "contract": metadata,
                "jit_generation_version": JIT_GENERATION_VERSION,
            },
            cache_root=cache_root,
            cxx_flags=None,
            artifact_name=ARTIFACT_NAME,
        )
    except MaceNonlinearHostArtifactError:
        raise
    except Exception as error:
        raise MaceNonlinearHostArtifactError(error) from error

    if (
        not result.available
        or result.artifact_path is None
        or result.manifest_path is None
    ):
        raise MaceNonlinearHostArtifactError(
            result.reason or "the JIT cache did not produce an artifact",
            diagnostics=result.diagnostics,
            cache_key=result.cache_key,
        )
    return MaceNonlinearHostArtifactResult(
        status=result.status,
        precision=precision,
        model_type=str(model_type),
        model_path=model_path,
        cache_key=str(result.cache_key),
        artifact_path=result.artifact_path,
        manifest_path=result.manifest_path,
        artifact_id=str(metadata["artifact_id"]),
        abi=metadata["abi"],
        abi_version=int(metadata["abi_version"]),
        node_state_policy=selected_node_state_policy,
        final_output_specializations=tuple(
            layer["final_output_specialization"] for layer in metadata["layers"]
        ),
        diagnostics=tuple(result.diagnostics),
    )


__all__ = [
    "DEFAULT_NODE_STATE_POLICY",
    "GENERATOR",
    "MODEL_TYPES",
    "MaceNonlinearHostArtifactError",
    "MaceNonlinearHostArtifactResult",
    "prepare_mh1_host_artifact",
]
