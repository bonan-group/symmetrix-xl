"""Local CPU training for the initial direct-MACE easy-weight profile."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calculator import Symmetrix, neighbor_list

_TRAINING_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _DirectForceBatchGraph:
    node_types: np.ndarray
    receiver_degrees: np.ndarray
    source_indices: np.ndarray
    source_types: np.ndarray
    edge_receivers: np.ndarray
    displacements: np.ndarray
    distances: np.ndarray
    structure_offsets: np.ndarray
    positions: np.ndarray
    edge_shifts: np.ndarray
    cells: np.ndarray
    pbc: np.ndarray


@dataclass(frozen=True)
class _PreparedDirectForceBatch:
    graph: _DirectForceBatchGraph
    generation: int
    edge_structures: np.ndarray
    owner: object
    structure_signature: bytes


_LEGACY_OPTIMIZER_DEFAULTS = {
    "optimizer": "adam",
    "learning_rate": 1.0e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1.0e-8,
}


class TrainingHistory:
    """Read-only snapshots of trainer history records."""

    def __init__(self, records: list[dict[str, Any]]):
        self._records = records

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._records))

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self.records)


class DirectMACEEnergyTrainer:
    """Fine-tune the ``easily-trainable-v1`` direct-MACE parameter subset.

    The learned atomic-energy baseline, compact-radial inputs, H0/A0, Bessel
    and Agnesi values, and ZBL parameters are deliberately fixed. Only direct
    post-spline MACE tensors are exposed to the optimizer.
    """

    def __init__(
        self,
        model_file: str | Path,
        *,
        dtype: str = "float64",
        neighbor_skin: float = 0.5,
        jit_cache: str | Path | None = None,
        head: str | None = None,
    ):
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'.")
        model_file = Path(model_file)
        self.model_data = json.loads(model_file.read_text(encoding="utf-8"))
        if self.model_data.get("model_type", "MACE") != "MACE":
            raise ValueError(
                "DirectMACEEnergyTrainer currently requires ordinary MACE JSON."
            )
        if self.model_data.get("has_field_coupling", False):
            raise ValueError(
                "DirectMACEEnergyTrainer does not support MACEField models."
            )

        import os

        if jit_cache is not None:
            os.environ["SYMMETRIX_JIT_CACHE"] = str(Path(jit_cache))
        self.calculator = Symmetrix(
            model_file,
            dtype=dtype,
            use_kokkos=True,
            streamed_edges="direct",
            execution_profile="speed",
            m1_polynomial_policy="retained",
            execution_mh0_state_policy="full-retention-v1",
            parameter_gradients=True,
            neighbor_skin=neighbor_skin,
            head=head,
        )
        self.evaluator = self.calculator.evaluator
        for setter_name, value in (
            ("_set_m1_polynomial_policy", "retained"),
            ("_set_mh0_state_policy", "full-retention-v1"),
            ("_set_harmonic_storage_policy", "retained"),
            ("_set_readout_policy", "retained"),
            ("_set_phi1_policy", "retained"),
        ):
            setter = getattr(self.evaluator, setter_name, None)
            if callable(setter):
                setter(value)
        enable_direct_gradients = getattr(
            self.evaluator, "set_direct_parameter_gradients", None
        )
        if not callable(enable_direct_gradients):
            raise RuntimeError(
                "The active native evaluator does not support direct training."
            )
        enable_direct_gradients(True, 256 * 1024 * 1024)
        self._head = self.calculator.head
        self._dtype = dtype
        self._neighbor_skin = float(neighbor_skin)
        self._parameters = {
            name: np.asarray(values, dtype=np.float64).copy()
            for name, values in self.evaluator.get_direct_parameters().items()
        }
        self._parameters_stale = False
        self._optimizer_config: dict[str, Any] | None = None
        self._optimizer_source: str | None = None
        self._optimizer_state: dict[str, dict[str, np.ndarray]] = {}
        self._optimizer_step_count = 0
        self._parameter_groups: list[dict[str, Any]] = []
        self._scheduler_config: dict[str, Any] | None = None
        self._scheduler_state: dict[str, Any] = {}
        self._history_records: list[dict[str, Any]] = []
        self._history = TrainingHistory(self._history_records)
        self._global_step = 0
        self._epoch = 0
        self._best_validation: dict[str, Any] | None = None
        self._report_console = False
        self._report_interval = 1
        self._jsonl_path: Path | None = None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self._parameters)

    @property
    def history(self) -> TrainingHistory:
        return self._history

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def learning_rates(self) -> tuple[float, ...]:
        return tuple(group["learning_rate"] for group in self._parameter_groups)

    @property
    def optimizer_config(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._optimizer_config)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        self._epoch = epoch

    @staticmethod
    def _finite_positive(name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
        return value

    @staticmethod
    def _finite_non_negative(name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return value

    def _normalize_parameter_groups(
        self,
        parameter_groups: Sequence[Mapping[str, Any]] | None,
        learning_rate: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        if parameter_groups is None:
            return [
                {
                    "names": list(self.parameter_names),
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                }
            ]
        if not parameter_groups:
            raise ValueError("parameter_groups must not be empty.")
        known = set(self.parameter_names)
        assigned: set[str] = set()
        normalized = []
        for index, raw_group in enumerate(parameter_groups):
            if not isinstance(raw_group, Mapping):
                raise TypeError(f"Parameter group {index} must be a mapping.")
            unknown_keys = set(raw_group) - {
                "names",
                "learning_rate",
                "weight_decay",
            }
            if unknown_keys:
                keys = ", ".join(sorted(unknown_keys))
                raise ValueError(f"Parameter group {index} has unknown keys: {keys}")
            names = list(raw_group.get("names", ()))
            if not names or any(not isinstance(name, str) for name in names):
                raise ValueError(
                    f"Parameter group {index} must contain non-empty string names."
                )
            duplicates = assigned.intersection(names)
            if len(set(names)) != len(names):
                duplicates.update(name for name in names if names.count(name) > 1)
            if duplicates:
                values = ", ".join(sorted(duplicates))
                raise ValueError(f"Parameter names occur in multiple groups: {values}")
            unknown = set(names) - known
            if unknown:
                values = ", ".join(sorted(unknown))
                raise ValueError(f"Unknown direct training parameters: {values}")
            group_learning_rate = self._finite_positive(
                f"parameter_groups[{index}].learning_rate",
                raw_group.get("learning_rate", learning_rate),
            )
            group_weight_decay = self._finite_non_negative(
                f"parameter_groups[{index}].weight_decay",
                raw_group.get("weight_decay", weight_decay),
            )
            normalized.append(
                {
                    "names": names,
                    "learning_rate": group_learning_rate,
                    "weight_decay": group_weight_decay,
                }
            )
            assigned.update(names)
        missing = known - assigned
        if missing:
            values = ", ".join(sorted(missing))
            raise ValueError(f"Parameter groups do not cover parameters: {values}")
        return normalized

    def configure_optimizer(
        self,
        *,
        optimizer: str = "adam",
        learning_rate: float = 1.0e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1.0e-8,
        amsgrad: bool = False,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        gradient_clip_norm: float | None = None,
        parameter_groups: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Configure persistent optimizer state for subsequent training calls."""
        optimizer = str(optimizer).lower()
        if optimizer not in ("sgd", "adam", "adamw"):
            raise ValueError("optimizer must be 'sgd', 'adam', or 'adamw'.")
        learning_rate = self._finite_positive("learning_rate", learning_rate)
        weight_decay = self._finite_non_negative("weight_decay", weight_decay)
        if gradient_clip_norm is not None:
            gradient_clip_norm = self._finite_positive(
                "gradient_clip_norm", gradient_clip_norm
            )
        if not isinstance(amsgrad, bool):
            raise TypeError("amsgrad must be a boolean.")
        if not isinstance(nesterov, bool):
            raise TypeError("nesterov must be a boolean.")
        beta1 = float(beta1)
        beta2 = float(beta2)
        epsilon = self._finite_positive("epsilon", epsilon)
        momentum = float(momentum)
        if not 0.0 <= beta1 < 1.0 or not np.isfinite(beta1):
            raise ValueError("beta1 must be finite and in [0, 1).")
        if not 0.0 <= beta2 < 1.0 or not np.isfinite(beta2):
            raise ValueError("beta2 must be finite and in [0, 1).")
        if not 0.0 <= momentum < 1.0 or not np.isfinite(momentum):
            raise ValueError("momentum must be finite and in [0, 1).")
        if optimizer == "sgd" and amsgrad:
            raise ValueError("amsgrad is only valid for Adam and AdamW.")
        if optimizer != "sgd" and (momentum != 0.0 or nesterov):
            raise ValueError("momentum and nesterov are only valid for SGD.")
        if nesterov and momentum <= 0.0:
            raise ValueError("nesterov requires positive SGD momentum.")
        groups = self._normalize_parameter_groups(
            parameter_groups, learning_rate, weight_decay
        )
        self._optimizer_config = {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "amsgrad": amsgrad,
            "momentum": momentum,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "parameter_groups": copy.deepcopy(groups),
        }
        self._optimizer_source = "configured"
        self._parameter_groups = groups
        self._optimizer_state = {}
        self._optimizer_step_count = 0
        self._scheduler_config = None
        self._scheduler_state = {}
        self._configure_native_optimizer()

    def _configure_native_optimizer(self) -> None:
        config = self._optimizer_config
        if config is None:
            return
        self.evaluator.configure_direct_optimizer(
            config["optimizer"],
            config["beta1"],
            config["beta2"],
            config["epsilon"],
            config["amsgrad"],
            config["momentum"],
            config["nesterov"],
            0.0
            if config["gradient_clip_norm"] is None
            else config["gradient_clip_norm"],
            [group["names"] for group in self._parameter_groups],
            [group["learning_rate"] for group in self._parameter_groups],
            [group["weight_decay"] for group in self._parameter_groups],
        )
        self._optimizer_step_count = 0
        self._optimizer_state = {}

    def _resolve_optimizer(
        self,
        *,
        optimizer: str | None,
        learning_rate: float | None,
        beta1: float | None,
        beta2: float | None,
        epsilon: float | None,
    ) -> None:
        overrides = {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
        }
        supplied = [name for name, value in overrides.items() if value is not None]
        if self._optimizer_source == "configured":
            if supplied:
                names = ", ".join(supplied)
                raise ValueError(
                    "Per-call optimizer arguments conflict with the configured "
                    f"optimizer: {names}."
                )
            return
        legacy = {
            name: defaults if overrides[name] is None else overrides[name]
            for name, defaults in _LEGACY_OPTIMIZER_DEFAULTS.items()
        }
        candidate = self._validated_optimizer_configuration(**legacy)
        if self._optimizer_config != candidate:
            self.configure_optimizer(**legacy)
            self._optimizer_source = "legacy"

    def _validated_optimizer_configuration(self, **legacy) -> dict[str, Any]:
        optimizer = str(legacy["optimizer"]).lower()
        if optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer must be 'sgd' or 'adam'.")
        learning_rate = self._finite_positive("learning_rate", legacy["learning_rate"])
        beta1 = float(legacy["beta1"])
        beta2 = float(legacy["beta2"])
        epsilon = self._finite_positive("epsilon", legacy["epsilon"])
        if not 0.0 <= beta1 < 1.0 or not np.isfinite(beta1):
            raise ValueError("beta1 must be finite and in [0, 1).")
        if not 0.0 <= beta2 < 1.0 or not np.isfinite(beta2):
            raise ValueError("beta2 must be finite and in [0, 1).")
        groups = self._normalize_parameter_groups(None, learning_rate, 0.0)
        return {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "amsgrad": False,
            "momentum": 0.0,
            "nesterov": False,
            "weight_decay": 0.0,
            "gradient_clip_norm": None,
            "parameter_groups": copy.deepcopy(groups),
        }

    def configure_lr_scheduler(self, scheduler: str | None, **kwargs) -> None:
        """Configure an explicitly stepped learning-rate scheduler."""
        if self._optimizer_config is None:
            raise RuntimeError("Configure an optimizer before its scheduler.")
        if scheduler is None:
            if kwargs:
                raise ValueError("A disabled scheduler accepts no options.")
            self._scheduler_config = None
            self._scheduler_state = {}
            return
        scheduler = str(scheduler).lower()
        if scheduler == "exponential":
            unknown = set(kwargs) - {"gamma"}
            if unknown:
                raise ValueError(f"Unknown exponential scheduler options: {unknown}")
            gamma = self._finite_positive("gamma", kwargs.get("gamma", 0.99))
            self._scheduler_config = {"scheduler": scheduler, "gamma": gamma}
            self._scheduler_state = {"step_count": 0}
            return
        if scheduler != "reduce_on_plateau":
            raise ValueError(
                "scheduler must be 'exponential', 'reduce_on_plateau', or None."
            )
        allowed = {
            "mode",
            "factor",
            "patience",
            "threshold",
            "threshold_mode",
            "cooldown",
            "min_lr",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unknown plateau scheduler options: {unknown}")
        mode = kwargs.get("mode", "min")
        threshold_mode = kwargs.get("threshold_mode", "rel")
        factor = float(kwargs.get("factor", 0.1))
        patience = kwargs.get("patience", 10)
        threshold = self._finite_non_negative(
            "threshold", kwargs.get("threshold", 1.0e-4)
        )
        cooldown = kwargs.get("cooldown", 0)
        min_lr = self._finite_non_negative("min_lr", kwargs.get("min_lr", 0.0))
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'.")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError("threshold_mode must be 'rel' or 'abs'.")
        if not np.isfinite(factor) or not 0.0 < factor < 1.0:
            raise ValueError("factor must be finite and in (0, 1).")
        for name, value in (("patience", patience), ("cooldown", cooldown)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        self._scheduler_config = {
            "scheduler": scheduler,
            "mode": mode,
            "factor": factor,
            "patience": patience,
            "threshold": threshold,
            "threshold_mode": threshold_mode,
            "cooldown": cooldown,
            "min_lr": min_lr,
        }
        self._scheduler_state = {
            "step_count": 0,
            "best": None,
            "bad_epochs": 0,
            "cooldown_counter": 0,
        }

    def scheduler_step(self, metric: float | None = None) -> tuple[float, ...]:
        if self._scheduler_config is None:
            raise RuntimeError("No learning-rate scheduler is configured.")
        config = self._scheduler_config
        state = self._scheduler_state
        if config["scheduler"] == "exponential":
            if metric is not None:
                raise ValueError("The exponential scheduler does not accept a metric.")
            for group in self._parameter_groups:
                group["learning_rate"] *= config["gamma"]
        else:
            if metric is None or not np.isfinite(metric):
                raise ValueError("A finite metric is required for reduce_on_plateau.")
            metric = float(metric)
            best = state["best"]
            improved = best is None or self._plateau_metric_improved(metric, best)
            if improved:
                state["best"] = metric
                state["bad_epochs"] = 0
            else:
                state["bad_epochs"] += 1
            if state["cooldown_counter"] > 0:
                state["cooldown_counter"] -= 1
                state["bad_epochs"] = 0
            if state["bad_epochs"] > config["patience"]:
                for group in self._parameter_groups:
                    old_learning_rate = group["learning_rate"]
                    reduced = max(
                        old_learning_rate * config["factor"], config["min_lr"]
                    )
                    group["learning_rate"] = min(old_learning_rate, reduced)
                state["cooldown_counter"] = config["cooldown"]
                state["bad_epochs"] = 0
        state["step_count"] += 1
        self._epoch += 1
        self.evaluator.set_direct_optimizer_learning_rates(list(self.learning_rates))
        return self.learning_rates

    def _plateau_metric_improved(self, metric: float, best: float) -> bool:
        config = self._scheduler_config
        if config["mode"] == "min":
            boundary = (
                best * (1.0 - config["threshold"])
                if config["threshold_mode"] == "rel"
                else best - config["threshold"]
            )
            return metric < boundary
        boundary = (
            best * (1.0 + config["threshold"])
            if config["threshold_mode"] == "rel"
            else best + config["threshold"]
        )
        return metric > boundary

    def enable_reporting(
        self,
        *,
        console: bool = False,
        jsonl_path: str | Path | None = None,
        interval: int = 1,
    ) -> None:
        if not isinstance(console, bool):
            raise TypeError("console must be a boolean.")
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError("interval must be a positive integer.")
        self._report_console = console
        self._report_interval = interval
        self._jsonl_path = None if jsonl_path is None else Path(jsonl_path)
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        stored = copy.deepcopy(dict(record))
        self._history_records.append(stored)
        if self._jsonl_path is not None:
            with self._jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(stored, separators=(",", ":")) + "\n")
                stream.flush()
        if self._report_console and (
            stored.get("type") != "train"
            or int(stored.get("step", 0)) % self._report_interval == 0
        ):
            fields = [
                f"type={stored['type']}",
                f"step={stored.get('step', self._global_step)}",
                f"epoch={stored.get('epoch', self._epoch)}",
            ]
            for name in ("loss", "mean_loss", "learning_rate"):
                if name in stored:
                    fields.append(f"{name}={stored[name]:.8g}")
            print("symmetrix training: " + " ".join(fields), flush=True)
        return copy.deepcopy(stored)

    def export_history_jsonl(self, path: str | Path, *, append: bool = False) -> int:
        """Write a snapshot of all current history records as JSON Lines."""
        if not isinstance(append, bool):
            raise TypeError("append must be a boolean.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with destination.open(mode, encoding="utf-8") as stream:
            for record in self._history_records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
        return len(self._history_records)

    def record_validation(
        self,
        *,
        metrics: Mapping[str, float],
        count: int,
    ) -> dict[str, Any]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer.")
        if not metrics:
            raise ValueError("metrics must not be empty.")
        normalized = {}
        for name, value in metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings.")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"Validation metric {name!r} must be finite.")
            normalized[name] = value
        record = {
            "type": "validation",
            "step": self._global_step,
            "epoch": self._epoch,
            "count": count,
            "metrics": normalized,
        }
        if "loss" in normalized and (
            self._best_validation is None
            or normalized["loss"] < self._best_validation["loss"]
        ):
            self._best_validation = {
                "loss": normalized["loss"],
                "step": self._global_step,
                "epoch": self._epoch,
            }
        return self._append_record(record)

    def get_parameters(self) -> dict[str, np.ndarray]:
        """Export current native optimizer parameters as host copies."""
        exported = self.evaluator.get_direct_parameters()
        self._parameters = {
            name: np.asarray(values, dtype=np.float64).copy()
            for name, values in exported.items()
        }
        self._parameters_stale = False
        return {name: value.copy() for name, value in self._parameters.items()}

    def _sync_optimizer_state_from_native(self) -> None:
        self._optimizer_state = {
            state_name: {
                name: np.asarray(values, dtype=np.float64).copy()
                for name, values in state.items()
            }
            for state_name, state in self.evaluator.direct_optimizer_state().items()
        }
        self._optimizer_step_count = int(self.evaluator.direct_optimizer_step_count)

    def _sync_optimizer_state_to_native(self) -> None:
        self.evaluator.set_direct_optimizer_state(
            self._optimizer_step_count,
            {
                state_name: {
                    name: np.asarray(values, dtype=np.float64).tolist()
                    for name, values in state.items()
                }
                for state_name, state in self._optimizer_state.items()
            },
        )

    def set_parameters(self, parameters: Mapping[str, np.ndarray]) -> None:
        unknown = set(parameters) - set(self._parameters)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown direct training parameters: {names}")
        if self._parameters_stale:
            self.get_parameters()
        for name, value in parameters.items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != self._parameters[name].shape:
                raise ValueError(
                    f"Parameter {name!r} has shape {array.shape}; expected "
                    f"{self._parameters[name].shape}."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Parameter {name!r} contains non-finite values.")
            self._parameters[name] = array.copy()
        self._sync_parameters()

    def _sync_parameters(self) -> None:
        self.evaluator.set_direct_parameters(
            {name: value.tolist() for name, value in self._parameters.items()}
        )
        self.calculator.reset()
        self._parameters_stale = False

    @staticmethod
    def _validate_directed_edge_counts(counts: Sequence[int]) -> None:
        total = 0
        for count in counts:
            total += count
            if total > np.iinfo(np.int32).max:
                raise ValueError(
                    "Native direct force graph exceeds 32-bit directed edges."
                )

    def _build_direct_force_batch_graph(self, structures):
        """Build disconnected graph state retained for physical-force training."""
        type_by_atomic_number = {
            atomic_number: index
            for index, atomic_number in enumerate(self.evaluator.atomic_numbers)
        }
        node_types = []
        receiver_degrees = []
        source_indices = []
        source_types = []
        edge_receivers = []
        displacements = []
        distances = []
        offsets = [0]
        positions = []
        edge_shifts = []
        cells = []
        pbc = []

        node_offset = 0
        for atoms in structures:
            if atoms is None or len(atoms) == 0:
                raise ValueError("Native batch structures must be non-empty.")
            atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
            unknown = sorted(set(atomic_numbers) - set(type_by_atomic_number))
            if unknown:
                values = ", ".join(str(value) for value in unknown)
                raise ValueError(f"Native batch contains unknown elements: {values}.")

            structure_positions = np.asarray(atoms.positions, dtype=np.float64)
            cell = np.asarray(atoms.cell, dtype=np.float64)
            if structure_positions.shape != (len(atoms), 3) or cell.shape != (3, 3):
                raise ValueError("Native batch structures have invalid geometry.")
            if (
                not np.isfinite(structure_positions).all()
                or not np.isfinite(cell).all()
            ):
                raise ValueError("Native batch structures have non-finite geometry.")

            effective_cutoff = self.calculator.cutoff + self.calculator.neighbor_skin
            neighbor_atoms = atoms
            if np.linalg.matrix_rank(cell) < 3:
                neighbor_atoms = atoms.copy()
                neighbor_atoms.set_cell(atoms.cell.complete(), scale_atoms=False)
            receivers, sources, edge_distances, xyz, shifts = neighbor_list(
                "ijdDS", neighbor_atoms, effective_cutoff
            )
            receivers = np.asarray(receivers, dtype=np.int32)
            sources = np.asarray(sources, dtype=np.int32)
            edge_distances = np.asarray(edge_distances, dtype=np.float64)
            xyz = np.asarray(xyz, dtype=np.float64)
            shifts = np.asarray(shifts, dtype=np.int32).reshape((-1, 3))
            periodic_axes = np.flatnonzero(atoms.pbc)
            normalized_shifts = np.zeros_like(shifts)
            if periodic_axes.size:
                shift_vectors = xyz - (
                    structure_positions[sources] - structure_positions[receivers]
                )
                fractional_shifts = shift_vectors @ np.linalg.pinv(cell[periodic_axes])
                normalized_shifts[:, periodic_axes] = np.rint(fractional_shifts).astype(
                    np.int32
                )
            shifts = normalized_shifts
            order = np.lexsort(
                (shifts[:, 2], shifts[:, 1], shifts[:, 0], sources, receivers)
            )
            receivers = np.ascontiguousarray(receivers[order], dtype=np.int32)
            sources = np.ascontiguousarray(sources[order], dtype=np.int32)
            edge_distances = np.ascontiguousarray(
                edge_distances[order], dtype=np.float64
            )
            xyz = np.ascontiguousarray(xyz[order], dtype=np.float64)
            shifts = np.ascontiguousarray(shifts[order], dtype=np.int32)

            local_types = np.asarray(
                [type_by_atomic_number[value] for value in atomic_numbers],
                dtype=np.int32,
            )
            r = edge_distances
            if (
                not np.isfinite(xyz).all()
                or not np.isfinite(r).all()
                or not (r > 0.0).all()
            ):
                raise ValueError(
                    "Native batch graph has an invalid distance or displacement."
                )
            inactive = r >= self.calculator.cutoff
            if np.any(inactive):
                xyz[inactive] *= (self.calculator.cutoff / r[inactive])[:, None]
                r[inactive] = self.calculator.cutoff

            limit = np.iinfo(np.int32).max
            offset_sources = sources.astype(np.int64) + node_offset
            offset_receivers = receivers.astype(np.int64) + node_offset
            if node_offset + len(structure_positions) > limit or np.any(
                offset_sources > limit
            ):
                raise ValueError("Native batch graph exceeds 32-bit node indices.")

            node_types.append(local_types)
            receiver_degrees.append(
                np.bincount(receivers, minlength=len(structure_positions)).astype(
                    np.int32
                )
            )
            source_indices.append(offset_sources.astype(np.int32))
            source_types.append(np.ascontiguousarray(local_types[sources]))
            edge_receivers.append(offset_receivers.astype(np.int32))
            displacements.append(np.ascontiguousarray(xyz, dtype=np.float64))
            distances.append(np.ascontiguousarray(r, dtype=np.float64))
            positions.append(np.ascontiguousarray(structure_positions))
            edge_shifts.append(np.ascontiguousarray(shifts, dtype=np.int32))
            cells.append(cell)
            pbc.append(np.asarray(atoms.pbc, dtype=np.int32))

            node_offset += len(structure_positions)
            offsets.append(node_offset)

        self._validate_directed_edge_counts(
            [edge_sources.size for edge_sources in source_indices]
        )
        node_types = np.concatenate(node_types).astype(np.int32, copy=False)
        receiver_degrees = np.concatenate(receiver_degrees).astype(np.int32, copy=False)
        source_indices = np.concatenate(source_indices).astype(np.int32, copy=False)
        source_types = np.concatenate(source_types).astype(np.int32, copy=False)
        edge_receivers = np.concatenate(edge_receivers).astype(np.int32, copy=False)
        displacements = np.concatenate(displacements).reshape(-1)
        distances = np.concatenate(distances)
        structure_offsets = np.asarray(offsets, dtype=np.uint64)
        positions = np.ascontiguousarray(np.concatenate(positions), dtype=np.float64)
        edge_shifts = np.ascontiguousarray(np.concatenate(edge_shifts), dtype=np.int32)
        cells = np.ascontiguousarray(np.stack(cells), dtype=np.float64)
        pbc = np.ascontiguousarray(np.stack(pbc), dtype=np.int32)
        if receiver_degrees.sum(dtype=np.int64) != len(source_indices):
            raise RuntimeError("Native batch receiver degrees are inconsistent.")
        return _DirectForceBatchGraph(
            node_types=node_types,
            receiver_degrees=receiver_degrees,
            source_indices=source_indices,
            source_types=source_types,
            edge_receivers=edge_receivers,
            displacements=displacements,
            distances=distances,
            structure_offsets=structure_offsets,
            positions=positions,
            edge_shifts=edge_shifts,
            cells=cells,
            pbc=pbc,
        )

    def _prepare_direct_force_batch(self, structures):
        if self.evaluator.streamed_edges_mode != "direct":
            raise RuntimeError(
                "Native batch preparation requires streamed_edges='direct'."
            )
        structures = list(structures)
        graph = self._build_direct_force_batch_graph(structures)
        edge_structures = (
            np.searchsorted(graph.structure_offsets, graph.edge_receivers, side="right")
            - 1
        ).astype(np.int32)
        generation = self.evaluator._prepare_factorized_graph(
            len(graph.node_types),
            graph.node_types,
            graph.receiver_degrees,
            graph.source_indices,
            graph.source_types,
        )
        return _PreparedDirectForceBatch(
            graph=graph,
            generation=generation,
            edge_structures=edge_structures,
            owner=self,
            structure_signature=self._direct_force_batch_signature(structures),
        )

    @staticmethod
    def _direct_force_batch_signature(structures):
        digest = hashlib.sha256()
        digest.update(np.asarray([len(structures)], dtype=np.int64).tobytes())
        for atoms in structures:
            arrays = (
                np.asarray(atoms.numbers, dtype=np.int32),
                np.asarray(atoms.positions, dtype=np.float64),
                np.asarray(atoms.cell, dtype=np.float64),
                np.asarray(atoms.pbc, dtype=np.uint8),
            )
            for values in arrays:
                contiguous = np.ascontiguousarray(values)
                digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
                digest.update(contiguous.tobytes())
        return digest.digest()

    def _validate_prepared_direct_force_batch(self, prepared, structures):
        if not isinstance(prepared, _PreparedDirectForceBatch):
            raise TypeError("Prepared batch has an invalid type.")
        if prepared.owner is not self:
            raise ValueError("Prepared batch belongs to a different trainer.")
        if prepared.structure_signature != self._direct_force_batch_signature(
            structures
        ):
            raise ValueError("Prepared batch geometry does not match the structures.")
        if prepared.generation != self.evaluator.factorized_graph_generation:
            raise ValueError("Prepared batch graph is no longer current.")
        return prepared

    def _evaluate_native_batch_loss_from_prepared(
        self, prepared, references, *, energy_residual_scales=None
    ):
        graph = prepared.graph
        evaluation = self.evaluator._compute_prepared_factorized_energy_loss(
            prepared.generation,
            graph.displacements,
            graph.distances,
            graph.structure_offsets,
            np.asarray(references, dtype=np.float64),
            (
                None
                if energy_residual_scales is None
                else np.asarray(energy_residual_scales, dtype=np.float64)
            ),
        )
        return evaluation

    def _evaluate_native_batch_loss(
        self, structures, references, *, energy_residual_scales=None
    ):
        prepared = self._prepare_direct_force_batch(structures)
        return self._evaluate_native_batch_loss_from_prepared(
            prepared,
            references,
            energy_residual_scales=energy_residual_scales,
        )

    def energy_and_loss_gradients(
        self,
        atoms,
        reference_energy: float | None = None,
    ) -> tuple[float, float | None, dict[str, np.ndarray]]:
        """Evaluate one structure and optionally return an MSE-like loss gradient."""
        # This diagnostic API must execute reverse even when ASE has a cached
        # energy for an equivalent structure.
        self.calculator.reset()
        atoms.calc = self.calculator
        energy = float(atoms.get_potential_energy())
        result = self.evaluator.direct_parameter_gradients()
        if not result["enabled"]:
            raise RuntimeError("Direct parameter gradients were disabled.")
        if not result["ready"]:
            raise RuntimeError(
                "Direct parameter gradients are not ready after evaluation."
            )
        gradients = {
            group["name"]: np.asarray(group["values"], dtype=np.float64).copy()
            for group in result["groups"]
        }
        if reference_energy is None:
            return energy, None, gradients

        reference = float(reference_energy)
        if not np.isfinite(reference):
            raise ValueError("reference_energy must be finite.")
        error = energy - reference
        loss = 0.5 * error * error
        factor = error
        for name in gradients:
            gradients[name] *= factor
        return energy, loss, gradients

    def _validate_gradients(
        self, gradients: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        missing = set(self._parameters) - set(gradients)
        unknown = set(gradients) - set(self._parameters)
        if missing or unknown:
            raise ValueError(
                "Gradient parameter names do not match the trainable parameter set."
            )
        normalized = {}
        for name, value in gradients.items():
            gradient = np.asarray(value, dtype=np.float64)
            if gradient.shape != self._parameters[name].shape:
                raise ValueError(
                    f"Gradient {name!r} has shape {gradient.shape}; expected "
                    f"{self._parameters[name].shape}."
                )
            if not np.all(np.isfinite(gradient)):
                raise ValueError(f"Gradient {name!r} contains non-finite values.")
            gradient = gradient.copy()
            normalized[name] = gradient
        return normalized

    def _clip_gradients(
        self, gradients: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, float | bool | None]]:
        normalized = self._validate_gradients(gradients)
        gradient_norm = float(
            np.sqrt(
                sum(
                    float(np.dot(value.ravel(), value.ravel()))
                    for value in normalized.values()
                )
            )
        )
        if not np.isfinite(gradient_norm):
            raise ValueError("The global gradient norm is non-finite.")
        maximum = self._optimizer_config["gradient_clip_norm"]
        coefficient = 1.0
        if maximum is not None and gradient_norm > maximum:
            coefficient = maximum / gradient_norm
            for name in normalized:
                normalized[name] *= coefficient
        return normalized, {
            "gradient_norm": gradient_norm,
            "clipped": coefficient < 1.0,
            "clip_coefficient": coefficient,
            "gradient_clip_norm": maximum,
        }

    def _state_buffer(self, state_name: str, parameter_name: str) -> np.ndarray:
        state = self._optimizer_state.setdefault(state_name, {})
        if parameter_name not in state:
            state[parameter_name] = np.zeros_like(self._parameters[parameter_name])
        return state[parameter_name]

    def _optimizer_step(
        self, gradients: Mapping[str, np.ndarray]
    ) -> dict[str, float | bool | None]:
        gradients, clipping = self._clip_gradients(gradients)
        config = self._optimizer_config
        optimizer = config["optimizer"]
        self._optimizer_step_count += 1
        for group in self._parameter_groups:
            learning_rate = group["learning_rate"]
            weight_decay = group["weight_decay"]
            for name in group["names"]:
                parameter = self._parameters[name]
                gradient = gradients[name]
                if optimizer == "sgd":
                    if weight_decay:
                        gradient = gradient + weight_decay * parameter
                    if config["momentum"]:
                        buffer = self._state_buffer("momentum", name)
                        if self._optimizer_step_count == 1:
                            buffer[...] = gradient
                        else:
                            buffer *= config["momentum"]
                            buffer += gradient
                        gradient = (
                            gradient + config["momentum"] * buffer
                            if config["nesterov"]
                            else buffer
                        )
                    parameter -= learning_rate * gradient
                    continue

                if optimizer == "adamw" and weight_decay:
                    parameter *= 1.0 - learning_rate * weight_decay
                elif optimizer == "adam" and weight_decay:
                    gradient = gradient + weight_decay * parameter
                first_moment = self._state_buffer("first_moment", name)
                second_moment = self._state_buffer("second_moment", name)
                first_moment *= config["beta1"]
                first_moment += (1.0 - config["beta1"]) * gradient
                second_moment *= config["beta2"]
                second_moment += (1.0 - config["beta2"]) * gradient * gradient
                denominator_moment = second_moment
                if config["amsgrad"]:
                    maximum = self._state_buffer("max_second_moment", name)
                    np.maximum(maximum, second_moment, out=maximum)
                    denominator_moment = maximum
                first_unbiased = first_moment / (
                    1.0 - config["beta1"] ** self._optimizer_step_count
                )
                second_unbiased = denominator_moment / (
                    1.0 - config["beta2"] ** self._optimizer_step_count
                )
                parameter -= (
                    learning_rate
                    * first_unbiased
                    / (np.sqrt(second_unbiased) + config["epsilon"])
                )
        self._sync_parameters()
        return clipping

    def _native_optimizer_step(self) -> dict[str, float | bool | None]:
        diagnostics = dict(self.evaluator.apply_direct_optimizer_step())
        self._optimizer_step_count = int(diagnostics.pop("step"))
        self._parameters_stale = True
        self.calculator.reset()
        return {
            "gradient_norm": float(diagnostics["gradient_norm"]),
            "clipped": bool(diagnostics["clipped"]),
            "clip_coefficient": float(diagnostics["clip_coefficient"]),
            "gradient_clip_norm": self._optimizer_config["gradient_clip_norm"],
        }

    def _handle_failed_native_training(
        self,
        *,
        gradients_ready_before: bool,
        capture_count_before: int,
        evaluation_completed: bool,
    ) -> None:
        preserve_existing_capture = (
            not evaluation_completed
            and gradients_ready_before
            and self.evaluator.direct_parameter_gradients_ready
            and self.evaluator.direct_parameter_gradient_capture_count
            == capture_count_before
        )
        if not preserve_existing_capture:
            self.evaluator._invalidate_direct_training_state()

    def _direct_training_diagnostics(self) -> dict[str, Any]:
        environment = dict(self.evaluator.execution_device_execution_environment)
        backend = str(environment.get("backend", "")).lower()
        device_backend = backend in ("cuda", "hip")
        return {
            "training_backend": backend if device_backend else "cpu",
            "training_device_artifact_id": (
                self.evaluator.jit_device_plugin_artifact_id
                if device_backend
                else self.evaluator.jit_host_plugin_artifact_id
            ),
            "parameter_gradient_kernel_policy": (
                self.evaluator.direct_parameter_gradient_kernel_policy
            ),
            "training_workspace_bytes": int(
                self.evaluator.direct_training_workspace_bytes
            ),
            "training_host_to_device_bytes": int(
                self.evaluator.direct_training_host_to_device_bytes
            ),
            "training_device_to_host_bytes": int(
                self.evaluator.direct_training_device_to_host_bytes
            ),
            "training_fallback_count": int(
                self.evaluator.direct_training_fallback_count
            ),
            "training_fallback_reason": str(
                self.evaluator.direct_training_fallback_reason
            ),
        }

    def _training_record(
        self,
        diagnostics: Mapping[str, Any],
        *,
        batch_size: int,
        batch_mode: str,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        self._global_step += 1
        record = {
            "type": "train",
            "step": self._global_step,
            "epoch": self._epoch,
            "optimizer": self._optimizer_config["optimizer"],
            "learning_rate": self._parameter_groups[0]["learning_rate"],
            "learning_rates": list(self.learning_rates),
            "batch_size": batch_size,
            "batch_mode": batch_mode,
            "dtype": self._dtype,
            "elapsed_seconds": float(elapsed_seconds),
            **dict(diagnostics),
        }
        return self._append_record(record)

    def train_step(
        self,
        atoms,
        reference_energy: float,
        *,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        beta1: float | None = None,
        beta2: float | None = None,
        epsilon: float | None = None,
    ) -> dict[str, float | bool | None]:
        """Run one scalar-energy optimizer step and return diagnostics."""
        reference = float(reference_energy)
        if not np.isfinite(reference):
            raise ValueError("reference_energy must be finite.")
        self._resolve_optimizer(
            optimizer=optimizer,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        started = time.perf_counter()
        gradients_ready_before = self.evaluator.direct_parameter_gradients_ready
        capture_count_before = self.evaluator.direct_parameter_gradient_capture_count
        evaluation_completed = False
        try:
            evaluation = self._evaluate_native_batch_loss([atoms], [reference])
            evaluation_completed = True
            energy = float(evaluation["energies"][0])
            loss = float(evaluation["loss"])
            clipping = self._native_optimizer_step()
        except Exception:
            self._handle_failed_native_training(
                gradients_ready_before=gradients_ready_before,
                capture_count_before=capture_count_before,
                evaluation_completed=evaluation_completed,
            )
            raise
        diagnostics = {
            "energy": energy,
            "loss": loss,
            "error": energy - reference,
            **clipping,
            **self._direct_training_diagnostics(),
        }
        self._training_record(
            diagnostics,
            batch_size=1,
            batch_mode="single",
            elapsed_seconds=time.perf_counter() - started,
        )
        return diagnostics

    def step_batch(
        self,
        batch,
        reference_energies,
        *,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        beta1: float | None = None,
        beta2: float | None = None,
        epsilon: float | None = None,
        batch_mode: str = "native",
    ) -> dict[str, float | int]:
        """Accumulate mean scalar-energy gradients over a local CPU batch."""
        structures = list(batch)
        references = [float(value) for value in reference_energies]
        if not structures or len(structures) != len(references):
            raise ValueError(
                "step_batch requires matching, non-empty structures and "
                "reference energies."
            )
        if batch_mode not in ("native", "sequential"):
            raise ValueError("batch_mode must be 'native' or 'sequential'.")
        if any(not np.isfinite(value) for value in references):
            raise ValueError("reference energies must be finite.")
        self._resolve_optimizer(
            optimizer=optimizer,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        started = time.perf_counter()

        if batch_mode == "native":
            gradients_ready_before = self.evaluator.direct_parameter_gradients_ready
            capture_count_before = (
                self.evaluator.direct_parameter_gradient_capture_count
            )
            evaluation_completed = False
            try:
                evaluation = self._evaluate_native_batch_loss(structures, references)
                evaluation_completed = True
                energies = [float(value) for value in evaluation["energies"]]
                mean_loss = float(evaluation["loss"])
                clipping = self._native_optimizer_step()
            except Exception:
                self._handle_failed_native_training(
                    gradients_ready_before=gradients_ready_before,
                    capture_count_before=capture_count_before,
                    evaluation_completed=evaluation_completed,
                )
                raise
        else:
            if self._parameters_stale:
                self.get_parameters()
            self._sync_optimizer_state_from_native()
            accumulated = {
                name: np.zeros_like(value) for name, value in self._parameters.items()
            }
            energies = []
            losses = []
            for atoms, reference in zip(structures, references, strict=True):
                energy, loss, gradients = self.energy_and_loss_gradients(
                    atoms, reference
                )
                energies.append(energy)
                losses.append(loss)
                for name, gradient in gradients.items():
                    accumulated[name] += gradient
            for name in accumulated:
                accumulated[name] /= len(structures)
            mean_loss = float(np.mean(losses))
            clipping = self._optimizer_step(accumulated)
            self._sync_optimizer_state_to_native()
        batch_size = len(structures)
        diagnostics = {
            "batch_size": batch_size,
            "mean_energy": float(np.mean(energies)),
            "mean_loss": mean_loss,
            **clipping,
            **self._direct_training_diagnostics(),
        }
        self._training_record(
            diagnostics,
            batch_size=batch_size,
            batch_mode=batch_mode,
            elapsed_seconds=time.perf_counter() - started,
        )
        return diagnostics

    def _validated_force_training_inputs(
        self,
        structures,
        reference_forces,
        displacement: float | None,
    ):
        if self.evaluator.streamed_edges_mode != "direct":
            raise ValueError("Direct force training requires streamed_edges='direct'.")
        if self._neighbor_skin <= 0.0 or not np.isfinite(self._neighbor_skin):
            raise ValueError("Direct force training requires a positive neighbor skin.")
        if displacement is None:
            displacement = 2.0e-3 if self._dtype == "float32" else 3.0e-4
        displacement = float(displacement)
        if not np.isfinite(displacement) or displacement <= 0.0:
            raise ValueError("Force-loss displacement must be finite and positive.")
        if 4.0 * displacement > self._neighbor_skin:
            raise ValueError(
                "Force-loss displacement exceeds the fixed-graph safety margin; "
                "require 4*displacement <= neighbor_skin."
            )

        structures = list(structures)
        references = list(reference_forces)
        if not structures or len(structures) != len(references):
            raise ValueError(
                "Force training requires matching, non-empty structures and force "
                "references."
            )
        normalized_references = []
        for index, (atoms, reference) in enumerate(zip(structures, references)):
            if atoms is None or len(atoms) == 0:
                raise ValueError("Force training structures must be non-empty.")
            values = np.asarray(reference, dtype=np.float64)
            if values.shape != (len(atoms), 3):
                raise ValueError(
                    f"Force reference {index} has shape {values.shape}; expected "
                    f"({len(atoms)}, 3)."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Force reference {index} contains non-finite values.")
            normalized_references.append(np.ascontiguousarray(values))
        return structures, normalized_references, displacement

    def _evaluate_native_force_loss(
        self,
        structures,
        references,
        displacement,
        *,
        include_forces: bool,
        _prepared_batch=None,
    ):
        prepared = (
            self._prepare_direct_force_batch(structures)
            if _prepared_batch is None
            else self._validate_prepared_direct_force_batch(_prepared_batch, structures)
        )
        graph = prepared.graph
        result = self.evaluator._compute_prepared_direct_force_loss(
            prepared.generation,
            np.ascontiguousarray(graph.positions.reshape(-1)),
            np.ascontiguousarray(graph.edge_shifts.reshape(-1)),
            np.ascontiguousarray(graph.cells.reshape(-1)),
            np.ascontiguousarray(graph.pbc.reshape(-1)),
            graph.structure_offsets,
            prepared.edge_structures,
            np.ascontiguousarray(
                np.concatenate([values.reshape(-1) for values in references])
            ),
            displacement,
            self._neighbor_skin,
            return_forces=include_forces,
        )
        return result

    @staticmethod
    def _force_training_evaluation(result, structures):
        forces = np.asarray(result["forces"], dtype=np.float64)
        split_forces = []
        offsets = np.asarray(result["structure_offsets"], dtype=np.int64)
        for index, atoms in enumerate(structures):
            start = 3 * offsets[index]
            stop = 3 * offsets[index + 1]
            split_forces.append(
                np.ascontiguousarray(forces[start:stop].reshape((len(atoms), 3)))
            )
        return {
            "schema": "symmetrix.direct.force-training-evaluation",
            "version": 1,
            "objective": "mean-half-squared-force",
            "batch_size": int(result["batch_size"]),
            "num_atoms": int(result["num_nodes"]),
            "num_force_components": int(result["num_force_components"]),
            "num_edges": int(result["num_edges"]),
            "energies": [float(value) for value in result["energies"]],
            "forces": split_forces,
            "loss": float(result["loss"]),
            "force_rmse": float(result["force_rmse"]),
            "force_mae": float(result["force_mae"]),
            "max_abs_residual": float(result["max_abs_residual"]),
            "zero_residual": bool(result["zero_residual"]),
            "displacement": float(result["displacement"]),
            "method": "central-coordinate-mixed-vjp",
            "timing_ms": {
                "base": float(result["base_evaluation_ms"]),
                "negative": float(result["negative_evaluation_ms"]),
                "positive": float(result["positive_evaluation_ms"]),
                "combination": float(result["combination_ms"]),
            },
        }

    def step_force_batch(
        self,
        structures,
        reference_forces,
        *,
        displacement: float | None = None,
    ) -> dict[str, Any]:
        """Fine-tune easy weights against full float64 force labels.

        Gradient formation, clipping, and optimizer updates remain in Kokkos
        storage.  No parameter gradients or per-atom forces are exported.
        """
        structures, references, displacement = self._validated_force_training_inputs(
            structures, reference_forces, displacement
        )
        self._resolve_optimizer(
            optimizer=None,
            learning_rate=None,
            beta1=None,
            beta2=None,
            epsilon=None,
        )
        started = time.perf_counter()
        gradients_ready_before = self.evaluator.direct_parameter_gradients_ready
        capture_count_before = self.evaluator.direct_parameter_gradient_capture_count
        evaluation_completed = False
        try:
            native_result = self._evaluate_native_force_loss(
                structures, references, displacement, include_forces=False
            )
            evaluation_completed = True
            clipping = self._native_optimizer_step()
        except Exception:
            self._handle_failed_native_training(
                gradients_ready_before=gradients_ready_before,
                capture_count_before=capture_count_before,
                evaluation_completed=evaluation_completed,
            )
            raise
        elapsed = time.perf_counter() - started
        num_atoms = int(native_result["num_nodes"])
        diagnostics = {
            "objective": "mean-half-squared-force",
            "batch_mode": "force-native",
            "batch_size": int(native_result["batch_size"]),
            "num_atoms": num_atoms,
            "num_force_components": int(native_result["num_force_components"]),
            "num_edges": int(native_result["num_edges"]),
            "mean_energy": float(np.mean(np.asarray(native_result["energies"]))),
            "loss": float(native_result["loss"]),
            "force_rmse": float(native_result["force_rmse"]),
            "force_mae": float(native_result["force_mae"]),
            "max_abs_residual": float(native_result["max_abs_residual"]),
            "zero_residual": bool(native_result["zero_residual"]),
            "displacement": displacement,
            "force_weight": 1.0,
            "force_timing_ms": {
                "base": float(native_result["base_evaluation_ms"]),
                "negative": float(native_result["negative_evaluation_ms"]),
                "positive": float(native_result["positive_evaluation_ms"]),
                "combination": float(native_result["combination_ms"]),
            },
            **clipping,
            **self._direct_training_diagnostics(),
            "elapsed_seconds": elapsed,
            "elapsed_us_per_atom": 1.0e6 * elapsed / num_atoms,
        }
        self._training_record(
            diagnostics,
            batch_size=int(native_result["batch_size"]),
            batch_mode="force-native",
            elapsed_seconds=elapsed,
        )
        return diagnostics

    def step_energy_force_batch(
        self,
        structures,
        reference_energies,
        reference_forces,
        *,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        energy_normalization: str = "structure",
        displacement: float | None = None,
    ) -> dict[str, Any]:
        """Apply one weighted energy-plus-force optimizer update."""
        energy_weight = float(energy_weight)
        force_weight = float(force_weight)
        if energy_normalization not in {"structure", "per_atom"}:
            raise ValueError("Energy normalization must be 'structure' or 'per_atom'.")
        if not np.isfinite(energy_weight) or not np.isfinite(force_weight):
            raise ValueError("Energy and force weights must be finite.")
        if energy_weight < 0.0 or force_weight < 0.0:
            raise ValueError("Energy and force weights must be non-negative.")
        if energy_weight == 0.0 and force_weight == 0.0:
            raise ValueError("At least one energy or force weight must be positive.")

        structures = list(structures)
        if not structures or any(
            atoms is None or len(atoms) == 0 for atoms in structures
        ):
            raise ValueError("Combined training structures must be non-empty.")
        if force_weight > 0.0:
            structures, forces, displacement = self._validated_force_training_inputs(
                structures, reference_forces, displacement
            )
        else:
            forces = None
            displacement = None
        if energy_weight > 0.0:
            energies = [float(value) for value in reference_energies]
            if len(energies) != len(structures):
                raise ValueError(
                    "Combined training requires one reference energy per structure."
                )
            if any(not np.isfinite(value) for value in energies):
                raise ValueError("Reference energies must be finite.")
        else:
            energies = None
        self._resolve_optimizer(
            optimizer=None,
            learning_rate=None,
            beta1=None,
            beta2=None,
            epsilon=None,
        )

        started = time.perf_counter()
        gradients_ready_before = self.evaluator.direct_parameter_gradients_ready
        capture_count_before = self.evaluator.direct_parameter_gradient_capture_count
        evaluation_completed = False
        try:
            prepared = self._prepare_direct_force_batch(structures)
            force_result = None
            energy_result = None
            if force_weight > 0.0:
                force_result = self._evaluate_native_force_loss(
                    structures,
                    forces,
                    displacement,
                    include_forces=False,
                    _prepared_batch=prepared,
                )
                evaluation_completed = True
                if energy_weight > 0.0:
                    self.evaluator._stash_direct_parameter_gradients(force_weight)
                else:
                    self.evaluator._scale_direct_parameter_gradients(force_weight)
            if energy_weight > 0.0:
                energy_residual_scales = None
                if energy_normalization == "per_atom":
                    counts = np.asarray(
                        [len(atoms) for atoms in structures], dtype=np.float64
                    )
                    energy_residual_scales = 1.0 / counts
                energy_result = self._evaluate_native_batch_loss_from_prepared(
                    prepared,
                    energies,
                    energy_residual_scales=energy_residual_scales,
                )
                evaluation_completed = True
                if force_weight > 0.0:
                    self.evaluator._combine_stashed_direct_parameter_gradients(
                        energy_weight
                    )
                else:
                    self.evaluator._scale_direct_parameter_gradients(energy_weight)

            clipping = self._native_optimizer_step()
        except Exception:
            self._handle_failed_native_training(
                gradients_ready_before=gradients_ready_before,
                capture_count_before=capture_count_before,
                evaluation_completed=evaluation_completed,
            )
            raise
        elapsed = time.perf_counter() - started
        result = force_result if force_result is not None else energy_result
        num_atoms = int(result["num_nodes"])
        energy_loss = (
            float(energy_result["loss"]) if energy_result is not None else None
        )
        force_loss = float(force_result["loss"]) if force_result is not None else None
        total_loss = energy_weight * (
            energy_loss if energy_loss is not None else 0.0
        ) + force_weight * (force_loss if force_loss is not None else 0.0)
        evaluated_energies = (
            energy_result["energies"]
            if energy_result is not None
            else force_result["energies"]
        )
        diagnostics = {
            "objective": "weighted-energy-force",
            "batch_mode": "energy-force-native",
            "batch_size": int(result["batch_size"]),
            "num_atoms": num_atoms,
            "num_force_components": (
                int(force_result["num_force_components"])
                if force_result is not None
                else 0
            ),
            "num_edges": int(result["num_edges"]),
            "mean_energy": float(np.mean(np.asarray(evaluated_energies))),
            "loss": total_loss,
            "energy_loss": energy_loss,
            "energy_normalization": energy_normalization,
            "force_loss": force_loss,
            "force_rmse": (
                float(force_result["force_rmse"]) if force_result is not None else None
            ),
            "force_mae": (
                float(force_result["force_mae"]) if force_result is not None else None
            ),
            "max_abs_residual": (
                float(force_result["max_abs_residual"])
                if force_result is not None
                else None
            ),
            "zero_residual": (
                bool(force_result["zero_residual"])
                if force_result is not None
                else None
            ),
            "displacement": displacement,
            "energy_weight": energy_weight,
            "force_weight": force_weight,
            "force_timing_ms": (
                {
                    "base": float(force_result["base_evaluation_ms"]),
                    "negative": float(force_result["negative_evaluation_ms"]),
                    "positive": float(force_result["positive_evaluation_ms"]),
                    "combination": float(force_result["combination_ms"]),
                }
                if force_result is not None
                else None
            ),
            **clipping,
            **self._direct_training_diagnostics(),
            "elapsed_seconds": elapsed,
            "elapsed_us_per_atom": 1.0e6 * elapsed / num_atoms,
        }
        self._training_record(
            diagnostics,
            batch_size=int(result["batch_size"]),
            batch_mode="energy-force-native",
            elapsed_seconds=elapsed,
        )
        return diagnostics

    def force_and_loss_gradients(
        self,
        structures,
        reference_forces,
        *,
        displacement: float | None = None,
        _prepared_batch=None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Explicitly synchronize and inspect the native force-loss gradient.

        This diagnostic boundary exports base forces and every easy-weight
        gradient to Python.  ``step_force_batch()`` does not call it.
        """
        structures, references, displacement = self._validated_force_training_inputs(
            structures, reference_forces, displacement
        )
        native_result = self._evaluate_native_force_loss(
            structures,
            references,
            displacement,
            include_forces=True,
            _prepared_batch=_prepared_batch,
        )
        evaluation = self._force_training_evaluation(native_result, structures)
        raw = self.evaluator.direct_parameter_gradients()
        if not raw["enabled"]:
            raise RuntimeError("Direct parameter gradients were disabled.")
        if not raw["ready"]:
            raise RuntimeError("Native force-loss gradients are not ready.")
        if raw["objective"] != "mean-half-squared-force":
            raise RuntimeError("Native force-loss gradients have the wrong objective.")
        if not np.isclose(
            float(raw["finite_difference_displacement"]),
            displacement,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError("Native force-loss displacement provenance is wrong.")
        gradients = {
            group["name"]: np.asarray(group["values"], dtype=np.float64).copy()
            for group in raw["groups"]
        }
        self._validate_gradients(gradients)
        return evaluation, gradients

    def _model_with_parameters(
        self, parameters: Mapping[str, np.ndarray]
    ) -> dict[str, Any]:
        model = copy.deepcopy(self.model_data)

        m0 = model["M0_weights"]
        for lm in range((model["L_max"] + 1) ** 2):
            values = parameters[f"M0_weights.LM{lm}"]
            index = 0
            for atomic_type in range(model["num_elements"]):
                for channel in range(model["num_channels"]):
                    count = len(m0[str(atomic_type)][str(lm)][str(channel)])
                    m0[str(atomic_type)][str(lm)][str(channel)] = values[
                        index : index + count
                    ].tolist()
                    index += count
            if index != values.size:
                raise RuntimeError("M0 checkpoint layout is inconsistent.")

        h1 = np.empty(
            (model["L_max"] + 1, model["num_channels"], model["num_channels"]),
            dtype=np.float64,
        )
        for angular in range(model["L_max"] + 1):
            h1[angular] = parameters[f"H1_weights.l{angular}"].reshape(
                model["num_channels"], model["num_channels"]
            )
        model["H1_weights"] = h1.ravel().tolist()

        model["A1_weights"] = [
            parameters[f"A1_weights.l{l}"].tolist() for l in range(model["l_max"] + 1)
        ]

        m1 = model["M1_weights"]
        for atomic_type in range(model["num_elements"]):
            for channel in range(model["num_channels"]):
                count = len(m1[str(atomic_type)][str(channel)])
                start = (atomic_type * model["num_channels"] + channel) * len(
                    model["M1_monomials"]
                )
                m1[str(atomic_type)][str(channel)] = parameters["M1_weights"][
                    start : start + count
                ].tolist()

        model["H2_weights_for_H1"] = [
            parameters["H2_weights_for_H1"][
                atomic_type * model["num_channels"] ** 2 : (atomic_type + 1)
                * model["num_channels"] ** 2
            ].tolist()
            for atomic_type in range(model["num_elements"])
        ]
        model["H2_weights_for_M1"] = (
            parameters["H2_weights_for_M1"]
            .reshape(model["num_channels"], model["num_channels"])
            .ravel()
            .tolist()
        )
        model["readout_1_weights"] = parameters["readout_1_weights"].tolist()
        model["readout_2_weights_1"] = (
            parameters["readout_2.weights.0"].ravel().tolist()
        )
        model["readout_2_weights_2"] = parameters["readout_2.weights.1"].tolist()
        prediction_heads = model.get("prediction_heads")
        if prediction_heads is not None:
            if self._head is None:
                raise RuntimeError(
                    "Multi-head training has no selected prediction head."
                )
            payload = prediction_heads["parameters"][self._head]
            for name in (
                "readout_1_weights",
                "readout_2_weights_1",
                "readout_2_weights_2",
            ):
                payload[name] = copy.deepcopy(model[name])
        return model

    def _model_signature(self) -> str:
        zero_parameters = {
            name: np.zeros_like(values) for name, values in self._parameters.items()
        }
        canonical = json.dumps(
            self._model_with_parameters(zero_parameters),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def save_model(self, path: str | Path) -> None:
        """Write compact JSON with only the easy-subset values changed."""
        model = self._model_with_parameters(self.get_parameters())

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(model, separators=(",", ":")), encoding="utf-8"
        )

    def save_training_state(self, path: str | Path) -> None:
        """Save model parameters and complete training state to an NPZ file."""
        if self._optimizer_config is None:
            raise RuntimeError("No optimizer state is available to checkpoint.")
        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("Training-state checkpoints must use the .npz suffix.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        parameter_arrays = {}
        parameters = self.get_parameters()
        self._sync_optimizer_state_from_native()
        native_state = self._optimizer_state
        for index, (name, values) in enumerate(parameters.items()):
            key = f"parameter_{index}"
            arrays[key] = np.asarray(values, dtype=np.float64)
            parameter_arrays[name] = key
        optimizer_arrays = []
        for state_name, state in sorted(native_state.items()):
            for name in parameters:
                if name not in state:
                    continue
                key = f"optimizer_{len(optimizer_arrays)}"
                arrays[key] = np.asarray(state[name], dtype=np.float64)
                optimizer_arrays.append(
                    {"state": state_name, "parameter": name, "array": key}
                )
        metadata = {
            "schema_version": _TRAINING_STATE_SCHEMA_VERSION,
            "model_signature": self._model_signature(),
            "dtype": self._dtype,
            "neighbor_skin": self._neighbor_skin,
            "streamed_edges": "direct",
            "parameter_arrays": parameter_arrays,
            "optimizer_config": self._optimizer_config,
            "optimizer_source": self._optimizer_source,
            "optimizer_step_count": self._optimizer_step_count,
            "parameter_groups": self._parameter_groups,
            "optimizer_arrays": optimizer_arrays,
            "scheduler_config": self._scheduler_config,
            "scheduler_state": self._scheduler_state,
            "global_step": self._global_step,
            "epoch": self._epoch,
            "best_validation": self._best_validation,
            "history_records": self._history_records,
        }
        arrays["metadata"] = np.asarray(
            json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        )
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    def load_training_state(self, path: str | Path) -> None:
        """Restore parameters and optimizer state after strict identity checks."""
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            if "metadata" not in archive:
                raise ValueError("Training-state checkpoint has no metadata.")
            try:
                metadata = json.loads(str(archive["metadata"].item()))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Training-state checkpoint metadata is invalid."
                ) from error
            if metadata.get("schema_version") != _TRAINING_STATE_SCHEMA_VERSION:
                raise ValueError("Unsupported training-state checkpoint schema.")
            if metadata.get("model_signature") != self._model_signature():
                raise ValueError(
                    "Training-state checkpoint does not match this model layout "
                    "and frozen-input signature."
                )
            if metadata.get("dtype") != self._dtype:
                raise ValueError("Training-state checkpoint evaluator dtype differs.")
            if metadata.get("streamed_edges") != "direct":
                raise ValueError("Training-state checkpoint graph mode differs.")
            if float(metadata.get("neighbor_skin")) != self._neighbor_skin:
                raise ValueError("Training-state checkpoint neighbor skin differs.")

            parameter_arrays = metadata.get("parameter_arrays", {})
            if set(parameter_arrays) != set(self._parameters):
                raise ValueError("Training-state checkpoint parameters differ.")
            restored_parameters = {}
            for name, key in parameter_arrays.items():
                if key not in archive:
                    raise ValueError(f"Training-state array {key!r} is missing.")
                values = np.asarray(archive[key], dtype=np.float64)
                if values.shape != self._parameters[name].shape:
                    raise ValueError(
                        f"Training-state parameter {name!r} has the wrong shape."
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Training-state parameter {name!r} is non-finite."
                    )
                restored_parameters[name] = values.copy()

            optimizer_config = metadata.get("optimizer_config")
            if not isinstance(optimizer_config, dict):
                raise ValueError(  # noqa: TRY004
                    "Training-state optimizer configuration is invalid."
                )
            self.configure_optimizer(
                optimizer=optimizer_config["optimizer"],
                learning_rate=optimizer_config["learning_rate"],
                beta1=optimizer_config["beta1"],
                beta2=optimizer_config["beta2"],
                epsilon=optimizer_config["epsilon"],
                amsgrad=optimizer_config["amsgrad"],
                momentum=optimizer_config["momentum"],
                nesterov=optimizer_config["nesterov"],
                weight_decay=optimizer_config["weight_decay"],
                gradient_clip_norm=optimizer_config["gradient_clip_norm"],
                parameter_groups=optimizer_config["parameter_groups"],
            )
            groups = metadata.get("parameter_groups")
            if not isinstance(groups, list) or len(groups) != len(
                self._parameter_groups
            ):
                raise ValueError("Training-state parameter groups are invalid.")
            for restored, current in zip(groups, self._parameter_groups, strict=True):
                if restored.get("names") != current["names"]:
                    raise ValueError("Training-state parameter-group names differ.")
                current["learning_rate"] = self._finite_positive(
                    "checkpoint learning_rate", restored.get("learning_rate")
                )
                current["weight_decay"] = self._finite_non_negative(
                    "checkpoint weight_decay", restored.get("weight_decay")
                )

            optimizer_step_count = metadata.get("optimizer_step_count", 0)
            if (
                isinstance(optimizer_step_count, bool)
                or not isinstance(optimizer_step_count, int)
                or optimizer_step_count < 0
            ):
                raise ValueError("Training-state optimizer step is invalid.")
            expected_state_names = set()
            if optimizer_config["optimizer"] == "sgd":
                if optimizer_config["momentum"]:
                    expected_state_names.add("momentum")
            else:
                expected_state_names.update(("first_moment", "second_moment"))
                if optimizer_config["amsgrad"]:
                    expected_state_names.add("max_second_moment")
            restored_state: dict[str, dict[str, np.ndarray]] = {}
            for entry in metadata.get("optimizer_arrays", []):
                state_name = entry.get("state")
                name = entry.get("parameter")
                key = entry.get("array")
                if name not in self._parameters or key not in archive:
                    raise ValueError("Training-state optimizer array is invalid.")
                values = np.asarray(archive[key], dtype=np.float64)
                if values.shape != self._parameters[name].shape:
                    raise ValueError("Training-state optimizer shape differs.")
                if not np.all(np.isfinite(values)):
                    raise ValueError("Training-state optimizer array is non-finite.")
                if name in restored_state.get(state_name, {}):
                    raise ValueError("Training-state optimizer array is duplicated.")
                restored_state.setdefault(state_name, {})[name] = values.copy()
            if set(restored_state) != expected_state_names or any(
                set(state) != set(self._parameters) for state in restored_state.values()
            ):
                raise ValueError("Training-state optimizer buffers are incomplete.")

            scheduler_config = metadata.get("scheduler_config")
            if scheduler_config is not None:
                scheduler_name = scheduler_config["scheduler"]
                scheduler_options = {
                    name: value
                    for name, value in scheduler_config.items()
                    if name != "scheduler"
                }
                self.configure_lr_scheduler(scheduler_name, **scheduler_options)
            # A restored checkpoint is authoritative persistent state even when
            # it originated from legacy per-call optimizer arguments.
            self._optimizer_source = "configured"
            self._optimizer_state = restored_state
            self._optimizer_step_count = optimizer_step_count
            self._scheduler_config = copy.deepcopy(scheduler_config)
            self._scheduler_state = copy.deepcopy(metadata.get("scheduler_state", {}))
            self._global_step = int(metadata.get("global_step", 0))
            self._epoch = int(metadata.get("epoch", 0))
            self._best_validation = copy.deepcopy(metadata.get("best_validation"))
            history = metadata.get("history_records", [])
            if not isinstance(history, list):
                raise ValueError("Training-state history is invalid.")  # noqa: TRY004
            self._history_records.clear()
            self._history_records.extend(copy.deepcopy(history))
        self.set_parameters(restored_parameters)
        self.evaluator.set_direct_optimizer_learning_rates(list(self.learning_rates))
        self.evaluator.set_direct_optimizer_state(
            optimizer_step_count,
            {
                state_name: {
                    name: np.asarray(values, dtype=np.float64).tolist()
                    for name, values in state.items()
                }
                for state_name, state in restored_state.items()
            },
        )
