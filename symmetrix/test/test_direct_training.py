import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from compact_r1_model import foundation_r1_test_model
from symmetrix import DirectMACEEnergyTrainer


CONTRACT = (
    Path(__file__).parent / "data" / "execution_contracts" / "jit_r1_contract.json"
)


def _new_trainer(model_data, tmp_path, name):
    model_path = tmp_path / f"{name}.json"
    model_path.write_text(json.dumps(model_data, separators=(",", ":")))
    return DirectMACEEnergyTrainer(
        model_path,
        dtype="float64",
        jit_cache=tmp_path / f"{name}-jit",
    )


def _release_trainer(trainer):
    trainer.evaluator = None
    trainer.calculator = None


@pytest.fixture()
def training_fixture(tmp_path):
    contract = json.loads(CONTRACT.read_text())
    model = foundation_r1_test_model(contract)
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))
    trainer = DirectMACEEnergyTrainer(
        model_path,
        dtype="float64",
        jit_cache=tmp_path / "jit",
    )
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    yield trainer, atoms
    atoms.calc = None
    trainer.evaluator = None
    trainer.calculator = None
    del trainer, atoms


def test_direct_parameter_groups_are_finite_and_stable(training_fixture):
    trainer, _ = training_fixture
    parameters = trainer.get_parameters()

    assert parameters
    assert trainer.parameter_names == tuple(parameters)
    assert all(np.isfinite(values).all() for values in parameters.values())
    assert all(values.ndim == 1 for values in parameters.values())
    assert trainer.evaluator.direct_parameter_gradients_enabled


def test_native_batch_matches_sequential_energy_loss(training_fixture):
    trainer, atoms = training_fixture
    second = atoms.copy()
    second.positions[1, 2] = 1.2
    structures = [atoms, second]
    references = [0.0, 0.0]

    sequential = [
        trainer.energy_and_loss_gradients(item, reference)[:2]
        for item, reference in zip(structures, references, strict=True)
    ]
    native = trainer._evaluate_native_batch_loss(structures, references)

    np.testing.assert_allclose(
        np.asarray(native["energies"], dtype=float),
        np.asarray([value[0] for value in sequential]),
        rtol=2.0e-10,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        float(native["loss"]),
        np.mean([value[1] for value in sequential]),
        rtol=2.0e-10,
        atol=2.0e-14,
    )


def test_sgd_step_matches_energy_gradient(training_fixture):
    trainer, atoms = training_fixture
    trainer.configure_optimizer(optimizer="sgd", learning_rate=1.0e-4)
    energy, _, gradients = trainer.energy_and_loss_gradients(atoms, 0.0)
    before = trainer.get_parameters()
    trainer.train_step(atoms, 0.0)
    after = trainer.get_parameters()

    for name in before:
        expected = before[name] - 1.0e-4 * gradients[name]
        np.testing.assert_allclose(after[name], expected, rtol=2.0e-10, atol=2.0e-12)
    assert np.isfinite(energy)
    assert trainer.evaluator.direct_optimizer_step_count == 1


def test_force_gradient_diagnostic_matches_zero_residual(training_fixture):
    trainer, atoms = training_fixture
    atoms.calc = trainer.calculator
    reference_forces = atoms.get_forces().copy()

    evaluation, gradients = trainer.force_and_loss_gradients(
        [atoms], [reference_forces]
    )

    assert evaluation["zero_residual"]
    assert evaluation["loss"] == 0.0
    assert gradients
    assert all(np.all(values == 0.0) for values in gradients.values())


def test_force_gradient_matches_parameter_finite_difference(training_fixture):
    trainer, atoms = training_fixture
    atoms.calc = trainer.calculator
    reference_forces = atoms.get_forces().copy()
    reference_forces[0, 0] += 0.25
    displacement = 3.0e-4

    evaluation, gradients = trainer.force_and_loss_gradients(
        [atoms], [reference_forces], displacement=displacement
    )
    name = next(iter(gradients))
    parameters = trainer.get_parameters()
    index = int(np.flatnonzero(np.abs(gradients[name]) > 1.0e-12)[0])
    step = 1.0e-5

    plus = {key: value.copy() for key, value in parameters.items()}
    plus[name][index] += step
    trainer.set_parameters(plus)
    loss_plus = trainer.force_and_loss_gradients(
        [atoms], [reference_forces], displacement=displacement
    )[0]["loss"]

    minus = {key: value.copy() for key, value in parameters.items()}
    minus[name][index] -= step
    trainer.set_parameters(minus)
    loss_minus = trainer.force_and_loss_gradients(
        [atoms], [reference_forces], displacement=displacement
    )[0]["loss"]
    trainer.set_parameters(parameters)

    numerical = (loss_plus - loss_minus) / (2.0 * step)
    assert evaluation["loss"] > 0.0
    np.testing.assert_allclose(
        gradients[name][index], numerical, rtol=3.0e-4, atol=3.0e-7
    )


def test_adamw_step_matches_clipping_and_decoupled_decay(training_fixture):
    trainer, atoms = training_fixture
    trainer.configure_optimizer(
        optimizer="adamw",
        learning_rate=2.0e-4,
        beta1=0.8,
        beta2=0.9,
        epsilon=1.0e-6,
        weight_decay=0.15,
        gradient_clip_norm=1.0e-3,
    )
    _, _, gradients = trainer.energy_and_loss_gradients(atoms, 1.0)
    before = trainer.get_parameters()
    norm = np.sqrt(sum(np.dot(value, value) for value in gradients.values()))
    coefficient = min(1.0, 1.0e-3 / norm)
    diagnostics = trainer.train_step(atoms, 1.0)

    assert diagnostics["clipped"]
    np.testing.assert_allclose(diagnostics["clip_coefficient"], coefficient)
    for name, gradient in gradients.items():
        clipped = gradient * coefficient
        first = 0.2 * clipped
        second = 0.1 * clipped * clipped
        expected = before[name] * (1.0 - 2.0e-4 * 0.15)
        expected -= 2.0e-4 * (first / 0.2) / (np.sqrt(second / 0.1) + 1.0e-6)
        np.testing.assert_allclose(
            trainer.get_parameters()[name], expected, rtol=2.0e-8, atol=2.0e-10
        )


def test_sgd_nesterov_parameter_groups_match_reference_update(training_fixture):
    trainer, atoms = training_fixture
    names = trainer.parameter_names
    split = len(names) // 2
    first_names = list(names[:split])
    second_names = list(names[split:])
    trainer.configure_optimizer(
        optimizer="sgd",
        momentum=0.6,
        nesterov=True,
        parameter_groups=[
            {"names": first_names, "learning_rate": 1.0e-4},
            {"names": second_names, "learning_rate": 3.0e-4, "weight_decay": 0.2},
        ],
    )
    _, _, gradients = trainer.energy_and_loss_gradients(atoms, 0.0)
    before = trainer.get_parameters()
    trainer.train_step(atoms, 0.0)

    for name, gradient in gradients.items():
        group_learning_rate = 1.0e-4 if name in first_names else 3.0e-4
        weight_decay = 0.0 if name in first_names else 0.2
        decayed = gradient + weight_decay * before[name]
        expected = before[name] - group_learning_rate * (decayed + 0.6 * decayed)
        np.testing.assert_allclose(
            trainer.get_parameters()[name], expected, rtol=2.0e-8, atol=2.0e-10
        )


def test_prepared_batch_rejects_changed_geometry(training_fixture):
    trainer, atoms = training_fixture
    prepared = trainer._prepare_direct_force_batch([atoms])
    changed = atoms.copy()
    changed.positions[1, 2] += 0.1

    with pytest.raises(ValueError, match="geometry does not match"):
        trainer.force_and_loss_gradients(
            [changed], [np.zeros((len(changed), 3))], _prepared_batch=prepared
        )


def test_public_native_and_sequential_batch_steps_match(training_fixture, tmp_path):
    trainer, atoms = training_fixture
    native = _new_trainer(trainer.model_data, tmp_path, "native")
    sequential = _new_trainer(trainer.model_data, tmp_path, "sequential")
    try:
        second = atoms.copy()
        second.positions[1, 2] = 1.2
        structures = [atoms.copy(), second]
        references = [1.0, -0.5]
        for candidate in (native, sequential):
            candidate.configure_optimizer(optimizer="adam", learning_rate=1.0e-4)

        native_result = native.step_batch(structures, references, batch_mode="native")
        sequential_result = sequential.step_batch(
            structures, references, batch_mode="sequential"
        )

        np.testing.assert_allclose(
            native_result["mean_loss"], sequential_result["mean_loss"], rtol=1e-10
        )
        assert native.evaluator.direct_optimizer_step_count == 1
        assert sequential.evaluator.direct_optimizer_step_count == 1
        for name, values in native.get_parameters().items():
            np.testing.assert_allclose(
                values, sequential.get_parameters()[name], rtol=2e-8, atol=2e-10
            )
    finally:
        _release_trainer(native)
        _release_trainer(sequential)


def test_public_force_and_joint_training_steps_update_parameters(training_fixture):
    trainer, atoms = training_fixture
    trainer.configure_optimizer(optimizer="sgd", learning_rate=1.0e-5)
    atoms.calc = trainer.calculator
    reference_forces = atoms.get_forces().copy()
    reference_forces[0, 0] += 0.2
    before = trainer.get_parameters()

    force_result = trainer.step_force_batch([atoms], [reference_forces])
    assert force_result["loss"] > 0.0
    assert force_result["objective"] == "mean-half-squared-force"
    assert trainer.evaluator.direct_optimizer_step_count == 1
    assert any(
        not np.array_equal(before[name], values)
        for name, values in trainer.get_parameters().items()
    )

    energy = float(atoms.get_potential_energy())
    joint_result = trainer.step_energy_force_batch(
        [atoms], [energy + 0.1], [reference_forces], energy_weight=0.5, force_weight=0.5
    )
    assert joint_result["objective"] == "weighted-energy-force"
    assert joint_result["energy_loss"] is not None
    assert joint_result["force_loss"] is not None
    assert trainer.evaluator.direct_optimizer_step_count == 2


def test_force_matches_coordinate_finite_difference_of_energy(training_fixture):
    trainer, atoms = training_fixture
    atoms.calc = trainer.calculator
    coordinate_step = 1.0e-5
    forces = atoms.get_forces().copy()
    numerical = np.zeros_like(forces)
    for atom_index in range(len(atoms)):
        for component in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions[atom_index, component] += coordinate_step
            minus.positions[atom_index, component] -= coordinate_step
            trainer.calculator.reset()
            energy_plus = trainer.calculator.get_potential_energy(plus)
            trainer.calculator.reset()
            energy_minus = trainer.calculator.get_potential_energy(minus)
            numerical[atom_index, component] = -(energy_plus - energy_minus) / (
                2.0 * coordinate_step
            )

    np.testing.assert_allclose(forces, numerical, rtol=2.0e-4, atol=2.0e-5)


def test_learning_rate_schedulers_update_groups_and_epoch(training_fixture):
    trainer, _ = training_fixture
    trainer.configure_optimizer(optimizer="sgd", learning_rate=2.0e-4)
    trainer.configure_lr_scheduler("exponential", gamma=0.5)

    assert trainer.scheduler_step() == (1.0e-4,)
    assert trainer.epoch == 1

    trainer.configure_lr_scheduler(
        "reduce_on_plateau", patience=1, factor=0.25, min_lr=1.0e-5
    )
    assert trainer.scheduler_step(metric=1.0) == (1.0e-4,)
    assert trainer.scheduler_step(metric=1.0) == (1.0e-4,)
    assert trainer.scheduler_step(metric=1.0) == (2.5e-5,)
    assert trainer.epoch == 4


def test_checkpoint_resume_preserves_next_energy_update(training_fixture, tmp_path):
    trainer, atoms = training_fixture
    trainer.configure_optimizer(optimizer="adam", learning_rate=1.0e-4)
    trainer.train_step(atoms, 0.0)
    checkpoint = tmp_path / "training.npz"
    trainer.save_training_state(checkpoint)

    resumed_model = tmp_path / "resumed-model.json"
    resumed_model.write_text(json.dumps(trainer.model_data, separators=(",", ":")))
    resumed = DirectMACEEnergyTrainer(
        resumed_model,
        dtype="float64",
        jit_cache=tmp_path / "resumed-jit",
    )
    resumed_atoms = atoms.copy()
    resumed.load_training_state(checkpoint)
    trainer.train_step(atoms, 0.0)
    resumed.train_step(resumed_atoms, 0.0)

    left = trainer.get_parameters()
    right = resumed.get_parameters()
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_allclose(left[name], right[name], rtol=2.0e-10, atol=2.0e-12)
    resumed.evaluator = None
    resumed.calculator = None
    del resumed, resumed_atoms, trainer, atoms


def test_amsgrad_and_scheduler_state_survive_checkpoint(training_fixture, tmp_path):
    trainer, atoms = training_fixture
    trainer.configure_optimizer(optimizer="adam", learning_rate=1.0e-4, amsgrad=True)
    trainer.configure_lr_scheduler("exponential", gamma=0.5)
    trainer.train_step(atoms, 0.0)
    trainer.scheduler_step()
    checkpoint = tmp_path / "amsgrad-training.npz"
    trainer.save_training_state(checkpoint)

    resumed_model = tmp_path / "amsgrad-resumed-model.json"
    resumed_model.write_text(json.dumps(trainer.model_data, separators=(",", ":")))
    resumed = DirectMACEEnergyTrainer(
        resumed_model,
        dtype="float64",
        jit_cache=tmp_path / "amsgrad-resumed-jit",
    )
    try:
        resumed.load_training_state(checkpoint)
        assert resumed.optimizer_config["amsgrad"]
        assert resumed.learning_rates == (5.0e-5,)
        assert resumed.epoch == 1
        assert "max_second_moment" in resumed.evaluator.direct_optimizer_state()
        assert resumed.scheduler_step() == (2.5e-5,)
    finally:
        _release_trainer(resumed)


def test_reporting_and_validation_history_are_exportable(training_fixture, tmp_path):
    trainer, atoms = training_fixture
    trainer.configure_optimizer(optimizer="sgd", learning_rate=1.0e-4)
    history_path = tmp_path / "history.jsonl"
    export_path = tmp_path / "history-export.jsonl"
    trainer.enable_reporting(jsonl_path=history_path)

    trainer.train_step(atoms, 0.0)
    validation = trainer.record_validation(metrics={"loss": 0.25}, count=1)
    exported = trainer.export_history_jsonl(export_path)

    assert validation["type"] == "validation"
    assert trainer.global_step == 1
    assert len(trainer.history) == 2
    assert exported == 2
    assert len(history_path.read_text().splitlines()) == 2
    assert len(export_path.read_text().splitlines()) == 2
