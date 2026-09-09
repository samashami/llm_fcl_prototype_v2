import copy
import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paper.plots_paper.lmss_geo.derive_norm_matched_schedule import derive_schedule
from src.fl import Client
from src.instrumentation.shrinkage import load_shrinkage_schedule
from src.instrumentation.subspace import (
    LayerTarget,
    SubspaceInstrumentation,
    aggregate_update_energy_records,
    realized_update_energy,
    scalar_shrink_rows,
    soft_project_rows,
)


class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.fc(x)


class NormMatchedShrinkageChecks(unittest.TestCase):
    def test_energy_accounting_identities_and_ratio_of_sums(self):
        raw = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
        phi = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        applied = soft_project_rows(raw, phi, 0.5)
        values = realized_update_energy(raw, applied)

        self.assertAlmostEqual(values["raw_update_energy"], 25.0)
        self.assertAlmostEqual(values["projected_update_energy"], 18.25)
        self.assertAlmostEqual(values["removed_displacement_energy"], 2.25)
        self.assertAlmostEqual(values["removed_energy"], 6.75)
        self.assertAlmostEqual(
            values["retained_energy_fraction"], 18.25 / 25.0
        )

        first = {"round": 1, **values}
        second_values = realized_update_energy(raw * 2.0, applied * 2.0)
        summary = aggregate_update_energy_records(
            [first, {"round": 1, **second_values}]
        )[0]
        self.assertAlmostEqual(
            summary["retained_energy_fraction"],
            (18.25 + 73.0) / (25.0 + 100.0),
        )

    def test_scalar_shrink_preserves_direction_and_scales_norm_exactly(self):
        matrix = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float64)
        shrunk = scalar_shrink_rows(matrix, 0.75)
        self.assertTrue(torch.equal(shrunk, matrix * 0.75))
        self.assertAlmostEqual(
            torch.linalg.vector_norm(shrunk).item(),
            0.75 * torch.linalg.vector_norm(matrix).item(),
        )
        self.assertTrue(torch.allclose(shrunk / 0.75, matrix, atol=0.0, rtol=0.0))

    def test_shrink_parameter_update_never_reads_phi(self):
        model = TinyLinear()
        instrumentation = SubspaceInstrumentation(
            [model],
            targets=(LayerTarget("fc", "fc"),),
            samples_per_batch=1,
            samples_per_phase=1,
        )
        monitor = instrumentation.monitors[0]

        class NoPhiBank:
            def basis(self, *args, **kwargs):
                raise AssertionError("shrinkage must not read Phi")

        monitor.bank = NoPhiBank()
        before = monitor.snapshot_target_weights()
        with torch.no_grad():
            model.fc.weight.add_(0.2)
        records = monitor.shrink_parameter_updates(before, {"fc": 0.5})
        self.assertEqual(len(records), 1)
        self.assertTrue(
            torch.allclose(model.fc.weight, before["fc"] + 0.1, atol=1e-7, rtol=0.0)
        )
        instrumentation.close()

    def test_round_zero_factor_one_leaves_realized_adam_update_unchanged(self):
        x = torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]])
        y = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)
        torch.manual_seed(17)
        initial = copy.deepcopy(TinyLinear().state_dict())

        def run(shrinkage):
            model = TinyLinear()
            model.load_state_dict(copy.deepcopy(initial))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            instrumentation = None
            client = Client(0, model, optimizer, loader, replay=None)
            if shrinkage:
                instrumentation = SubspaceInstrumentation(
                    [model],
                    targets=(LayerTarget("fc", "fc"),),
                    samples_per_batch=1,
                    samples_per_phase=1,
                )
                monitor = instrumentation.monitors[0]
                client.gradient_monitor = monitor
                monitor.snapshot_target_weights = Mock(
                    wraps=monitor.snapshot_target_weights
                )
                monitor.shrink_parameter_updates = Mock(
                    wraps=monitor.shrink_parameter_updates
                )
                client.train_one_epoch(
                    replay_ratio=0.0,
                    log_interval=99,
                    update_control="shrinkage",
                    shrinkage_factors={"fc": 1.0},
                    round_id=0,
                )
            else:
                client.train_one_epoch(replay_ratio=0.0, log_interval=99)
            state = copy.deepcopy(model.state_dict())
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            monitor_calls = None
            if instrumentation is not None:
                monitor_calls = (
                    instrumentation.monitors[0].snapshot_target_weights.call_count,
                    instrumentation.monitors[0].shrink_parameter_updates.call_count,
                )
                instrumentation.close()
            return state, optimizer_state, client.update_energy_rows, monitor_calls

        baseline, baseline_optimizer, _, _ = run(False)
        controlled, controlled_optimizer, rows, monitor_calls = run(True)
        self.assertTrue(torch.equal(baseline["fc.weight"], controlled["fc.weight"]))
        self.assertEqual(baseline_optimizer["param_groups"], controlled_optimizer["param_groups"])
        for baseline_state, controlled_state in zip(
            baseline_optimizer["state"].values(),
            controlled_optimizer["state"].values(),
        ):
            self.assertEqual(baseline_state.keys(), controlled_state.keys())
            for key in baseline_state:
                self.assertTrue(torch.equal(baseline_state[key], controlled_state[key]))
        self.assertEqual(rows, [])
        self.assertEqual(monitor_calls, (0, 0))

    def test_online_shrinkage_matches_counterfactual_projection_energy_and_direction(self):
        model = TinyLinear().to(dtype=torch.float64)
        instrumentation = SubspaceInstrumentation(
            [model],
            targets=(LayerTarget("fc", "fc"),),
            samples_per_batch=1,
            samples_per_phase=1,
        )
        monitor = instrumentation.monitors[0]
        basis = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
        monitor.bank.bases["fc"] = basis
        before = monitor.snapshot_target_weights()["fc"]

        raw = torch.tensor(
            [[3.0, -4.0, 1.0], [2.0, 5.0, -1.0]],
            dtype=torch.float64,
        )
        projected = soft_project_rows(raw, basis, 0.5)
        expected_factor = math.sqrt(
            float(projected.square().sum().item()) / max(float(raw.square().sum().item()), 1e-12)
        )

        with torch.no_grad():
            model.fc.weight.copy_(before + raw)
        records = monitor.online_shrink_parameter_updates(
            {"fc": before},
            projection_lambda=0.5,
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        applied = model.fc.weight.detach() - before
        self.assertAlmostEqual(
            record["counterfactual_projected_energy"],
            float(projected.square().sum().item()),
            places=12,
        )
        self.assertAlmostEqual(
            record["applied_control_energy"],
            float(applied.square().sum().item()),
            places=12,
        )
        self.assertAlmostEqual(
            record["retained_energy_fraction"],
            record["counterfactual_projected_energy"] / max(record["raw_update_energy"], 1e-12),
            places=12,
        )
        self.assertAlmostEqual(record["shrinkage_factor"], expected_factor, places=12)
        self.assertAlmostEqual(
            record["applied_control_energy"],
            record["counterfactual_projected_energy"],
            places=12,
        )
        self.assertIs(record["online_shrinkage_guard_passed"], True)
        self.assertTrue(torch.allclose(applied, expected_factor * raw, atol=1e-12, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                applied,
                ((applied * raw).sum() / max(raw.square().sum(), 1e-12)) * raw,
                atol=1e-12,
                rtol=0.0,
            )
        )
        instrumentation.close()

    def test_online_shrinkage_rejects_non_contractive_malformed_phi(self):
        model = TinyLinear().to(dtype=torch.float64)
        instrumentation = SubspaceInstrumentation(
            [model],
            targets=(LayerTarget("fc", "fc"),),
            samples_per_batch=1,
            samples_per_phase=1,
        )
        monitor = instrumentation.monitors[0]
        monitor.bank.bases["fc"] = torch.tensor(
            [[2.0], [0.0], [0.0]], dtype=torch.float64
        )
        before = monitor.snapshot_target_weights()["fc"]
        raw = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64
        )
        with torch.no_grad():
            model.fc.weight.copy_(before + raw)

        with self.assertRaisesRegex(RuntimeError, "contractivity guard failed"):
            monitor.online_shrink_parameter_updates(
                {"fc": before}, projection_lambda=1.0
            )
        self.assertTrue(torch.equal(model.fc.weight.detach(), before + raw))
        instrumentation.close()

    def test_online_shrinkage_applied_energy_passes_numerical_guard(self):
        model = TinyLinear().to(dtype=torch.float32)
        instrumentation = SubspaceInstrumentation(
            [model],
            targets=(LayerTarget("fc", "fc"),),
            samples_per_batch=1,
            samples_per_phase=1,
        )
        monitor = instrumentation.monitors[0]
        generator = torch.Generator(device="cpu").manual_seed(53)
        monitor.bank.bases["fc"] = torch.linalg.qr(
            torch.randn(3, 2, generator=generator, dtype=torch.float32)
        ).Q
        before = monitor.snapshot_target_weights()["fc"]
        raw = torch.tensor(
            [[0.3, -0.7, 1.1], [-1.3, 0.2, 0.9]], dtype=torch.float32
        )
        with torch.no_grad():
            model.fc.weight.copy_(before + raw)

        record = monitor.online_shrink_parameter_updates(
            {"fc": before}, projection_lambda=0.75
        )[0]
        self.assertIs(record["online_shrinkage_guard_passed"], True)
        self.assertTrue(
            math.isclose(
                record["applied_control_energy"],
                record["counterfactual_projected_energy"],
                rel_tol=1e-5,
                abs_tol=1e-12,
            )
        )
        instrumentation.close()

    def test_online_mode_does_not_require_csv_schedule(self):
        x = torch.tensor(
            [[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=torch.float64
        )
        y = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
        torch.manual_seed(41)
        model = TinyLinear().to(dtype=torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        instrumentation = SubspaceInstrumentation(
            [model],
            targets=(LayerTarget("fc", "fc"),),
            samples_per_batch=1,
            samples_per_phase=1,
        )
        basis = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=torch.float64,
        )
        instrumentation.monitors[0].bank.bases["fc"] = basis
        client = Client(
            0,
            model,
            optimizer,
            loader,
            replay=None,
            gradient_monitor=instrumentation.monitors[0],
        )
        client.train_one_epoch(
            replay_ratio=0.0,
            log_interval=99,
            update_control="shrinkage_online",
            projection_lambda=0.5,
            round_id=1,
        )
        self.assertEqual(len(client.update_energy_rows), 1)
        self.assertEqual(client.update_energy_rows[0]["update_control"], "shrinkage_online")
        self.assertEqual(client.update_energy_rows[0]["projection_lambda"], 0.5)
        instrumentation.close()

    def test_shrinkage_applies_schedule_when_projection_lambda_is_zero(self):
        x = torch.tensor(
            [[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=torch.float64
        )
        y = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
        torch.manual_seed(41)
        initial = copy.deepcopy(TinyLinear().to(dtype=torch.float64).state_dict())
        initial_weight = initial["fc.weight"].clone()

        baseline_model = TinyLinear().to(dtype=torch.float64)
        baseline_model.load_state_dict(copy.deepcopy(initial))
        baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
        baseline_client = Client(
            0, baseline_model, baseline_optimizer, loader, replay=None
        )
        baseline_client.train_one_epoch(replay_ratio=0.0, log_interval=99)
        raw_displacement = baseline_model.fc.weight.detach() - initial_weight

        controlled_model = TinyLinear().to(dtype=torch.float64)
        controlled_model.load_state_dict(copy.deepcopy(initial))
        controlled_optimizer = torch.optim.Adam(controlled_model.parameters(), lr=1e-3)
        instrumentation = SubspaceInstrumentation(
            [controlled_model],
            targets=(LayerTarget("fc", "fc"),),
            samples_per_batch=1,
            samples_per_phase=1,
        )
        controlled_client = Client(
            0,
            controlled_model,
            controlled_optimizer,
            loader,
            replay=None,
            gradient_monitor=instrumentation.monitors[0],
        )
        factor = 0.4
        controlled_client.train_one_epoch(
            replay_ratio=0.0,
            log_interval=99,
            update_control="shrinkage",
            projection_lambda=0.0,
            shrinkage_factors={"fc": factor},
            round_id=1,
        )
        controlled_displacement = controlled_model.fc.weight.detach() - initial_weight

        self.assertTrue(
            torch.allclose(
                controlled_displacement,
                factor * raw_displacement,
                atol=1e-15,
                rtol=1e-12,
            )
        )
        self.assertFalse(torch.equal(controlled_displacement, raw_displacement))
        self.assertEqual(len(controlled_client.update_energy_rows), 1)
        record = controlled_client.update_energy_rows[0]
        self.assertEqual(record["update_control"], "shrinkage")
        self.assertEqual(record["projection_lambda"], 0.0)
        self.assertEqual(record["shrinkage_factor"], factor)
        self.assertAlmostEqual(record["retained_norm_fraction"], factor)
        self.assertAlmostEqual(record["retained_energy_fraction"], factor**2)
        instrumentation.close()

    def test_schedule_validation_rejects_missing_duplicate_and_bad_round_zero(self):
        expected = {(0, 0, "fc"), (1, 0, "fc")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.csv"

            def write(rows):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["round", "client", "layer", "shrinkage_factor"],
                    )
                    writer.writeheader()
                    writer.writerows(rows)

            valid = [
                {"round": 0, "client": 0, "layer": "fc", "shrinkage_factor": 1},
                {"round": 1, "client": 0, "layer": "fc", "shrinkage_factor": 0.8},
            ]
            write(valid)
            self.assertEqual(load_shrinkage_schedule(path, expected)[(1, 0, "fc")], 0.8)

            write(valid[:1])
            with self.assertRaisesRegex(ValueError, "missing entries"):
                load_shrinkage_schedule(path, expected)

            write(valid + [valid[1]])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_shrinkage_schedule(path, expected)

            invalid_round_zero = [dict(valid[0]), valid[1]]
            invalid_round_zero[0]["shrinkage_factor"] = 0.9
            write(invalid_round_zero)
            with self.assertRaisesRegex(ValueError, "round-0"):
                load_shrinkage_schedule(path, expected)

            invalid_factor = [valid[0], dict(valid[1])]
            invalid_factor[1]["shrinkage_factor"] = 1.1
            write(invalid_factor)
            with self.assertRaisesRegex(ValueError, "invalid shrinkage factor"):
                load_shrinkage_schedule(path, expected)

            write(valid + [
                {"round": 2, "client": 0, "layer": "fc", "shrinkage_factor": 0.8}
            ])
            with self.assertRaisesRegex(ValueError, "unexpected entries"):
                load_shrinkage_schedule(path, expected)

    def test_derivation_uses_energy_sums_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step_log = root / "calibration.csv"
            output = root / "schedule.csv"
            rows = [
                {
                    "round": 1,
                    "client": 0,
                    "layer": "fc",
                    "update_control": "projection",
                    "projection_lambda": 0.75,
                    "raw_update_energy": 1.0,
                    "projected_update_energy": 0.25,
                },
                {
                    "round": 1,
                    "client": 0,
                    "layer": "fc",
                    "update_control": "projection",
                    "projection_lambda": 0.75,
                    "raw_update_energy": 9.0,
                    "projected_update_energy": 2.25,
                },
            ]
            with step_log.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            schedule = derive_schedule(
                step_log, output, rounds=2, clients=1, layers=("fc",)
            )
            self.assertEqual(schedule.loc[0, "shrinkage_factor"], 1.0)
            self.assertEqual(schedule.loc[1, "shrinkage_factor"], 0.5)
            self.assertEqual(schedule.loc[1, "source_file"], step_log.name)
            self.assertRegex(schedule.loc[1, "source_sha256"], r"^[0-9a-f]{64}$")
            with self.assertRaises(FileExistsError):
                derive_schedule(step_log, output, rounds=2, clients=1, layers=("fc",))

    def test_single_round_late_schedule_is_complete_and_loader_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step_log = root / "late_lambda100_update_energy_steps.csv"
            output = root / "late_round5_schedule.csv"
            rows = []
            for client in range(4):
                for layer in ("layer4_1_conv2", "fc"):
                    rows.append({
                        "round": 5,
                        "client": client,
                        "layer": layer,
                        "update_control": "projection",
                        "projection_lambda": 1.0,
                        "raw_update_energy": 4.0,
                        "projected_update_energy": 1.0,
                    })
            with step_log.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            schedule = derive_schedule(
                step_log,
                output,
                rounds=7,
                clients=4,
                layers=("layer4_1_conv2", "fc"),
                projection_lambda=1.0,
                calibration_round=5,
            )
            late = schedule[schedule["round"] == 5]
            self.assertEqual(len(late), 8)
            self.assertEqual(
                len(late[["client", "layer"]].drop_duplicates()), 8
            )
            self.assertTrue((late["shrinkage_factor"] == 0.5).all())
            unused = schedule[schedule["round"] != 5]
            self.assertTrue((unused["shrinkage_factor"] == 1.0).all())
            self.assertEqual(schedule["source_sha256"].nunique(), 1)
            self.assertRegex(schedule.iloc[0]["source_sha256"], r"^[0-9a-f]{64}$")

            expected = {
                (round_id, client, layer)
                for round_id in range(7)
                for client in range(4)
                for layer in ("layer4_1_conv2", "fc")
            }
            loaded = load_shrinkage_schedule(output, expected)
            self.assertEqual(len(loaded), len(expected))
            self.assertEqual(loaded[(5, 0, "fc")], 0.5)
            self.assertEqual(loaded[(4, 0, "fc")], 1.0)


if __name__ == "__main__":
    unittest.main()
