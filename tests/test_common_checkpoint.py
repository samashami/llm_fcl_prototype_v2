import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    branch_control_metadata,
    capture_rng_state,
    endpoint_state,
    flatten_fingerprints,
    load_checkpoint,
    restore_rng_state,
    requires_endpoint_equality,
    save_checkpoint,
    validate_determinism_gate,
    validate_resume_protocol,
    verify_endpoint_reference,
    write_endpoint_reference,
)
from src.fl import Client
from src.instrumentation.subspace import LayerTarget, SubspaceInstrumentation
from src.strategies.replay import ReplayBuffer
from src.run_llm_fcl_controller import (
    local_epoch_budget,
    should_stop_local_training,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.fc(x)


class WorkerRandomDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, -0.2, 0.7],
                [-0.3, 0.8, 0.5],
                [0.9, 0.1, -0.4],
                [0.2, -0.7, 0.6],
                [-0.5, 0.4, 0.2],
                [0.7, -0.1, 0.8],
                [0.3, 0.6, -0.9],
            ],
            dtype=torch.float64,
        )
        self.y = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # Exercise Torch, Python, and NumPy RNGs inside DataLoader workers.
        noise = torch.rand(3, dtype=torch.float64) * 1e-3
        noise += random.random() * 1e-4
        noise += float(np.random.random()) * 1e-4
        return self.x[index] + noise, self.y[index]


def seed_worker_like_controller(worker_id):
    np.random.seed(91 + worker_id)
    random.seed(91 + worker_id)


TARGETS = (LayerTarget("fc", "fc"),)


def make_runtime(model_state=None):
    model = TinyNet().to(dtype=torch.float64)
    if model_state is not None:
        model.load_state_dict(copy.deepcopy(model_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay = ReplayBuffer(capacity=8)
    client = Client(0, model, optimizer, None, replay=replay)
    instrumentation = SubspaceInstrumentation(
        [model], targets=TARGETS, max_rank=2,
        samples_per_batch=2, samples_per_phase=4,
    )
    client.gradient_monitor = instrumentation.monitors[0]
    return model, client, instrumentation


def make_loader(generator):
    return DataLoader(
        WorkerRandomDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker_like_controller,
        generator=generator,
    )


def run_round(model, client, instrumentation, generator, round_id, local_epochs=1):
    client.loader = make_loader(generator)
    instrumentation.begin_round(round_id)
    for epoch in range(local_epochs):
        loss, accuracy, _ = client.train_one_epoch(
            replay_ratio=0.5,
            epoch=epoch,
            total_epochs=local_epochs,
            projection_lambda=0.0,
            update_control="projection",
            round_id=round_id,
            log_interval=99,
        )
    subspace = instrumentation.end_round()
    summary = {
        "round": round_id,
        "loss": loss,
        "accuracy": accuracy,
        "beta_hat": subspace["beta_hat"],
        "rho_hat": subspace["rho_hat"],
    }
    return endpoint_state(
        model,
        [client],
        instrumentation,
        summary,
        rng_state=capture_rng_state(generator),
    )


class CommonCheckpointChecks(unittest.TestCase):
    def setUp(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)

    def test_lossless_replay_fifo_round_trip(self):
        replay = ReplayBuffer(capacity=3)
        x = torch.tensor([[1.25, -2.5], [3.75, 4.5]], dtype=torch.float64)
        y = torch.tensor([7, 8], dtype=torch.int64)
        replay.add_batch(x, y)
        restored = ReplayBuffer(capacity=1)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.capacity, 3)
        self.assertEqual(len(restored.data), 2)
        for expected, actual in zip(replay.data, restored.data):
            self.assertTrue(torch.equal(expected[0], actual[0]))
            self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_exact_one_round_resume_with_four_workers(self):
        generator = torch.Generator().manual_seed(29)
        model, client, instrumentation = make_runtime()

        # Establish Adam moments, replay contents, persistent counters, and Phi.
        run_round(model, client, instrumentation, generator, round_id=0)
        checkpoint_payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "next_round": 1,
            "parent_run_id": "synthetic-parent",
            "global_model": copy.deepcopy(model.state_dict()),
            "clients": [{
                "cid": 0,
                "optimizer": copy.deepcopy(client.optimizer.state_dict()),
                "replay": client.replay.state_dict(),
                "persistent": client.persistent_state_dict(),
            }],
            "subspace": instrumentation.state_dict(),
            "metrics": {"aulc_running": 0.25, "best_recall": np.array([0.1, 0.2])},
            "rng": capture_rng_state(generator),
            "data_state": {"synthetic": True},
            "manifest": {"protocol": {"synthetic": True}},
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "pre_round_01.pt"
            metadata = save_checkpoint(checkpoint_path, checkpoint_payload)
            self.assertEqual(metadata["size_bytes"], checkpoint_path.stat().st_size)

            parent_endpoint = run_round(
                model, client, instrumentation, generator, round_id=1,
                local_epochs=3,
            )
            reference_path = root / "parent_endpoint_round_01.json"
            write_endpoint_reference(
                reference_path,
                parent_endpoint,
                {"parent_run_id": "synthetic-parent", "executed_round": 1},
            )

            loaded, loaded_metadata = load_checkpoint(checkpoint_path)
            self.assertEqual(loaded_metadata["sha256"], metadata["sha256"])
            restored_model, restored_client, restored_instrumentation = make_runtime()
            restored_model.load_state_dict(loaded["global_model"])
            restored_client.optimizer.load_state_dict(loaded["clients"][0]["optimizer"])
            restored_client.replay.load_state_dict(loaded["clients"][0]["replay"])
            restored_client.load_persistent_state_dict(
                loaded["clients"][0]["persistent"]
            )
            restored_instrumentation.load_state_dict(loaded["subspace"])
            restored_generator = torch.Generator()
            # Restore RNG last, exactly as the production resume path does.
            restore_rng_state(loaded["rng"], restored_generator)
            restored_endpoint = run_round(
                restored_model,
                restored_client,
                restored_instrumentation,
                restored_generator,
                round_id=1,
                local_epochs=3,
            )

            gate = verify_endpoint_reference(reference_path, restored_endpoint)
            self.assertTrue(gate["passed"], json.dumps(gate["differences"], indent=2))
            self.assertEqual(gate["differences"], [])
            self.assertEqual(
                flatten_fingerprints(parent_endpoint),
                flatten_fingerprints(restored_endpoint),
            )
            restored_instrumentation.close()
        instrumentation.close()

    @staticmethod
    def _executed_epochs(stop_values, branch_local_epochs):
        count = 0
        for stop in stop_values[:local_epoch_budget(5, branch_local_epochs)]:
            count += 1
            if should_stop_local_training(stop, branch_local_epochs):
                break
        return count

    def test_fixed_branch_compute_is_identical_across_treatments(self):
        projection_zero = self._executed_epochs([False, True, True], 3)
        projection_one = self._executed_epochs([True, True, True], 3)
        shrinkage = self._executed_epochs([False, False, True], 3)
        self.assertEqual(
            (projection_zero, projection_one, shrinkage), (3, 3, 3)
        )
        batches_per_epoch = 4
        self.assertEqual(
            tuple(count * batches_per_epoch for count in (
                projection_zero, projection_one, shrinkage
            )),
            (12, 12, 12),
        )

    def test_absent_branch_budget_preserves_ordinary_early_stopping(self):
        self.assertEqual(self._executed_epochs([False, True, False], None), 2)

    def test_shrinkage_does_not_enter_lambda_zero_endpoint_comparator(self):
        self.assertTrue(requires_endpoint_equality("projection", 0.0))
        self.assertFalse(requires_endpoint_equality("projection", 1.0))
        self.assertFalse(requires_endpoint_equality("shrinkage", 0.0))

    def test_resume_protocol_allows_only_treatment_change(self):
        parent = {"seed": 42, "epochs": 5, "update_control": "projection"}
        validate_resume_protocol(parent, dict(parent))
        validate_resume_protocol(
            parent, {"seed": 42, "epochs": 5, "update_control": "shrinkage"}
        )
        with self.assertRaisesRegex(RuntimeError, "outside update_control"):
            validate_resume_protocol(
                parent, {"seed": 43, "epochs": 5, "update_control": "shrinkage"}
            )

    def test_gate_must_match_the_resumed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            gate_path = Path(directory) / "gate.json"
            gate_path.write_text(
                json.dumps({
                    "parent_run_id": "parent",
                    "executed_round": 5,
                    "source_checkpoint_sha256": "abc",
                    "determinism_gate": {"passed": True},
                }),
                encoding="utf-8",
            )
            document = validate_determinism_gate(
                gate_path,
                parent_run_id="parent",
                executed_round=5,
                source_checkpoint_sha256="abc",
            )
            self.assertTrue(document["determinism_gate"]["passed"])
            with self.assertRaisesRegex(RuntimeError, "another checkpoint"):
                validate_determinism_gate(
                    gate_path,
                    parent_run_id="parent",
                    executed_round=1,
                    source_checkpoint_sha256="abc",
                )

    def test_projection_and_shrinkage_load_same_start_state_hash(self):
        generator = torch.Generator().manual_seed(29)
        model, client, instrumentation = make_runtime()
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "next_round": 5,
            "parent_run_id": "synthetic-parent",
            "global_model": copy.deepcopy(model.state_dict()),
            "clients": [],
            "subspace": instrumentation.state_dict(),
            "metrics": {},
            "rng": capture_rng_state(generator),
            "data_state": {},
            "manifest": {"protocol": {"update_control": "projection"}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pre_round_05.pt"
            saved = save_checkpoint(path, payload)
            _, projection_metadata = load_checkpoint(path)
            _, shrinkage_metadata = load_checkpoint(path)
            self.assertEqual(
                projection_metadata["starting_state_hash"],
                shrinkage_metadata["starting_state_hash"],
            )
            self.assertEqual(
                projection_metadata["starting_state_hash"],
                saved["starting_state_hash"],
            )
        instrumentation.close()

    def test_branch_metadata_records_schedule_checksum_and_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            schedule = Path(directory) / "schedule.csv"
            schedule.write_text("frozen schedule\n", encoding="utf-8")
            metadata = branch_control_metadata(
                update_control="shrinkage",
                branch_local_epochs=3,
                shrinkage_schedule=schedule,
                client_epoch_counts={0: 3, 1: 3},
                client_optimizer_step_counts={0: 12, 1: 15},
            )
            self.assertEqual(metadata["update_control"], "shrinkage")
            self.assertEqual(metadata["branch_local_epochs"], 3)
            self.assertEqual(metadata["shrinkage_schedule_path"], str(schedule))
            self.assertRegex(metadata["shrinkage_schedule_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(metadata["per_client_epoch_counts"], {"0": 3, "1": 3})
            self.assertEqual(
                metadata["per_client_optimizer_step_counts"],
                {"0": 12, "1": 15},
            )

    def test_gate_reports_exact_differing_path(self):
        model, client, instrumentation = make_runtime()
        endpoint = endpoint_state(
            model, [client], instrumentation, {"round": 1, "accuracy": 50.0}
        )
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.json"
            write_endpoint_reference(reference, endpoint, {"executed_round": 1})
            with torch.no_grad():
                model.fc.weight[0, 0].add_(1.0)
            changed = endpoint_state(
                model, [client], instrumentation, {"round": 1, "accuracy": 50.0}
            )
            gate = verify_endpoint_reference(reference, changed)
            self.assertFalse(gate["passed"])
            paths = [difference["path"] for difference in gate["differences"]]
            self.assertIn("root/str:'global_model'/str:'fc.weight'", paths)
        instrumentation.close()


if __name__ == "__main__":
    unittest.main()
