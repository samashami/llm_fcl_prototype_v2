import copy
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.fl import Client
from src.instrumentation.subspace import (
    LayerTarget,
    SubspaceBank,
    SubspaceInstrumentation,
    gradient_energy,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        x = torch.relu(self.features(x))
        return self.fc(self.pool(x).flatten(1))


TARGETS = (
    LayerTarget("features", "features"),
    LayerTarget("fc", "fc"),
)


class SubspaceChecks(unittest.TestCase):
    def test_basis_is_orthonormal(self):
        bank = SubspaceBank(explained_energy=0.9, max_rank=4)
        generator = torch.Generator().manual_seed(1)
        bank.update({"layer": torch.randn(9, 12, generator=generator)})
        phi = bank.bases["layer"]
        identity = torch.eye(phi.shape[1])
        self.assertTrue(torch.allclose(phi.T @ phi, identity, atol=1e-5, rtol=1e-5))
        self.assertLess(bank.orthonormality_errors()["layer"], 1e-5)
        first_rank = phi.shape[1]
        bank.update({"layer": torch.randn(9, 12, generator=generator)})
        self.assertGreaterEqual(bank.bases["layer"].shape[1], first_rank)
        self.assertLessEqual(bank.bases["layer"].shape[1], 4)
        self.assertLess(bank.orthonormality_errors()["layer"], 1e-5)

    def test_energy_decomposition(self):
        generator = torch.Generator().manual_seed(2)
        gradient = torch.randn(5, 7, generator=generator)
        phi = torch.linalg.qr(torch.randn(7, 3, generator=generator)).Q
        values = gradient_energy(gradient, phi)
        projected = (gradient @ phi) @ phi.T
        residual = gradient - projected
        self.assertAlmostEqual(values["inside"], projected.square().sum().item(), places=5)
        self.assertAlmostEqual(values["outside"], residual.square().sum().item(), places=5)
        self.assertAlmostEqual(
            values["total"], values["inside"] + values["outside"], places=5
        )

    def test_first_phase_has_no_prior_basis(self):
        model = TinyNet()
        instrumentation = SubspaceInstrumentation(
            [model], targets=TARGETS, max_rank=3,
            samples_per_batch=2, samples_per_phase=4,
        )
        instrumentation.begin_round(phase_id=0)
        self.assertEqual(instrumentation.bank.ranks(), {})

        model.train()
        loss = model(torch.randn(2, 1, 4, 4)).sum()
        loss.backward()
        instrumentation.monitors[0].measure_gradients()
        summary = instrumentation.end_round()
        self.assertFalse(summary["protected_basis_exists"])
        self.assertEqual(summary["protected_basis_rank"], 0)
        self.assertEqual(summary["gradient_measurement_count"], 0)
        self.assertGreater(summary["protected_basis_rank_after_update"], 0)
        instrumentation.close()

    def test_instrumentation_does_not_change_training(self):
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(6, 1, 4, 4, generator=generator)
        y = torch.tensor([0, 1, 1, 0, 1, 0])
        loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)
        torch.manual_seed(4)
        initial = TinyNet().state_dict()

        def run(enabled):
            model = TinyNet()
            model.load_state_dict(copy.deepcopy(initial))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
            client = Client(0, model, optimizer, loader, replay=None)
            instrumentation = None
            if enabled:
                instrumentation = SubspaceInstrumentation(
                    [model], targets=TARGETS, max_rank=2,
                    samples_per_batch=2, samples_per_phase=4,
                )
                # Pretend phase zero has already supplied valid protected bases;
                # this exercises gradient measurement during the identical update.
                basis_generator = torch.Generator().manual_seed(5)
                instrumentation.bank.bases["features"] = torch.linalg.qr(
                    torch.randn(9, 2, generator=basis_generator)
                ).Q
                instrumentation.bank.bases["fc"] = torch.linalg.qr(
                    torch.randn(3, 2, generator=basis_generator)
                ).Q
                instrumentation.completed_phases.add(0)
                instrumentation.begin_round(phase_id=0)
                client.gradient_monitor = instrumentation.monitors[0]

            loss, accuracy, _ = client.train_one_epoch(replay_ratio=0.0, log_interval=99)
            parameters = [p.detach().clone() for p in model.parameters()]
            gradients = [p.grad.detach().clone() for p in model.parameters()]
            if instrumentation is not None:
                summary = instrumentation.end_round()
                self.assertGreater(summary["gradient_measurement_count"], 0)
                instrumentation.close()
            return loss, accuracy, parameters, gradients

        disabled = run(False)
        enabled = run(True)
        self.assertEqual(disabled[0], enabled[0])
        self.assertEqual(disabled[1], enabled[1])
        for expected, measured in zip(disabled[2], enabled[2]):
            self.assertTrue(torch.equal(expected, measured))
        for expected, measured in zip(disabled[3], enabled[3]):
            self.assertTrue(torch.equal(expected, measured))


if __name__ == "__main__":
    unittest.main()
