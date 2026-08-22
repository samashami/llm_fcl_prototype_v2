import copy
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.fl import Client
from src.instrumentation.subspace import LayerTarget, SubspaceInstrumentation


class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.fc(x)


class SoftProjectionIntegrationChecks(unittest.TestCase):
    def test_lambda_zero_matches_existing_adam_step_exactly(self):
        x = torch.tensor(
            [[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=torch.float32
        )
        y = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

        torch.manual_seed(19)
        initial_state = copy.deepcopy(TinyLinear().state_dict())

        def run(with_projection_integration):
            model = TinyLinear()
            model.load_state_dict(copy.deepcopy(initial_state))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            client = Client(0, model, optimizer, loader, replay=None)
            instrumentation = None
            if with_projection_integration:
                instrumentation = SubspaceInstrumentation(
                    [model],
                    targets=(LayerTarget("fc", "fc"),),
                    max_rank=2,
                    samples_per_batch=1,
                    samples_per_phase=1,
                )
                generator = torch.Generator(device="cpu").manual_seed(23)
                instrumentation.bank.bases["fc"] = torch.linalg.qr(
                    torch.randn(3, 2, generator=generator)
                ).Q
                instrumentation.completed_phases.add(0)
                instrumentation.begin_round(phase_id=0)
                client.gradient_monitor = instrumentation.monitors[0]

            client.train_one_epoch(
                replay_ratio=0.0,
                log_interval=99,
                projection_lambda=0.0,
            )
            parameters = copy.deepcopy(model.state_dict())
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            if instrumentation is not None:
                instrumentation.end_round()
                instrumentation.close()
            return parameters, optimizer_state

        baseline_parameters, baseline_optimizer = run(False)
        projected_parameters, projected_optimizer = run(True)

        for name in baseline_parameters:
            self.assertTrue(
                torch.equal(baseline_parameters[name], projected_parameters[name])
            )
        self.assertEqual(baseline_optimizer["param_groups"], projected_optimizer["param_groups"])
        for baseline_state, projected_state in zip(
            baseline_optimizer["state"].values(),
            projected_optimizer["state"].values(),
        ):
            self.assertEqual(baseline_state.keys(), projected_state.keys())
            for key in baseline_state:
                self.assertTrue(
                    torch.equal(baseline_state[key], projected_state[key])
                )

    def test_lambda_one_projects_realized_adam_displacement(self):
        x = torch.tensor(
            [[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]], dtype=torch.float64
        )
        y = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

        torch.manual_seed(29)
        model = TinyLinear().to(dtype=torch.float64)
        weight_before = model.fc.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        client = Client(0, model, optimizer, loader, replay=None)

        instrumentation = SubspaceInstrumentation(
            [model],
            targets=(LayerTarget("fc", "fc"),),
            max_rank=2,
            samples_per_batch=1,
            samples_per_phase=1,
        )
        generator = torch.Generator(device="cpu").manual_seed(31)
        phi = torch.linalg.qr(
            torch.randn(3, 2, generator=generator, dtype=torch.float64)
        ).Q
        instrumentation.bank.bases["fc"] = phi
        instrumentation.completed_phases.add(0)
        instrumentation.begin_round(phase_id=0)
        client.gradient_monitor = instrumentation.monitors[0]

        client.train_one_epoch(
            replay_ratio=0.0,
            log_interval=99,
            projection_lambda=1.0,
        )
        displacement = model.fc.weight.detach() - weight_before
        self.assertGreater(torch.linalg.vector_norm(displacement).item(), 0.0)
        self.assertTrue(
            torch.allclose(
                displacement @ phi,
                torch.zeros(2, 2, dtype=torch.float64),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        instrumentation.end_round()
        instrumentation.close()


if __name__ == "__main__":
    unittest.main()
