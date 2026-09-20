import hashlib
import random
import subprocess
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.random_phi_control import (
    ENERGY_MATCH_ATOL,
    ENERGY_MATCH_RTOL,
    random_orthonormal_basis,
    random_projection_norm_matched_update,
)


BASE_COMMIT = "79c507dac97cfd4aa13a558679659d169da767a8"
AUDITED_FILES = (
    "src/run_llm_fcl_controller.py",
    "src/checkpointing.py",
    "src/fl.py",
    "src/strategies/replay.py",
    "src/instrumentation/subspace.py",
)
ROOT = Path(__file__).resolve().parents[1]


def numpy_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


class RandomPhiControlTests(unittest.TestCase):
    def test_same_seed_layer_shape_rank_is_exactly_identical(self):
        first, first_seed, first_hash = random_orthonormal_basis(42, "layer4_1_conv2", 16, 4)
        second, second_seed, second_hash = random_orthonormal_basis(42, "layer4_1_conv2", 16, 4)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_seed, second_seed)
        self.assertEqual(first_hash, second_hash)

    def test_seed_and_layer_change_basis(self):
        basis, _, _ = random_orthonormal_basis(42, "layer4_1_conv2", 16, 4)
        seed_changed, _, _ = random_orthonormal_basis(43, "layer4_1_conv2", 16, 4)
        layer_changed, _, _ = random_orthonormal_basis(42, "fc", 16, 4)
        self.assertFalse(torch.equal(basis, seed_changed))
        self.assertFalse(torch.equal(basis, layer_changed))

    def test_orthonormal_and_rank_matches_learned_basis(self):
        basis, _, _ = random_orthonormal_basis(42, "fc", 16, 4)
        self.assertTrue(torch.allclose(basis.T @ basis, torch.eye(4, dtype=basis.dtype), atol=1e-12, rtol=1e-12))
        self.assertEqual(torch.linalg.matrix_rank(basis).item(), 4)
        learned_phi = torch.eye(16, 4, dtype=torch.float64)
        self.assertEqual(basis.shape[1], learned_phi.shape[1])

    def test_basis_construction_preserves_all_rng_states(self):
        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)
        cpu_before = torch.get_rng_state().clone()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        random_orthonormal_basis(42, "layer4_1_conv2", 16, 4)
        self.assertTrue(torch.equal(cpu_before, torch.get_rng_state()))
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(numpy_state_equal(numpy_before, np.random.get_state()))
        if cuda_before is not None:
            self.assertEqual(len(cuda_before), len(torch.cuda.get_rng_state_all()))
            for expected, actual in zip(cuda_before, torch.cuda.get_rng_state_all()):
                self.assertTrue(torch.equal(expected, actual))

    def test_basis_construction_preserves_dataloader_generator_state(self):
        loader_generator = torch.Generator(device="cpu").manual_seed(987)
        before = loader_generator.get_state().clone()
        random_orthonormal_basis(42, "fc", 16, 4)
        self.assertTrue(torch.equal(before, loader_generator.get_state()))

    def test_orientation_differs_and_energy_matches(self):
        raw = torch.tensor([[1.0, 2.0, -1.0], [0.5, -1.0, 3.0]], dtype=torch.float64)
        learned_phi = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
        random_basis, _, _ = random_orthonormal_basis(42, "fc", 3, 1)
        learned = raw - (raw @ learned_phi) @ learned_phi.T
        random_projected = raw - (raw @ random_basis) @ random_basis.T
        self.assertFalse(torch.allclose(learned, random_projected))
        applied, record = random_projection_norm_matched_update(raw, learned_phi, random_basis)
        self.assertTrue(math_isclose(record["applied_random_energy"], record["learned_counterfactual_energy"]))
        self.assertTrue(torch.isfinite(applied).all())

    def test_zero_random_denominator_with_nonzero_target_fails(self):
        raw = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        learned_phi = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        random_basis = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, "effectively zero"):
            random_projection_norm_matched_update(raw, learned_phi, random_basis)

    def test_audited_files_match_base_commit(self):
        for relative_path in AUDITED_FILES:
            expected = subprocess.run(
                ["git", "show", f"{BASE_COMMIT}:{relative_path}"],
                cwd=ROOT, check=True, capture_output=True,
            ).stdout
            actual = (ROOT / relative_path).read_bytes()
            self.assertEqual(
                hashlib.sha256(actual).hexdigest(), hashlib.sha256(expected).hexdigest(), relative_path
            )


def math_isclose(left, right):
    return abs(left - right) <= ENERGY_MATCH_ATOL + ENERGY_MATCH_RTOL * max(abs(left), abs(right))


if __name__ == "__main__":
    unittest.main()
