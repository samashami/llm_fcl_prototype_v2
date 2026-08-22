import unittest

import torch

from src.instrumentation.subspace import soft_project_rows


def make_random_phi(input_features, rank, seed, dtype=torch.float64):
    """Return a deterministic CPU basis with orthonormal columns."""
    if not 0 < rank <= input_features:
        raise ValueError("rank must be in [1, input_features]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randn(
        input_features, rank, generator=generator, dtype=dtype, device="cpu"
    )
    return torch.linalg.qr(samples, mode="reduced").Q


class SoftProjectionChecks(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device="cpu").manual_seed(7)
        self.matrix = torch.randn(4, 6, generator=generator, dtype=torch.float64)
        self.phi = make_random_phi(6, 3, seed=11)

    def test_lambda_zero_returns_matrix_exactly_without_aliasing(self):
        original = self.matrix.clone()
        projected = soft_project_rows(self.matrix, self.phi, 0.0)
        self.assertTrue(torch.equal(projected, self.matrix))
        self.assertNotEqual(projected.data_ptr(), self.matrix.data_ptr())
        self.assertTrue(torch.equal(self.matrix, original))

    def test_lambda_one_removes_parallel_component(self):
        projected = soft_project_rows(self.matrix, self.phi, 1.0)
        self.assertTrue(
            torch.allclose(
                projected @ self.phi,
                torch.zeros(4, 3, dtype=self.matrix.dtype),
                atol=1e-12,
                rtol=1e-12,
            )
        )

    def test_intermediate_lambda_scales_only_parallel_component(self):
        parallel = (self.matrix @ self.phi) @ self.phi.T
        perpendicular = self.matrix - parallel
        for lambda_value in (0.25, 0.5, 0.75):
            with self.subTest(lambda_value=lambda_value):
                projected = soft_project_rows(self.matrix, self.phi, lambda_value)
                expected = perpendicular + (1.0 - lambda_value) * parallel
                self.assertTrue(
                    torch.allclose(projected, expected, atol=1e-12, rtol=1e-12)
                )

    def test_perpendicular_component_is_unchanged(self):
        original_perpendicular = self.matrix - (self.matrix @ self.phi) @ self.phi.T
        for lambda_value in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(lambda_value=lambda_value):
                projected = soft_project_rows(self.matrix, self.phi, lambda_value)
                projected_perpendicular = projected - (
                    projected @ self.phi
                ) @ self.phi.T
                self.assertTrue(
                    torch.allclose(
                        projected_perpendicular,
                        original_perpendicular,
                        atol=1e-12,
                        rtol=1e-12,
                    )
                )

    def test_random_phi_has_orthonormal_columns(self):
        identity = torch.eye(self.phi.shape[1], dtype=self.phi.dtype)
        self.assertTrue(
            torch.allclose(self.phi.T @ self.phi, identity, atol=1e-12, rtol=1e-12)
        )

    def test_invalid_shapes_fail_clearly(self):
        cases = (
            (torch.randn(6), self.phi, "matrix must be 2-D"),
            (self.matrix, torch.randn(6), "phi must be 2-D"),
            (
                self.matrix,
                torch.empty(6, 0, dtype=self.matrix.dtype),
                "at least one basis vector",
            ),
            (
                self.matrix,
                torch.randn(5, 2, dtype=self.matrix.dtype),
                "input dimension",
            ),
        )
        for matrix, phi, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    soft_project_rows(matrix, phi, 0.5)

    def test_unsupported_lambda_values_fail_clearly(self):
        for lambda_value in (-0.25, 0.1, 0.3, 1.25, True, "0.5"):
            with self.subTest(lambda_value=lambda_value):
                with self.assertRaisesRegex(ValueError, "lambda_value must be one of"):
                    soft_project_rows(self.matrix, self.phi, lambda_value)

    def test_cpu_dtype_and_device_are_preserved(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                matrix = self.matrix.to(dtype=dtype)
                phi = self.phi.to(dtype=dtype)
                projected = soft_project_rows(matrix, phi, 0.5)
                self.assertEqual(projected.dtype, dtype)
                self.assertEqual(projected.device, matrix.device)
                self.assertEqual(projected.device.type, "cpu")

    def test_random_phi_is_reproducible(self):
        first = make_random_phi(10, 4, seed=1234)
        second = make_random_phi(10, 4, seed=1234)
        different = make_random_phi(10, 4, seed=1235)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))


if __name__ == "__main__":
    unittest.main()
