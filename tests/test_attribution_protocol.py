import unittest
import json
import tempfile
from pathlib import Path

from src.attribution_protocol import (
    accumulate_post_initialization_utility,
    block_epoch_budget,
    load_frozen_action_schedule,
    nearest_strategy_id,
    resolve_domain_order,
    restrict_action_axis,
    stage_and_block,
    validate_frozen_schedule_provenance,
)


class AttributionProtocolTests(unittest.TestCase):
    def test_fixed_highlr_configuration_is_isolated(self):
        from pathlib import Path
        source = Path("src/run_llm_fcl_controller.py").read_text(encoding="utf-8")
        self.assertIn('args.controller == "fixed_highlr" and not args.attribution_protocol', source)
        self.assertIn('args.controller in {"fixed", "fixed_highlr"}', source)
        self.assertIn('hp_lr = 1.5e-4', source)
        self.assertIn('if args.attribution_protocol and stage_position == 0:', source)
        self.assertIn('fixed_lr=args.lr', source)

    def test_two_blocks_map_to_one_stage(self):
        self.assertEqual(stage_and_block(0, 2), (0, 0))
        self.assertEqual(stage_and_block(1, 2), (0, 1))
        self.assertEqual(stage_and_block(12, 2), (6, 0))
        self.assertEqual(stage_and_block(13, 2), (6, 1))

    def test_epoch_budget_preserves_odd_stage_total(self):
        budgets = [block_epoch_budget(5, 2, block) for block in range(2)]
        self.assertEqual(budgets, [3, 2])
        self.assertEqual(sum(budgets), 5)

    def test_utility_uses_exactly_twelve_post_initialization_blocks(self):
        total, count = 0.0, 0
        for round_id in range(14):
            stage_id, _ = stage_and_block(round_id, 2)
            total, count = accumulate_post_initialization_utility(
                total,
                count,
                stage_id=stage_id,
                mean_seen_domain_accuracy=float(round_id),
            )
        self.assertEqual(count, 12)
        self.assertEqual(total / count, sum(range(2, 14)) / 12)

    def test_heldout_order_swaps_brightness_and_noise(self):
        self.assertEqual(resolve_domain_order("development", 7), (0, 1, 2, 3, 4, 5, 6))
        self.assertEqual(resolve_domain_order("heldout", 7), (0, 4, 2, 3, 1, 5, 6))

    def test_axis_restrictions_preserve_only_the_active_actuator(self):
        requested = {
            "lr": 4.5e-4,
            "client_selection_k": 2,
            "client_params": [
                {"id": 0, "replay_ratio": 0.3, "lr_scale": 0.8, "ewc_lambda": 4.0}
            ],
            "policy_source": "test",
        }
        eta = restrict_action_axis(
            requested,
            mode="eta_only",
            n_clients=4,
            fixed_lr=1e-4,
            fixed_replay_ratio=0.5,
        )
        rho = restrict_action_axis(
            requested,
            mode="rho_only",
            n_clients=4,
            fixed_lr=1e-4,
            fixed_replay_ratio=0.5,
        )
        self.assertEqual(eta["lr"], 4.5e-4)
        self.assertTrue(all(row["replay_ratio"] == 0.5 for row in eta["client_params"]))
        self.assertEqual(rho["lr"], 1e-4)
        self.assertTrue(all(row["replay_ratio"] == 0.3 for row in rho["client_params"]))
        self.assertTrue(all(row["lr_scale"] == 1.0 for row in eta["client_params"]))
        self.assertEqual(len(rho["client_params"]), 4)

    def test_frozen_schedule_requires_every_declared_round_and_is_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_protocol.json").write_text(
                json.dumps({"manifest": {"protocol": {}}}),
                encoding="utf-8",
            )
            for round_id in (2, 3):
                (root / f"action_round_{round_id}.json").write_text(
                    json.dumps({"lr": 1e-4, "client_params": []}),
                    encoding="utf-8",
                )
            actions, first_hash, provenance = load_frozen_action_schedule(
                directory, [2, 3]
            )
            _, second_hash, _ = load_frozen_action_schedule(directory, [2, 3])
            self.assertEqual(sorted(actions), [2, 3])
            self.assertIn("manifest", provenance)
            self.assertEqual(first_hash, second_hash)
            with self.assertRaises(ValueError):
                load_frozen_action_schedule(directory, [2, 3, 4])

    def test_frozen_schedule_provenance_rejects_fallbacks(self):
        palette = {0: {"lr": 1e-4, "replay_ratio": 0.5}}
        provenance = {
            "manifest": {
                "protocol": {
                    "attribution_protocol": True,
                    "controller": "lmss_openrouter",
                    "control_mode": "joint",
                    "domain_order": "development",
                    "evaluation_source": "validation",
                    "seed": 40,
                    "resolved_lmss_model": "model/version",
                },
                "code": {
                    "files_sha256": {
                        "src/policy/lmss_openrouter.py": "policy-hash"
                    }
                },
            },
            "strategy_palette": {"0": palette[0]},
        }
        action = {
            "strategy_id": 0,
            "control_mode": "joint",
            "lr": 1e-4,
            "client_params": [{"id": 0, "replay_ratio": 0.5}],
            "controller_metadata": {
                "fallback": False,
                "call_count": 1,
                "requested_model": "model/version",
                "prompt_sha256": "a" * 64,
            },
        }
        validate_frozen_schedule_provenance(
            {2: action},
            provenance,
            expected_model="model/version",
            expected_palette=palette,
            expected_policy_sha256="policy-hash",
        )
        fallback = json.loads(json.dumps(action))
        fallback["controller_metadata"]["fallback"] = True
        with self.assertRaisesRegex(ValueError, "fallback"):
            validate_frozen_schedule_provenance(
                {2: fallback},
                provenance,
                expected_model="model/version",
                expected_palette=palette,
                expected_policy_sha256="policy-hash",
            )
        altered = json.loads(json.dumps(action))
        altered["client_params"][0]["replay_ratio"] = 0.4
        with self.assertRaisesRegex(ValueError, "replay ratio"):
            validate_frozen_schedule_provenance(
                {2: altered},
                provenance,
                expected_model="model/version",
                expected_palette=palette,
                expected_policy_sha256="policy-hash",
            )

    def test_frozen_schedule_requires_source_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "run_protocol"):
                load_frozen_action_schedule(directory, [2])

    def test_joint_rule_projection_uses_the_nearest_frozen_strategy(self):
        palette = {
            0: {"lr": 1e-4, "replay_ratio": 0.6},
            1: {"lr": 3e-4, "replay_ratio": 0.4},
        }
        self.assertEqual(
            nearest_strategy_id(
                target_lr=2.9e-4,
                target_replay_ratio=0.41,
                palette=palette,
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
