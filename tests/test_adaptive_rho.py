import math
import unittest

from src.policy.adaptive_rho import (
    FORGETTING_THRESHOLD,
    REPLAY_PALETTE,
    USES_ROLLBACK,
    build_action,
    select_replay_ratio,
)
from src.policy.lmss_local import STRATEGY_PALETTE as LOCAL_PALETTE
from src.policy.lmss_openrouter import STRATEGY_PALETTE as OPENROUTER_PALETTE
from src.agent_io import validate_action
from src.run_llm_fcl_controller import (
    V4_CLIENT_LR_MAX,
    V4_CLIENT_LR_MIN,
    V4_FORGET_THR,
    V4_LR_BOOST,
    V4_LR_COOLDOWN,
    V4_REP_STEP_HIGH,
    V4_REP_STEP_LOW,
)


class AdaptiveRhoTests(unittest.TestCase):
    def decide(self, round_id, previous=0.5, forgetting=0.0):
        return select_replay_ratio(
            round_id=round_id,
            previous_replay_ratio=previous,
            forgetting=forgetting,
        )

    def test_rounds_zero_and_one_are_fixed_warmup(self):
        for round_id in (0, 1):
            decision = self.decide(round_id, previous=0.7, forgetting=1.0)
            self.assertEqual(decision.replay_ratio, 0.5)
            self.assertEqual(decision.reason, "warmup")

    def test_forgetting_above_threshold_increases_one_level(self):
        decision = self.decide(2, previous=0.5, forgetting=FORGETTING_THRESHOLD + 1e-3)
        self.assertEqual(decision.replay_ratio, 0.6)
        self.assertEqual(decision.reason, "increase_forgetting")

    def test_forgetting_below_threshold_decreases_one_level(self):
        decision = self.decide(2, previous=0.5, forgetting=FORGETTING_THRESHOLD - 1e-3)
        self.assertEqual(decision.replay_ratio, 0.4)
        self.assertEqual(decision.reason, "decrease_forgetting")

    def test_equality_retains_previous_level(self):
        decision = self.decide(2, previous=0.6, forgetting=FORGETTING_THRESHOLD)
        self.assertEqual(decision.replay_ratio, 0.6)
        self.assertEqual(decision.reason, "retain_equal")

    def test_nonfinite_forgetting_retains_previous_level(self):
        for forgetting in (float("nan"), float("inf"), float("-inf")):
            decision = self.decide(2, previous=0.6, forgetting=forgetting)
            self.assertEqual(decision.replay_ratio, 0.6)
            self.assertEqual(decision.reason, "retain_nonfinite")
            self.assertFalse(math.isfinite(decision.forgetting))

    def test_palette_boundaries(self):
        self.assertEqual(
            self.decide(2, previous=0.7, forgetting=0.1).reason, "at_upper_bound"
        )
        self.assertEqual(
            self.decide(2, previous=0.3, forgetting=0.0).reason, "at_lower_bound"
        )

    def test_persistence_across_sequential_decisions(self):
        first = self.decide(2, previous=0.5, forgetting=0.1)
        second = self.decide(3, previous=first.replay_ratio, forgetting=0.1)
        third = self.decide(4, previous=second.replay_ratio, forgetting=0.0)
        self.assertEqual((first.replay_ratio, second.replay_ratio, third.replay_ratio), (0.6, 0.7, 0.6))

    def test_action_keeps_lr_fixed_and_client_scales_unity(self):
        decision = self.decide(2, previous=0.5, forgetting=0.1)
        action = build_action(n_clients=4, fixed_lr=1e-4, decision=decision)
        self.assertEqual(action["lr"], 1e-4)
        self.assertEqual(action["aggregation"], {"method": "FedAvg"})
        self.assertEqual(action["client_selection_k"], 4)
        self.assertTrue(all(p["replay_ratio"] == 0.6 for p in action["client_params"]))
        self.assertTrue(all(p["lr_scale"] == 1.0 for p in action["client_params"]))

    def test_no_rollback_is_enabled(self):
        self.assertFalse(USES_ROLLBACK)

    def test_existing_fixed_v4_and_lmss_policy_contracts_are_unchanged(self):
        fixed = validate_action(
            {
                "client_selection_k": 4,
                "aggregation": {"method": "FedAvg"},
                "client_params": [
                    {"id": client_id, "replay_ratio": 0.5, "lr_scale": 1.0, "ewc_lambda": 0.0}
                    for client_id in range(4)
                ],
            },
            n_clients=4,
            policy_source="Fixed",
        )
        self.assertEqual(fixed["policy_source"], "Fixed")
        self.assertTrue(all(p["replay_ratio"] == 0.5 for p in fixed["client_params"]))
        self.assertTrue(all(p["lr_scale"] == 1.0 for p in fixed["client_params"]))
        self.assertEqual((V4_FORGET_THR, V4_REP_STEP_HIGH, V4_REP_STEP_LOW), (0.05, 0.10, 0.05))
        self.assertEqual((V4_LR_BOOST, V4_LR_COOLDOWN), (1.35, 1.50))
        self.assertEqual((V4_CLIENT_LR_MIN, V4_CLIENT_LR_MAX), (0.8, 1.2))
        expected = {
            0: (1e-4, 0.8, 0.6),
            1: (1e-4, 1.0, 0.5),
            2: (1.5e-4, 1.0, 0.7),
            3: (3e-4, 1.0, 0.4),
            4: (4.5e-4, 1.0, 0.3),
        }
        for palette in (LOCAL_PALETTE, OPENROUTER_PALETTE):
            self.assertEqual(
                {key: (value["lr"], value["lr_scale"], value["replay_ratio"]) for key, value in palette.items()},
                expected,
            )
        self.assertEqual(REPLAY_PALETTE, (0.3, 0.4, 0.5, 0.6, 0.7))


if __name__ == "__main__":
    unittest.main()
