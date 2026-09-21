import unittest

from src.policy.qcdg import QueueState, env_flag_true, queue_fields_for_state, should_invoke_lmss, update_queue


class QCDGTests(unittest.TestCase):
    def test_queue_grows_for_worse_replay_loss(self):
        state = update_queue(QueueState(), l_new=1.0, l_buffer=3.0)
        self.assertGreater(state.Q, 0.0)
        self.assertEqual(state.P, 1.0)
        self.assertEqual(state.delta_q, state.Q)

    def test_queue_is_nonnegative_and_ema_is_deterministic(self):
        first = update_queue(QueueState(), l_new=2.0, l_buffer=2.0)
        second = update_queue(first, l_new=2.0, l_buffer=0.5)
        self.assertEqual(first.Q, 0.0)
        self.assertEqual(second.Q, 0.0)
        self.assertEqual(second.ema_l_new, 2.0)

    def test_queue_features_are_bounded(self):
        fields = queue_fields_for_state(QueueState(Q=100.0, P=3.0, delta_q=-100.0))
        self.assertLessEqual(fields["Q_norm"], 1.0)
        self.assertGreaterEqual(fields["P_norm"], 0.0)
        self.assertGreaterEqual(fields["delta_q_norm"], -1.0)

    def test_gate_first_round_drift_and_cooldown(self):
        self.assertTrue(should_invoke_lmss(delta_q=0, divergence=0, delta_acc=0, round_id=0))
        self.assertFalse(should_invoke_lmss(delta_q=0, divergence=0, delta_acc=0, round_id=2))
        self.assertFalse(should_invoke_lmss(delta_q=1, divergence=0, delta_acc=0, round_id=3, last_invoked_round=2))
        self.assertTrue(should_invoke_lmss(delta_q=1, divergence=0, delta_acc=0, round_id=4, last_invoked_round=2))

    def test_environment_boolean_parser(self):
        self.assertFalse(env_flag_true("QCDG_TEST_UNSET"))


if __name__ == "__main__":
    unittest.main()
