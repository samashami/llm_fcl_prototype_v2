import unittest

from src.external.gfedcl_datail import build_frozen_partition


class TestGFedCLDataILAdapter(unittest.TestCase):
    def test_smoke_partition_matches_frozen_protocol_shape(self):
        state = build_frozen_partition(
            data_dir="./data",
            seed=42,
            num_clients=4,
            val_size=700,
            subset_per_client=140,
            domain_order_name="heldout",
        )
        self.assertEqual(len(state["client_splits"]), 4)
        self.assertEqual([len(x) for x in state["client_splits"]], [140] * 4)
        self.assertEqual(len(state["schedules"]), 4)
        self.assertTrue(all(len(s) == 7 for s in state["schedules"]))
        self.assertTrue(all(sum(len(b) for b in s) == 140 for s in state["schedules"]))
        self.assertEqual(len(state["domain_order"]), 7)


if __name__ == "__main__":
    unittest.main()
