# QCDG-LMSS experiment

QCDG (queue-conditioned drift-gated LMSS) is an optional controller experiment.
It does not alter the FCL optimizer, replay buffer, FedAvg, or treatment math.

## Flags

- `--use_queue_features` (or `USE_QUEUE_FEATURES=1`) adds `Q`, `P`, `delta_q`,
  and bounded variants to LMSS state/prompt input.
- `--use_drift_gate` (or `USE_DRIFT_GATE=1`) invokes an LMSS controller only
  when the first round or a drift signal triggers; otherwise it reuses its
  previous validated action.
- `--qcdg_tau_q`, `--qcdg_tau_d`, and `--qcdg_tau_a` set thresholds for
  delta-Q (default `.02`), divergence (default `.05`), and absolute accuracy
  change (default `.01`). Environment counterparts are `QCDG_TAU_Q`,
  `QCDG_TAU_D`, and `QCDG_TAU_A`.

QCDG requires `lmss_api`, `lmss_local`, or `lmss_openrouter`. With neither
QCDG flag, those controllers follow their original call path.

## Signals

After local training in round `r`, the controller computes mean final-epoch
client loss `L_new` and replay CE `L_buffer` from up to 64 buffered examples
per client. Empty buffers use that client's `L_new`. The replay probe uses a
private deterministic sampler, so it does not consume the replay buffer's
global Python RNG sequence.

`delta` is an EMA of `L_new` (`beta=.9`) and:

```text
Q_r = max(0, Q_(r-1) + L_buffer - L_new - delta)
P_r = L_new
delta_q = Q_r - Q_(r-1)
```

The gate calls LMSS in round zero or when `delta_q > tau_q`,
`divergence > tau_d`, or `abs(delta_acc) > tau_a`; it prevents back-to-back
calls by default.

Round CSV output adds `Q`, `P`, `delta_q`, `qcdg_trigger`, `lmss_called`, and
`qcdg_strategy` only when QCDG is enabled.
