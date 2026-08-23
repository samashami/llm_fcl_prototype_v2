# LMSS-GEO Experiments

## Stage 0 – Measurement Only

### Equal Split
- Seed: 42
- Controller: Fixed
- Projection: Disabled
- Measurement: Enabled

Results:
- `equal_split/seed42/`

---

### Dirichlet α = 0.5
- Seed: 42
- Controller: Fixed
- Projection: Disabled
- Measurement: Enabled

Results:
- `dirichlet_a05/seed42/`

---

### Dirichlet α = 0.1
- Seed: 42
- Controller: Fixed
- Projection: Disabled
- Measurement: Enabled

Results:
- `dirichlet_a01/seed42/`

## Stage 0 result

Across valid rounds 1–6, mean β̂ increased with client heterogeneity: 0.4947
(equal), 0.5952 (Dirichlet α=0.5), and 0.6449 (Dirichlet α=0.1). Its
round-wise range also increased from 0.0123 to 0.0731 and 0.1002,
respectively. Mean ρ̂ decreased correspondingly from 0.7109 to 0.6359 and
0.5952; ρ̂ is derived from the complementary gradient-energy fraction and is
not independent confirmation of β̂. Final accuracies were 68.00%, 62.31%, and
52.80%. Average measurement overhead across all seven rounds was 50.86 s,
49.29 s, and 48.31 s per round.

The equal-split β̂ trace was nearly flat, while both non-IID traces had a clear
early decline followed by relative stabilization. This seed-42-only viability
check therefore supports a **GO to fixed-lambda Stage 1**: heterogeneity changes
the measured interference regime, and non-IID runs provide some temporal
variation against which projection strength can be tested. It does not establish
statistical significance or yet demonstrate that state-conditioned control
outperforms a fixed λ.

Reproduce the analysis from the repository root with:

```bash
python3 paper/plots_paper/lmss_geo/analyze_stage0_measurement.py
```

Generated tables and plots are in `stage0_measurement/analysis/`.

## Stage 1 – Fixed soft projection

The seed-42 Dirichlet α=0.1 sweep used fixed
λ ∈ {0, 0.25, 0.5, 0.75, 1}. Final accuracy decreased monotonically from
52.80% to 50.64% as λ increased, while mean forgetting decreased from 3.79%
to 2.66%. AULC also decreased monotonically (42.86% to 41.15%), and final
client divergence increased rather than decreased (0.0768 to 0.0921).

No interior λ wins a requested endpoint: λ=0 gives the best final accuracy,
AULC, and divergence, while λ=1 gives the lowest forgetting. However, λ=0.75
is a descriptive knee, retaining about 75% of the full forgetting reduction at
about 39% of the full accuracy cost. This indicates a stability–plasticity
response, but does not establish that learned Φ protects knowledge better than
generic directional update shrinkage.

Decision: **CONDITIONAL GO**. Before Stage 2, run a same-rank deterministic
random-Φ control at λ=0.75 and repeat the λ={0, 0.75, 1} comparison for at
least one additional seed. Stage 2 adaptive-lambda experiments should not start
until those controls confirm the learned-basis effect and response ordering.
These seed-42 results do not establish statistical significance.

Reproduce the analysis from the repository root with:

```bash
python3 paper/plots_paper/lmss_geo/analyze_stage1_fixed_lambda.py
```

Generated tables and plots are in `stage1_fixed_lambda/analysis/`.

## Stage 1 mechanism validation

The seed-42 learned-Φ λ=0.75 calibration run was compared with the frozen
round × client × layer norm-matched scalar-shrinkage control. Across controlled
rounds 1–6, all 48 round/client/layer step counts matched exactly. The global
raw- and retained-energy differences were 0.37% and 0.41%; the largest
cell-level energy difference was 3.89%, and the maximum retained-energy-fraction
error was 6.88e-8. The control therefore matched the intended realized energy
budget closely enough for the mechanism comparison.

At round 6, learned-Φ obtained 51.96% accuracy and 2.94% mean forgetting,
versus 52.19% and 3.14% for shrinkage. Learned-Φ had lower forgetting in every
controlled round, but shrinkage had higher accuracy and AULC in every round;
final divergence and β̂/ρ̂ were effectively equal. Neither method is
Pareto-dominant. The older deterministic λ=0.75 run exactly reproduced the
calibration run's non-timing summary metrics, so the observed paired differences
exceed deterministic rerun variation, but this one-seed result does not establish
statistical reliability. A further caveat is that the nominally no-op `s=1`
shrinkage path was already 0.03 percentage points higher in round-0 accuracy;
therefore a small finite-precision/control-path perturbation cannot be excluded
from the later trajectory separation.

Verdict: **INCONCLUSIVE**. Directional projection is not equivalent to scalar
step shrinkage, but learned Φ does not yet show a superior
stability–plasticity trade-off. Stage 2 should not begin yet. The single next
experiment is a norm-matched random-Φ control at the same λ, seed, and protocol
to test whether the learned directions outperform arbitrary directional
restriction.

Reproduce the analysis from the repository root with:

```bash
python3 paper/plots_paper/lmss_geo/analyze_stage1_mechanism.py
```

Generated tables and plots are in `stage1_fixed_lambda/analysis_mechanism/`.
