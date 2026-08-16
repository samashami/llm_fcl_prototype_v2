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
