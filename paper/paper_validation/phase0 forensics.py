#!/usr/bin/env python3
"""
Phase 0 forensics (action A0.1/A0.2 of the LMSS revision plan).

Recomputes Table 2 (CIFAR-100) directly from raw per-round training logs and
regenerates Fig. 3 from the SAME dataframe, so the figure and the table cannot
disagree. Then runs four diagnostic checks aimed at the Fig. 3(a) / Table 2
contradiction flagged by the supervisor (comments 0 and 43):

  CHECK 1  Recomputed per-method final-round divergence / AULC / final
           accuracy vs. the values printed in the paper (PAPER_TABLE2 below).
  CHECK 2  Label-swap detector: tests whether swapping the LMSS (local) and
           LMSS (API) series fits the paper's Table 2 better than the current
           labeling, using final-round divergence for both assignments.
  CHECK 3  Arithmetic-compatibility audit: verifies that Table 2 divergence
           is the arithmetic mean of the three final-round seed values.
  CHECK 4  Single-seed vs. multi-seed: prints seed-wise final-round values
           next to the cross-seed Table 2 mean.

USAGE
-----
  # 1) Sanity-check the tooling on synthetic data that reproduces the
  #    suspected bug (swapped LMSS labels):
  python phase0_forensics.py --demo

  # 2) Run on the canonical repository logs (the default):
  python "paper/paper_validation/phase0 forensics.py"

CANONICAL INPUT STRUCTURE
-------------------------
  experiments/results/cifar100/paper_results/
    <method>/seed<seed>/fcl_run_summary_*.csv

  Methods: fixed, v4, lmss, openrouter
  Seeds:   42, 43, 44
  Required columns: round, divergence, global_acc

The older consolidated and <method>_seed<seed>.csv formats remain available
for the synthetic demo and backward-compatible diagnostics.

Accuracy may be either fraction (0-1) or percent (0-100); it is normalised
to percent automatically. Method names are normalised via METHOD_ALIASES.

OUTPUTS (written to ./phase0_out/)
----------------------------------
  table2_recomputed.csv / .md   -- the recomputed Table 2
  fig3_regenerated.pdf / .png   -- Fig. 3 (a) divergence log-scale,
                                   (b) accuracy; same dataframe as the table;
                                   <=4 colourblind-safe colours, distinct
                                   markers, grayscale-readable linestyles
  forensics_report.txt          -- everything printed to the console
"""

import argparse
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Paper constants -- update here if the manuscript changes
# --------------------------------------------------------------------------

# Values as printed in Table 2 of LLM-FCL_v2 (CIFAR-100, mean over seeds 42-44)
PAPER_TABLE2 = {
    "Fixed":        {"divergence": 0.0092, "aulc": 0.6381, "acc": 68.00, "forgetting": 0.0478},
    "V4":           {"divergence": 0.3138, "aulc": 0.6389, "acc": 67.96, "forgetting": 0.0605},
    "LMSS (local)": {"divergence": 0.0676, "aulc": 0.6540, "acc": 69.75, "forgetting": 0.0483},
    "LMSS (API)":   {"divergence": 0.0080, "aulc": 0.6567, "acc": 69.95, "forgetting": 0.0488},
}

METHOD_ALIASES = {
    "fixed": "Fixed", "baseline": "Fixed",
    "v4": "V4", "heuristic": "V4", "heuristic_v4": "V4", "heuristicv4": "V4",
    "lmss": "LMSS (local)", "lmss_local": "LMSS (local)",
    "lmss(local)": "LMSS (local)", "local": "LMSS (local)",
    "qwen": "LMSS (local)", "lmss_qwen": "LMSS (local)",
    "lmss_api": "LMSS (API)", "lmss(api)": "LMSS (API)", "api": "LMSS (API)",
    "openrouter": "LMSS (API)", "lmss_openrouter": "LMSS (API)", "gpt4omini": "LMSS (API)",
}

EXPECTED_SEEDS = (42, 43, 44)
N_ROUNDS = 7          # CIFAR-100: rounds 0..6
TOL_DIV = 0.005       # tolerance when comparing recomputed vs. paper values
TOL_AULC = 0.005
TOL_ACC = 0.5         # percentage points

# Colourblind-safe, grayscale-distinguishable styling (A3.1 conventions)
STYLE = {
    "Fixed":        dict(color="#0072B2", linestyle="--", marker="o"),
    "V4":           dict(color="#D55E00", linestyle=":",  marker="s"),
    "LMSS (local)": dict(color="#009E73", linestyle="-",  marker="D"),
    "LMSS (API)":   dict(color="#CC79A7", linestyle="-.", marker="^"),
}

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_LOGDIR = REPO_ROOT / "experiments" / "results" / "cifar100" / "paper_results"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def norm_method(raw: str) -> str:
    key = re.sub(r"[\s\-]+", "_", str(raw).strip().lower()).replace("__", "_")
    key = key.replace("(", "").replace(")", "")
    return METHOD_ALIASES.get(key, METHOD_ALIASES.get(key.replace("_", ""), str(raw).strip()))


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    required = {"method", "seed", "round", "divergence", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got: {list(df.columns)}")
    df = df[list(required)].copy()
    df["method"] = df["method"].map(norm_method)
    df["seed"] = df["seed"].astype(int)
    df["round"] = df["round"].astype(int)
    df["divergence"] = df["divergence"].astype(float)
    df["accuracy"] = df["accuracy"].astype(float)
    # Normalise accuracy to percent
    if df["accuracy"].max() <= 1.5:
        df["accuracy"] *= 100.0
    return df.sort_values(["method", "seed", "round"]).reset_index(drop=True)


def load_custom(logdir: Path) -> pd.DataFrame | None:
    """Load canonical <method>/seed<seed>/fcl_run_summary_*.csv files."""
    files = sorted(logdir.glob("*/seed*/fcl_run_summary_*.csv"))
    if not files:
        return None

    expected = len(PAPER_TABLE2) * len(EXPECTED_SEEDS)
    if len(files) != expected:
        raise ValueError(f"Expected {expected} canonical summary CSVs, found {len(files)}")

    rows = []
    for f in files:
        seed_match = re.fullmatch(r"seed(\d+)", f.parent.name)
        if seed_match is None:
            raise ValueError(f"Cannot extract seed from directory: {f.parent}")
        df = pd.read_csv(f)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        required = {"round", "divergence", "global_acc"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{f}: missing canonical columns {missing}")
        df = df.rename(columns={"global_acc": "accuracy"})
        df["method"] = f.parent.parent.name
        df["seed"] = int(seed_match.group(1))
        rows.append(df)
    return _tidy(pd.concat(rows, ignore_index=True))


def load_logs(logdir: Path) -> pd.DataFrame:
    custom = load_custom(logdir)
    if custom is not None:
        return custom

    consolidated = logdir / "all_runs.csv"
    if consolidated.exists():
        return _tidy(pd.read_csv(consolidated))

    rows = []
    pat = re.compile(r"^(?P<method>.+?)_seed(?P<seed>\d+)$")
    for f in sorted(logdir.glob("*.csv")):
        m = pat.match(f.stem)
        if not m:
            print(f"  [skip] {f.name} (does not match <method>_seed<seed>.csv)")
            continue
        df = pd.read_csv(f)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        df["method"] = m["method"]
        df["seed"] = int(m["seed"])
        rows.append(df)
    if not rows:
        sys.exit(f"No usable logs found in {logdir}. See INPUT FORMATS in the header.")
    return _tidy(pd.concat(rows, ignore_index=True))


# --------------------------------------------------------------------------
# Recomputation (single source of truth for BOTH table and figure)
# --------------------------------------------------------------------------

def per_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "seed"])
    out = g.agg(mean_divergence=("divergence", "mean"),
                max_divergence=("divergence", "max"),
                final_divergence=("divergence", "last"),
                final_round=("round", "last"),
                aulc=("accuracy", lambda a: a.mean() / 100.0),
                final_acc=("accuracy", "last"),
                n_rounds=("round", "count")).reset_index()
    argmax = g.apply(lambda x: int(x.loc[x["divergence"].idxmax(), "round"]),
                     include_groups=False).rename("argmax_round").reset_index()
    return out.merge(argmax, on=["method", "seed"])


def recompute_table2(seed_summary: pd.DataFrame) -> pd.DataFrame:
    g = seed_summary.groupby("method")
    t = pd.DataFrame({
        "divergence_mean": g["final_divergence"].mean(),
        "divergence_sd": g["final_divergence"].std(ddof=1),
        "aulc_mean": g["aulc"].mean(),
        "aulc_sd": g["aulc"].std(ddof=1),
        "acc_mean": g["final_acc"].mean(),
        "acc_sd": g["final_acc"].std(ddof=1),
        "n_seeds": g["seed"].nunique(),
    })
    order = [m for m in PAPER_TABLE2 if m in t.index] + \
            [m for m in t.index if m not in PAPER_TABLE2]
    return t.loc[order]


def regenerate_fig3(df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    mean_curves = (df.groupby(["method", "round"])
                     .agg(divergence=("divergence", "mean"),
                          accuracy=("accuracy", "mean")).reset_index())
    for method, sub in mean_curves.groupby("method"):
        st = STYLE.get(method, dict(marker="x", linestyle="-"))
        axes[0].plot(sub["round"], sub["divergence"], label=method,
                     linewidth=2, markersize=6, **st)
        axes[1].plot(sub["round"], sub["accuracy"], label=method,
                     linewidth=2, markersize=6, **st)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Federated round"); axes[0].set_ylabel("Divergence over rounds (log scale)")
    axes[0].set_title("(a) Divergence over rounds"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Federated round"); axes[1].set_ylabel("Global accuracy (%)")
    axes[1].set_title("(b) Accuracy"); axes[1].grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outpath.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Forensic checks
# --------------------------------------------------------------------------

def check1_table_match(t2: pd.DataFrame) -> None:
    print("\nCHECK 1 -- recomputed vs. paper Table 2")
    print(f"{'method':<14}{'metric':<24}{'recomputed':>12}{'paper':>10}{'delta':>10}  verdict")
    any_fail = False
    for m, paper in PAPER_TABLE2.items():
        if m not in t2.index:
            print(f"{m:<14}{'-':<12}{'MISSING IN LOGS':>32}")
            any_fail = True
            continue
        rows = [("Final-round divergence", t2.loc[m, "divergence_mean"], paper["divergence"], TOL_DIV),
                ("AULC",       t2.loc[m, "aulc_mean"],       paper["aulc"],       TOL_AULC),
                ("final acc",  t2.loc[m, "acc_mean"],        paper["acc"],        TOL_ACC)]
        for name, rec, pap, tol in rows:
            d = rec - pap
            verdict = "OK" if abs(d) <= tol else "MISMATCH"
            any_fail |= verdict != "OK"
            print(f"{m:<14}{name:<24}{rec:>12.4f}{pap:>10.4f}{d:>+10.4f}  {verdict}")
    print("=> Table 2", "REPRODUCES from logs." if not any_fail
          else "does NOT reproduce -- the printed table is wrong (or logs differ from the reported runs).")


def check2_label_swap(t2: pd.DataFrame) -> None:
    print("\nCHECK 2 -- LMSS label-swap detector (comment 43's prime suspect)")
    a, b = "LMSS (local)", "LMSS (API)"
    if a not in t2.index or b not in t2.index:
        print("  Need both LMSS variants in the logs; skipping.")
        return
    rec = {a: t2.loc[a, "divergence_mean"], b: t2.loc[b, "divergence_mean"]}
    pap = {m: PAPER_TABLE2[m]["divergence"] for m in (a, b)}
    err_identity = abs(rec[a] - pap[a]) + abs(rec[b] - pap[b])
    err_swapped = abs(rec[a] - pap[b]) + abs(rec[b] - pap[a])
    print(f"  identity assignment error  = {err_identity:.4f}")
    print(f"  swapped  assignment error  = {err_swapped:.4f}")
    if err_swapped < err_identity / 2:
        print("  => The SWAPPED assignment fits Table 2 far better: the series")
        print("     labels for LMSS (local) / LMSS (API) are most likely exchanged")
        print("     in the plotting script (or in the log filenames). Fix the labels,")
        print("     regenerate Fig. 3, and re-check which variant the text credits.")
    elif err_identity <= err_swapped:
        print("  => PASS: current labeling fits Table 2 better; a plain swap is")
        print("     not supported by the final-round divergence values.")


def check3_arithmetic(seed_summary: pd.DataFrame) -> None:
    print("\nCHECK 3 -- arithmetic compatibility (final-round seed mean)")
    print("  Table 2 must equal the mean of final-round divergence for seeds 42-44.")
    flagged = False
    for m, rows in seed_summary.groupby("method"):
        if m not in PAPER_TABLE2:
            continue
        rows = rows.sort_values("seed")
        seeds = tuple(int(s) for s in rows["seed"])
        final_rounds = tuple(int(r) for r in rows["final_round"])
        recomputed = float(rows["final_divergence"].mean())
        paper_mean = PAPER_TABLE2[m]["divergence"]
        compatible = (seeds == EXPECTED_SEEDS and
                      all(r == N_ROUNDS - 1 for r in final_rounds) and
                      abs(recomputed - paper_mean) <= TOL_DIV)
        if not compatible:
            flagged = True
            print(f"  INCOMPATIBLE  {m:<14}: seeds={seeds}, final_rounds={final_rounds}, "
                  f"mean={recomputed:.4f}, paper={paper_mean:.4f}.")
    if not flagged:
        print("  PASS: all methods use seeds 42-44 at round 6 and reproduce the paper mean.")


def check4_single_seed(seed_summary: pd.DataFrame, t2: pd.DataFrame) -> None:
    print("\nCHECK 4 -- final-round single-seed vs. multi-seed audit")
    print(f"{'method':<14}" + "".join(f"seed{int(s)} final_div".rjust(19) for s in EXPECTED_SEEDS)
          + "Table 2 mean".rjust(16))
    for m in t2.index:
        vals = []
        for s in EXPECTED_SEEDS:
            row = seed_summary[(seed_summary["method"] == m) & (seed_summary["seed"] == s)]
            vals.append(f"{row['final_divergence'].iloc[0]:.4f}" if len(row) else "-")
        print(f"{m:<14}" + "".join(v.rjust(18) for v in vals)
              + f"{t2.loc[m, 'divergence_mean']:.4f}".rjust(14))
    print("  Table 2 uses the cross-seed mean; Figure 3 separately shows the")
    print("  per-round trajectory averaged across seeds.")


# --------------------------------------------------------------------------
# Demo data -- reproduces the suspected bug so the tooling can be validated
# --------------------------------------------------------------------------

def make_demo(logdir: Path) -> None:
    """Synthesises logs in which the series NAMED 'lmss_api' carries the
    round-5 spike (true mean ~0.067) and the series NAMED 'lmss_local' is the
    stable one (~0.008) -- i.e., labels swapped relative to Table 2, exactly
    the historical situation Fig. 3(a) was suspected to show. Canonical
    validation uses final-round divergence rather than a temporal mean."""
    rng = np.random.default_rng(0)
    logdir.mkdir(parents=True, exist_ok=True)
    rounds = np.arange(N_ROUNDS)
    acc_base = np.array([58.5, 61.0, 62.7, 64.3, 66.2, 67.6, 68.4])

    def curve(div_profile, acc_offset, noise):
        return {s: (np.clip(div_profile + rng.normal(0, noise, N_ROUNDS), 1e-4, None),
                    acc_base + acc_offset + rng.normal(0, 0.4, N_ROUNDS))
                for s in EXPECTED_SEEDS}

    profiles = {
        "fixed":      curve(np.array([.002, .011, .013, .011, .013, .008, .006]), 0.0, .001),
        "v4":         curve(np.array([.002, .15, .25, .42, .28, .52, .57]), -0.2, .005),
        # swapped on purpose:
        "lmss_local": curve(np.array([.008, .007, .004, .0035, .008, .010, .009]), 1.4, .001),
        "lmss_api":   curve(np.array([.075, .007, .0045, .0075, .007, .28, .008]), 1.5, .002),
    }
    for method, seeds in profiles.items():
        for s, (div, acc) in seeds.items():
            pd.DataFrame({"round": rounds, "divergence": div, "accuracy": acc}) \
              .to_csv(logdir / f"{method}_seed{s}.csv", index=False)
    print(f"[demo] synthetic logs written to {logdir}/ (LMSS labels deliberately swapped)")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=CANONICAL_LOGDIR,
                    help="canonical paper_results directory")
    ap.add_argument("--out", type=Path, default=Path("phase0_out"))
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic logs that reproduce the suspected label swap")
    args = ap.parse_args()

    if args.demo:
        args.logs = Path("demo_logs")
        make_demo(args.logs)
    args.out.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s); buf.write(s); return len(s)

    with redirect_stdout(Tee()):
        df = load_logs(args.logs)
        print(f"Loaded {df['method'].nunique()} methods x "
              f"{df['seed'].nunique()} seeds x {df['round'].nunique()} rounds "
              f"({len(df)} rows) from {args.logs}/")

        ss = per_seed_summary(df)
        t2 = recompute_table2(ss)

        print("\nRECOMPUTED TABLE 2 (mean ± sample sd over seeds)")
        for m, r in t2.iterrows():
            print(f"  {m:<14} acc {r.acc_mean:6.2f} ± {r.acc_sd:4.2f}   "
                  f"AULC {r.aulc_mean:.4f} ± {r.aulc_sd:.4f}   "
                  f"Final-round divergence {r.divergence_mean:.4f} ± {r.divergence_sd:.4f}   "
                  f"(n={int(r.n_seeds)})")
        t2.to_csv(args.out / "table2_recomputed.csv")

        check1_table_match(t2)
        check2_label_swap(t2)
        check3_arithmetic(ss)
        check4_single_seed(ss, t2)

        regenerate_fig3(df, args.out / "fig3_regenerated")
        print(f"\nWrote: {args.out}/table2_recomputed.csv, "
              f"{args.out}/fig3_regenerated.pdf/.png")
        print("Figure and table are generated from the SAME dataframe; once your")
        print("real logs pass CHECK 1, paste the regenerated figure into the paper.")

    (args.out / "forensics_report.txt").write_text(buf.getvalue())


if __name__ == "__main__":
    main()
