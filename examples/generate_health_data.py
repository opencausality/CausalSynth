"""Example: Generate causally-faithful synthetic health data.

Demonstrates the full CausalSynth pipeline:
1. Load a real health dataset
2. Define the causal DAG
3. Fit a Structural Causal Model (SCM) to the real data
4. Generate synthetic data by ancestral sampling
5. Validate that causal structure is preserved

Run:
    python examples/generate_health_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from causalsynth.config import Settings, configure_logging
from causalsynth.dag.loader import load_dag
from causalsynth.generation.postprocess import postprocess
from causalsynth.generation.sampler import generate_samples
from causalsynth.scm.builder import fit_scm
from causalsynth.validation.causal import validate_causal_fidelity
from causalsynth.validation.report import build_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
configure_logging()

REAL_DATA = Path(__file__).parent.parent / "tests" / "fixtures" / "real_health_data.csv"
DAG_FILE = Path(__file__).parent.parent / "tests" / "fixtures" / "health_dag.json"


def main() -> None:
    """Run the full CausalSynth pipeline on health data."""
    settings = Settings()

    # ── Load data ──────────────────────────────────────────────────────────────
    real_df = pd.read_csv(REAL_DATA)
    print(f"Real dataset: {len(real_df)} rows, columns: {list(real_df.columns)}")
    print(f"\nReal data statistics:")
    print(real_df.describe().round(2).to_string())

    # ── Load DAG ───────────────────────────────────────────────────────────────
    dag = load_dag(DAG_FILE)
    print(f"\nCausal DAG: {len(dag.nodes)} nodes, {len(dag.edges)} edges")
    print("Causal structure:")
    for cause, effect in dag.edges:
        print(f"  {cause} ──► {effect}")

    # ── Fit SCM ────────────────────────────────────────────────────────────────
    print("\nFitting Structural Causal Model...")
    scm = fit_scm(real_df, dag, noise_type="gaussian")

    print(f"\nSCM structural equations:")
    for eq in scm.equations:
        if eq.parents:
            terms = " + ".join(f"{v:.3f}×{p}" for p, v in eq.coefficients.items())
            print(f"  {eq.variable} = {eq.intercept:.3f} + {terms} + ε({eq.noise_type}, σ={eq.noise_std:.3f})")
        else:
            print(f"  {eq.variable} = {eq.intercept:.3f} + ε({eq.noise_type}, σ={eq.noise_std:.3f})")

    # ── Generate synthetic data ───────────────────────────────────────────────
    n_synthetic = 1000
    print(f"\nGenerating {n_synthetic:,} synthetic samples (seed=42)...")
    synth_df = generate_samples(scm, n_samples=n_synthetic, seed=42)
    synth_df = postprocess(synth_df, scm)

    print(f"\nSynthetic data statistics:")
    print(synth_df.describe().round(2).to_string())

    # ── Validate ──────────────────────────────────────────────────────────────
    print("\nValidating causal fidelity...")
    val_result = validate_causal_fidelity(real_df, synth_df, dag)
    report = build_report(val_result, dag)

    verdict_symbols = {"PASSED": "✅", "WARNING": "⚠️", "FAILED": "❌"}
    print(f"\n{'═' * 60}")
    print("VALIDATION RESULTS")
    print("═" * 60)
    print(f"Causal fidelity score: {report.causal_fidelity_score:.1%}")
    print(f"MMD score: {report.mmd_score:.4f} (lower is better)")
    print(f"Verdict: {verdict_symbols.get(report.verdict, '')} {report.verdict}")
    print(f"\n{report.verdict_explanation}")

    print(f"\nEdges preserved ({len(report.edges_preserved)}):")
    for cause, effect in report.edges_preserved:
        print(f"  ✅ {cause} → {effect}")

    if report.edges_broken:
        print(f"\nEdges broken ({len(report.edges_broken)}):")
        for cause, effect in report.edges_broken:
            print(f"  ❌ {cause} → {effect}")

    print(f"\nKolmogorov-Smirnov tests (p-value, higher = more similar):")
    for var, p_val in sorted(report.ks_test_results.items()):
        status = "✅" if p_val > 0.05 else "⚠️"
        print(f"  {status} {var}: p={p_val:.4f}")

    # ── Save output ────────────────────────────────────────────────────────────
    output_path = Path("synthetic_health_data.csv")
    synth_df.to_csv(output_path, index=False)
    print(f"\nSynthetic data saved → {output_path}")


if __name__ == "__main__":
    main()
