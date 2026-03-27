"""Validation report builder and Rich-formatted printer."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from causalsynth.data.schema import CausalDAG, ValidationReport

logger = logging.getLogger("causalsynth.validation.report")
console = Console()

# Verdict thresholds
FIDELITY_PASS = 0.80   # >= 80% edges preserved → PASSED
FIDELITY_WARN = 0.50   # >= 50% → WARNING
# < 50% → FAILED

KS_WARN_THRESHOLD = 0.05  # p < 0.05 in more than half of columns → WARNING


def build_validation_report(
    preserved: list[tuple[str, str]],
    broken: list[tuple[str, str]],
    ks_results: dict[str, float],
    mmd_score: float,
    dag: CausalDAG,
) -> ValidationReport:
    """Build a ValidationReport from component validation results.

    Args:
        preserved: DAG edges found to be statistically preserved.
        broken: DAG edges not preserved in synthetic data.
        ks_results: KS test p-values per column.
        mmd_score: Maximum Mean Discrepancy score.
        dag: The ground-truth DAG.

    Returns:
        A populated ValidationReport.
    """
    n_total = len(dag.edges)
    n_preserved = len(preserved)

    causal_fidelity = float(n_preserved) / float(n_total) if n_total > 0 else 1.0

    # Determine overall verdict
    if causal_fidelity >= FIDELITY_PASS:
        verdict = "PASSED"
        verdict_explanation = (
            f"Causal fidelity is {causal_fidelity:.0%} "
            f"({n_preserved}/{n_total} edges preserved). "
            "The synthetic data faithfully reproduces the causal structure."
        )
    elif causal_fidelity >= FIDELITY_WARN:
        verdict = "WARNING"
        verdict_explanation = (
            f"Causal fidelity is {causal_fidelity:.0%} "
            f"({n_preserved}/{n_total} edges preserved). "
            "Some causal edges were not reproduced; review broken edges."
        )
    else:
        verdict = "FAILED"
        verdict_explanation = (
            f"Causal fidelity is only {causal_fidelity:.0%} "
            f"({n_preserved}/{n_total} edges preserved). "
            "The synthetic data does not adequately preserve causal structure. "
            "Check DAG correctness and sample size."
        )

    # Amend verdict for poor statistical fidelity
    if ks_results:
        low_p_cols = [col for col, p in ks_results.items() if p < KS_WARN_THRESHOLD]
        frac_low = len(low_p_cols) / len(ks_results)
        if frac_low > 0.5 and verdict == "PASSED":
            verdict = "WARNING"
            verdict_explanation += (
                f" However, {len(low_p_cols)}/{len(ks_results)} columns have "
                "significantly different marginal distributions (KS p < 0.05)."
            )

    report = ValidationReport(
        causal_fidelity_score=round(causal_fidelity, 4),
        ks_test_results=ks_results,
        mmd_score=round(mmd_score, 6) if not (mmd_score != mmd_score) else 0.0,
        edges_preserved=preserved,
        edges_broken=broken,
        verdict=verdict,
        verdict_explanation=verdict_explanation,
    )

    logger.info(
        "Validation report built: verdict=%s, fidelity=%.2f, MMD=%.4f",
        verdict,
        causal_fidelity,
        mmd_score,
    )
    return report


def print_validation_report(report: ValidationReport) -> None:
    """Print a Rich-formatted validation report to the console."""
    # Verdict panel
    verdict_color = {
        "PASSED": "bold green",
        "WARNING": "bold yellow",
        "FAILED": "bold red",
    }.get(report.verdict, "bold white")

    console.print(
        Panel(
            f"[{verdict_color}]{report.verdict}[/]\n\n{report.verdict_explanation}",
            title="CausalSynth Validation Report",
            border_style=verdict_color.split()[-1],
        )
    )

    # Summary metrics
    console.print()
    metrics_table = Table(title="Summary Metrics", border_style="cyan")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_column("Interpretation")

    fid = report.causal_fidelity_score
    fid_color = "green" if fid >= FIDELITY_PASS else ("yellow" if fid >= FIDELITY_WARN else "red")
    metrics_table.add_row(
        "Causal Fidelity",
        f"[{fid_color}]{fid:.2%}[/]",
        f"{len(report.edges_preserved)}/{len(report.edges_preserved) + len(report.edges_broken)} DAG edges preserved",
    )

    mmd = report.mmd_score
    mmd_color = "green" if mmd < 0.1 else ("yellow" if mmd < 0.3 else "red")
    metrics_table.add_row(
        "MMD Score",
        f"[{mmd_color}]{mmd:.4f}[/]",
        "Lower is better (0 = identical distributions)",
    )

    if report.ks_test_results:
        avg_ks_p = sum(report.ks_test_results.values()) / len(report.ks_test_results)
        ks_color = "green" if avg_ks_p >= 0.1 else ("yellow" if avg_ks_p >= 0.05 else "red")
        metrics_table.add_row(
            "Avg KS p-value",
            f"[{ks_color}]{avg_ks_p:.3f}[/]",
            "Higher is better (>0.05 = distributions similar)",
        )

    console.print(metrics_table)

    # Edges table
    if report.edges_preserved or report.edges_broken:
        console.print()
        edges_table = Table(title="DAG Edge Preservation", border_style="cyan")
        edges_table.add_column("Cause", style="bold yellow")
        edges_table.add_column("", justify="center")
        edges_table.add_column("Effect", style="bold magenta")
        edges_table.add_column("Status", justify="center")

        for cause, effect in report.edges_preserved:
            edges_table.add_row(cause, "->", effect, "[green]PRESERVED[/]")
        for cause, effect in report.edges_broken:
            edges_table.add_row(cause, "->", effect, "[red]BROKEN[/]")

        console.print(edges_table)

    # Per-column KS results
    if report.ks_test_results:
        console.print()
        ks_table = Table(title="Per-Column KS Tests", border_style="cyan")
        ks_table.add_column("Column", style="bold")
        ks_table.add_column("KS p-value", justify="right")
        ks_table.add_column("Distribution")

        for col, p in sorted(report.ks_test_results.items()):
            color = "green" if p >= 0.1 else ("yellow" if p >= 0.05 else "red")
            interpretation = (
                "Similar" if p >= 0.1 else ("Borderline" if p >= 0.05 else "Different")
            )
            ks_table.add_row(col, f"[{color}]{p:.4f}[/]", interpretation)

        console.print(ks_table)
