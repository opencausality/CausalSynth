"""Command-line interface for CausalSynth."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from causalsynth.config import configure_logging, get_settings

app = typer.Typer(
    name="causalsynth",
    help="Generate causally-faithful synthetic data that preserves your causal DAG's structure.",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger("causalsynth.cli")


# ── generate ──────────────────────────────────────────────────────────────────


@app.command()
def generate(
    data: Path = typer.Argument(..., help="Real dataset CSV file to fit the SCM from."),
    dag: Path = typer.Argument(..., help="Causal DAG JSON file (list of [source, target] edges with 'nodes' key)."),
    output: Path = typer.Argument(..., help="Output path for the synthetic CSV."),
    n_samples: int = typer.Option(1000, "--samples", "-n", help="Number of synthetic rows to generate."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    noise: str = typer.Option("gaussian", "--noise", help="Noise type: gaussian, laplace, or uniform."),
    privacy: bool = typer.Option(False, "--privacy", help="Apply differential privacy noise."),
    epsilon: float = typer.Option(1.0, "--epsilon", help="Privacy budget (only with --privacy)."),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Run causal validation after generation."),
    report_out: Optional[Path] = typer.Option(None, "--report", help="Save validation report JSON."),
) -> None:
    """Fit an SCM to real data and generate synthetic samples."""
    configure_logging()
    settings = get_settings()

    for p in [data, dag]:
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            raise typer.Exit(1)

    import pandas as pd

    from causalsynth.dag.loader import load_dag
    from causalsynth.exceptions import SCMFitError
    from causalsynth.generation.postprocess import postprocess
    from causalsynth.generation.privacy import apply_privacy_noise
    from causalsynth.generation.sampler import generate_samples
    from causalsynth.scm.builder import fit_scm
    from causalsynth.validation.causal import validate_causal_fidelity
    from causalsynth.validation.report import build_report

    df = pd.read_csv(data)
    console.print(f"  Real data: {len(df)} rows, {len(df.columns)} columns")

    causal_dag = load_dag(dag)
    console.print(f"  DAG: {len(causal_dag.nodes)} nodes, {len(causal_dag.edges)} edges")

    # ── Fit SCM ────────────────────────────────────────────────────────────────
    try:
        with console.status("[bold green]Fitting structural causal model..."):
            scm = fit_scm(df, causal_dag, noise_type=noise)
    except SCMFitError as exc:
        console.print(f"[red]SCM fitting failed: {exc}[/red]")
        raise typer.Exit(1)

    console.print(f"  SCM equations: {len(scm.equations)}")

    # ── Sample synthetic data ─────────────────────────────────────────────────
    with console.status(f"[bold green]Generating {n_samples:,} synthetic samples..."):
        synth_df = generate_samples(scm, n_samples=n_samples, seed=seed)

    # ── Post-process ──────────────────────────────────────────────────────────
    synth_df = postprocess(synth_df, scm)

    # ── Privacy ───────────────────────────────────────────────────────────────
    if privacy:
        synth_df = apply_privacy_noise(synth_df, epsilon=epsilon, seed=seed)
        console.print(f"  [dim]Differential privacy applied (ε={epsilon})[/dim]")

    # ── Save output ────────────────────────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(output, index=False)
    console.print(f"  [green]Synthetic data saved → {output}[/green]")

    # ── Validation ────────────────────────────────────────────────────────────
    if validate:
        with console.status("[bold green]Validating causal fidelity..."):
            val_result = validate_causal_fidelity(df, synth_df, causal_dag)
            report = build_report(val_result, causal_dag)

        verdict_color = {"PASSED": "green", "WARNING": "yellow", "FAILED": "red"}[report.verdict]
        console.print(
            Panel(
                f"[bold]Causal fidelity:[/bold] {report.causal_fidelity_score:.1%}\n"
                f"[bold]MMD score:[/bold] {report.mmd_score:.4f} (lower = better)\n"
                f"[bold]Verdict:[/bold] [{verdict_color}]{report.verdict}[/{verdict_color}]\n"
                f"{report.verdict_explanation}",
                title="[bold cyan]Validation Report[/bold cyan]",
            )
        )

        if report_out:
            report_out.write_text(report.model_dump_json(indent=2))
            console.print(f"  [dim]Validation report saved → {report_out}[/dim]")


# ── validate ──────────────────────────────────────────────────────────────────


@app.command()
def validate(
    real: Path = typer.Argument(..., help="Real dataset CSV."),
    synthetic: Path = typer.Argument(..., help="Synthetic dataset CSV."),
    dag: Path = typer.Argument(..., help="Causal DAG JSON file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save validation report JSON."),
) -> None:
    """Compare a synthetic dataset against real data for causal fidelity."""
    configure_logging()

    for p in [real, synthetic, dag]:
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            raise typer.Exit(1)

    import pandas as pd

    from causalsynth.dag.loader import load_dag
    from causalsynth.validation.causal import validate_causal_fidelity
    from causalsynth.validation.report import build_report

    real_df = pd.read_csv(real)
    synth_df = pd.read_csv(synthetic)
    causal_dag = load_dag(dag)

    val_result = validate_causal_fidelity(real_df, synth_df, causal_dag)
    report = build_report(val_result, causal_dag)

    verdict_color = {"PASSED": "green", "WARNING": "yellow", "FAILED": "red"}[report.verdict]
    table = Table(title="Causal Fidelity Validation", show_lines=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Causal fidelity", f"{report.causal_fidelity_score:.1%}")
    table.add_row("MMD score", f"{report.mmd_score:.4f}")
    table.add_row("Edges preserved", str(len(report.edges_preserved)))
    table.add_row("Edges broken", str(len(report.edges_broken)))
    table.add_row("Verdict", f"[{verdict_color}]{report.verdict}[/{verdict_color}]")
    console.print(table)

    for var, p_val in sorted(report.ks_test_results.items()):
        status = "✅" if p_val > 0.05 else "⚠️"
        console.print(f"  {status} {var}: KS p={p_val:.4f}")

    if output:
        output.write_text(report.model_dump_json(indent=2))
        console.print(f"\n[green]Report saved → {output}[/green]")


# ── show ──────────────────────────────────────────────────────────────────────


@app.command()
def show(
    dag: Path = typer.Argument(..., help="Causal DAG JSON file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save visualization HTML."),
) -> None:
    """Visualize a causal DAG as an interactive HTML graph."""
    configure_logging()

    if not dag.exists():
        console.print(f"[red]File not found: {dag}[/red]")
        raise typer.Exit(1)

    from causalsynth.dag.loader import load_dag
    from causalsynth.graph.visualizer import visualize_dag

    causal_dag = load_dag(dag)
    out_path = output or dag.with_suffix(".html")
    visualize_dag(causal_dag, out_path)
    console.print(f"[green]DAG visualization saved → {out_path}[/green]")


# ── serve ─────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """Start the CausalSynth REST API server."""
    configure_logging()

    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn required: pip install uvicorn[/red]")
        raise typer.Exit(1)

    from causalsynth.api.server import create_app

    console.print(f"[bold cyan]CausalSynth API[/bold cyan] → http://{host}:{port}/docs")
    uvicorn.run(create_app(), host=host, port=port)
