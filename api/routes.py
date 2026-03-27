"""CausalSynth REST API route definitions.

Endpoints:
    GET  /health        — Health check
    POST /generate      — Generate synthetic data from a DAG + real data
    POST /validate      — Validate synthetic data against real data + DAG
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from causalsynth import __version__

logger = logging.getLogger("causalsynth.api.routes")

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__
    message: str = "CausalSynth API is running."


class GenerateRequest(BaseModel):
    """Parameters for the /generate endpoint when using JSON body."""
    n_samples: int = Field(default=1000, ge=1, description="Number of synthetic rows.")
    seed: int = Field(default=42, description="Random seed.")
    noise_type: str = Field(default="gaussian", description="Noise distribution.")
    privacy_epsilon: Optional[float] = Field(
        default=None, description="DP epsilon (None = no DP)."
    )


class ValidationSummary(BaseModel):
    verdict: str
    causal_fidelity_score: float
    mmd_score: float
    avg_ks_pvalue: float
    edges_preserved: int
    edges_broken: int
    verdict_explanation: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@router.post("/generate", tags=["Generation"])
async def generate(
    real_csv: UploadFile = File(..., description="Real data CSV file."),
    dag_json: UploadFile = File(..., description="DAG JSON file."),
    n_samples: int = Form(default=1000, ge=1),
    seed: int = Form(default=42),
    noise_type: str = Form(default="gaussian"),
    privacy_epsilon: Optional[float] = Form(default=None),
) -> Response:
    """Generate synthetic data that preserves the causal structure.

    Upload a CSV of real data and a DAG JSON file.  Returns synthetic data
    as a CSV download.

    Args:
        real_csv: Multipart CSV upload of the real data.
        dag_json: Multipart JSON upload of the causal DAG.
        n_samples: Number of synthetic rows to generate.
        seed: Random seed for reproducibility.
        noise_type: One of 'gaussian', 'laplace', 'uniform'.
        privacy_epsilon: DP budget (None = no DP noise).

    Returns:
        CSV file as a streaming download.
    """
    from causalsynth.dag.loader import load_dag
    from causalsynth.generation.postprocess import postprocess
    from causalsynth.generation.privacy import add_differential_privacy
    from causalsynth.generation.sampler import generate_samples
    from causalsynth.scm.builder import fit_scm

    # Read uploaded files
    try:
        real_bytes = await real_csv.read()
        real_df = pd.read_csv(io.BytesIO(real_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse real CSV: {exc}")

    try:
        dag_bytes = await dag_json.read()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(dag_bytes)
            tmp_path = Path(tmp.name)
        dag = load_dag(tmp_path)
        tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse DAG JSON: {exc}")

    # Fit SCM and generate
    try:
        scm = fit_scm(real_df, dag, noise_type=noise_type)
        synthetic_df = generate_samples(scm, n_samples=n_samples, seed=seed)
        synthetic_df = postprocess(synthetic_df, real_df, dag)

        if privacy_epsilon is not None:
            synthetic_df = add_differential_privacy(
                synthetic_df, epsilon=privacy_epsilon
            )
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    # Return as CSV download
    output = io.StringIO()
    synthetic_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=synthetic_{n_samples}rows.csv"
        },
    )


@router.post("/validate", response_model=ValidationSummary, tags=["Validation"])
async def validate(
    real_csv: UploadFile = File(..., description="Real data CSV."),
    synthetic_csv: UploadFile = File(..., description="Synthetic data CSV."),
    dag_json: UploadFile = File(..., description="DAG JSON."),
    significance_level: float = Form(default=0.05),
) -> ValidationSummary:
    """Validate synthetic data against real data and the causal DAG.

    Returns a JSON validation summary including causal fidelity score,
    MMD, KS test results, and an overall verdict.
    """
    from causalsynth.dag.loader import load_dag
    from causalsynth.validation.causal import validate_causal_structure
    from causalsynth.validation.report import build_validation_report
    from causalsynth.validation.statistical import compute_ks_tests, compute_mmd

    try:
        real_df = pd.read_csv(io.BytesIO(await real_csv.read()))
        synth_df = pd.read_csv(io.BytesIO(await synthetic_csv.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV files: {exc}")

    try:
        dag_bytes = await dag_json.read()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(dag_bytes)
            tmp_path = Path(tmp.name)
        dag = load_dag(tmp_path)
        tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse DAG: {exc}")

    try:
        preserved, broken = validate_causal_structure(
            real_df, synth_df, dag, significance_level=significance_level
        )
        ks_results = compute_ks_tests(real_df, synth_df)
        mmd = compute_mmd(real_df, synth_df)
        report = build_validation_report(preserved, broken, ks_results, mmd, dag)
    except Exception as exc:
        logger.exception("Validation failed")
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}")

    avg_ks = sum(ks_results.values()) / len(ks_results) if ks_results else 0.0

    return ValidationSummary(
        verdict=report.verdict,
        causal_fidelity_score=report.causal_fidelity_score,
        mmd_score=report.mmd_score,
        avg_ks_pvalue=avg_ks,
        edges_preserved=len(report.edges_preserved),
        edges_broken=len(report.edges_broken),
        verdict_explanation=report.verdict_explanation,
    )
