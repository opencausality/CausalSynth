"""Core data models for CausalSynth.

Defines the Pydantic schemas for DAGs, structural equations, SCMs, and
validation reports.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class CausalDAG(BaseModel):
    """A directed acyclic graph representing causal relationships."""

    nodes: list[str] = Field(description="Variable names in the DAG.")
    edges: list[tuple[str, str]] = Field(
        description="Directed causal edges as (cause, effect) pairs."
    )

    @field_validator("nodes")
    @classmethod
    def _nodes_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("DAG must have at least one node.")
        return v

    @model_validator(mode="after")
    def _edges_reference_valid_nodes(self) -> "CausalDAG":
        node_set = set(self.nodes)
        for cause, effect in self.edges:
            if cause not in node_set:
                raise ValueError(f"Edge cause '{cause}' not in nodes list.")
            if effect not in node_set:
                raise ValueError(f"Edge effect '{effect}' not in nodes list.")
        return self


class StructuralEquation(BaseModel):
    """Structural equation for one endogenous variable.

    Encodes: X_i = intercept + sum(coeff_j * X_j for j in parents) + noise
    """

    variable: str = Field(description="The target variable this equation defines.")
    parents: list[str] = Field(
        default_factory=list,
        description="Parent variables (direct causes) of this variable.",
    )
    coefficients: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of parent_name -> linear coefficient.",
    )
    intercept: float = Field(default=0.0, description="Constant offset term.")
    noise_type: str = Field(
        default="gaussian",
        description="Noise distribution: 'gaussian', 'laplace', or 'uniform'.",
    )
    noise_std: float = Field(
        default=1.0,
        gt=0,
        description="Scale parameter for gaussian/laplace noise.",
    )
    noise_range: tuple[float, float] = Field(
        default=(-1.0, 1.0),
        description="[min, max] range for uniform noise.",
    )

    @field_validator("noise_type")
    @classmethod
    def _valid_noise(cls, v: str) -> str:
        if v not in ("gaussian", "laplace", "uniform"):
            raise ValueError(f"Invalid noise_type '{v}'.")
        return v


class SCM(BaseModel):
    """A fitted Structural Causal Model.

    Contains the DAG, one structural equation per variable (in topological
    order), and per-variable statistics for post-processing.
    """

    dag: CausalDAG = Field(description="The causal DAG this SCM is built on.")
    equations: list[StructuralEquation] = Field(
        description="One structural equation per variable, in topological order."
    )
    topological_order: list[str] = Field(
        description="Variables ordered so all parents appear before children."
    )
    feature_stats: dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "Per-variable statistics from the real data "
            "(mean, std, min, max, dtype) for post-processing."
        ),
    )


class ValidationReport(BaseModel):
    """Summary of how well the synthetic data preserves causal structure."""

    causal_fidelity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of DAG edges preserved in synthetic data (0–1).",
    )
    ks_test_results: dict[str, float] = Field(
        description="KS test p-value per variable. High p-value = distributions similar."
    )
    mmd_score: float = Field(
        ge=0.0,
        description=(
            "Maximum Mean Discrepancy between real and synthetic data. "
            "Lower is better."
        ),
    )
    edges_preserved: list[tuple[str, str]] = Field(
        description="DAG edges whose causal signal is present in synthetic data."
    )
    edges_broken: list[tuple[str, str]] = Field(
        description="DAG edges whose causal signal is absent in synthetic data."
    )
    verdict: str = Field(
        description="Overall verdict: 'PASSED', 'WARNING', or 'FAILED'.",
    )
    verdict_explanation: str = Field(
        description="Human-readable explanation of the verdict.",
    )
