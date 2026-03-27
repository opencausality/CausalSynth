"""CausalSynth custom exceptions.

All exceptions inherit from CausalSynthError so callers can catch the
base class if they want to handle any CausalSynth error uniformly.
"""

from __future__ import annotations


class CausalSynthError(Exception):
    """Base exception for all CausalSynth errors."""


class DAGLoadError(CausalSynthError):
    """Raised when a DAG cannot be loaded or parsed from a file."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load DAG from '{path}': {reason}")


class DAGDiscoveryError(CausalSynthError):
    """Raised when automatic DAG discovery fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"DAG discovery failed: {reason}")


class SCMFitError(CausalSynthError):
    """Raised when fitting a Structural Causal Model to data fails."""

    def __init__(self, variable: str, reason: str) -> None:
        self.variable = variable
        self.reason = reason
        super().__init__(
            f"Failed to fit structural equation for variable '{variable}': {reason}"
        )


class ValidationError(CausalSynthError):
    """Raised when validation of data or schema fails."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Validation error: {reason}")
