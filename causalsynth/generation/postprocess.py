"""Post-processing of synthetic data.

Adjusts the raw SCM output to better match the original data's column
properties without altering the causal structure:

1. Round columns that were integer-valued in the real data.
2. Clip values to [min - margin, max + margin] to prevent unrealistic extremes.
3. Preserve column order from real data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from causalsynth.data.schema import CausalDAG

logger = logging.getLogger("causalsynth.generation.postprocess")

# Extra margin beyond the real data range (10 % of range)
CLIP_MARGIN_FRACTION = 0.10


def postprocess(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    dag: CausalDAG,
) -> pd.DataFrame:
    """Post-process synthetic data to match original column properties.

    Operations applied in order:
    1. Round integer columns (inferred from real data dtypes).
    2. Clip values to [min, max] + 10% margin from the real data.
    3. Restore original column order.

    Args:
        synthetic: Raw synthetic DataFrame from sampler.
        real: Original real DataFrame used for calibration.
        dag: The causal DAG (used to identify relevant columns).

    Returns:
        Post-processed synthetic DataFrame.
    """
    result = synthetic.copy()

    for col in result.columns:
        if col not in real.columns:
            continue

        real_col = real[col]
        real_min = float(real_col.min())
        real_max = float(real_col.max())
        data_range = real_max - real_min
        margin = data_range * CLIP_MARGIN_FRACTION

        # Clip to real data range + margin
        clip_lo = real_min - margin
        clip_hi = real_max + margin
        result[col] = result[col].clip(lower=clip_lo, upper=clip_hi)

        # Round integers
        if pd.api.types.is_integer_dtype(real_col):
            result[col] = result[col].round().astype(real_col.dtype)
            logger.debug("Rounded integer column '%s'.", col)

    # Restore column order to match real data (only columns present in both)
    ordered_cols = [c for c in real.columns if c in result.columns]
    extra_cols = [c for c in result.columns if c not in ordered_cols]
    result = result[ordered_cols + extra_cols]

    logger.debug(
        "Post-processing complete: %d rows, %d columns.",
        len(result),
        len(result.columns),
    )
    return result


def infer_integer_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that appear to be integer-valued in the DataFrame.

    A column is considered integer if:
    - Its dtype is an integer dtype, or
    - All non-null values are equal to their rounded counterparts.

    Args:
        df: DataFrame to inspect.

    Returns:
        List of column names that are integer-valued.
    """
    integer_cols = []
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            integer_cols.append(col)
        elif pd.api.types.is_float_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0 and np.allclose(non_null, non_null.round(), atol=1e-6):
                integer_cols.append(col)
    return integer_cols
