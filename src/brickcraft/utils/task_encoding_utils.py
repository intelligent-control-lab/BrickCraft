"""Validation helpers for 6D task encodings: x, y, z, orientation, reference type, target type."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


TASK_ENCODING_KEY = "observation.task_encoding"
TASK_ENCODING_DIM = 6


def _feature_shape(feature: Any, *, source: str) -> tuple[int, ...]:
    if isinstance(feature, Mapping):
        shape = feature.get("shape")
    else:
        shape = getattr(feature, "shape", None)
    if shape is None:
        raise ValueError(f"{source} does not define a shape")
    return tuple(int(dim) for dim in shape)


def task_encoding_shape_from_features(
    features: Mapping[str, Any],
    *,
    source: str = "features",
) -> tuple[int, ...] | None:
    """Return and validate the task-encoding feature shape when present."""
    if TASK_ENCODING_KEY not in features:
        return None

    shape = _feature_shape(features[TASK_ENCODING_KEY], source=f"{source}[{TASK_ENCODING_KEY!r}]")
    if shape != (TASK_ENCODING_DIM,):
        raise ValueError(
            f"{source} task encoding must have shape (6,), got {shape}"
        )
    return shape


def validate_task_encoding(
    value: Sequence[float] | np.ndarray,
    *,
    expected_shape: Sequence[int] | None = None,
    source: str = "task_encoding",
) -> np.ndarray:
    """Return a float32 6D vector and optionally match a model shape."""
    encoding = np.asarray(value, dtype=np.float32)
    if encoding.shape != (TASK_ENCODING_DIM,):
        raise ValueError(
            f"{source} must be a 1D vector with 6 elements, got shape {encoding.shape}"
        )
    if not np.isfinite(encoding).all():
        raise ValueError(f"{source} must contain only finite values")

    if expected_shape is not None:
        model_shape = tuple(int(dim) for dim in expected_shape)
        if encoding.shape != model_shape:
            raise ValueError(
                f"{source} dimension mismatch: model expects shape {model_shape}, "
                f"but received shape {encoding.shape}"
            )
    return encoding
