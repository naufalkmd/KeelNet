"""Compatibility helpers for Hugging Face integrations."""

from __future__ import annotations

import inspect
from typing import Any


def trainer_processing_kwargs(trainer_cls: type, processor: Any) -> dict[str, Any]:
    """Return the right Trainer kwarg for tokenizer/processor compatibility."""

    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": processor}
    if "tokenizer" in parameters:
        return {"tokenizer": processor}
    return {}
