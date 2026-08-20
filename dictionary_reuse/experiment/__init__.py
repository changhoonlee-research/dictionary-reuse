"""DiR experiment orchestration."""

from .pipeline import run_experiment
from .validation import validate_config

__all__ = ["run_experiment", "validate_config"]
