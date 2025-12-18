"""
ScheduleAnything: Schedule any optimizer hyperparameter, not just learning rate.

This package provides infrastructure for scheduling arbitrary optimizer parameters
using PyTorch's scheduler interface.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "__version__",
]

try:
    __version__ = version("torch-schedule-anything")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
