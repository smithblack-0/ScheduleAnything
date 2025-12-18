"""
ScheduleAnything: Schedule any optimizer hyperparameter, not just learning rate.

This package provides infrastructure for scheduling arbitrary optimizer parameters
using PyTorch's scheduler interface.
"""

from importlib.metadata import PackageNotFoundError, version

from .infrastructure import (
    arbitrary_schedule_factory,
    SynchronousSchedule,
    extend_optimizer,
    get_param_groups_regrouped_by_key,
)
from .builtin_schedules import (
    cosine_annealing_with_warmup,
    cosine_annealing_with_inverse_warmup,
    polynomial_decay_with_warmup,
    polynomial_decay_with_inverse_warmup,
    linear_decay_with_warmup,
    linear_decay_with_inverse_warmup,
    quadratic_decay_with_warmup,
    quadratic_decay_with_inverse_warmup,
    sqrt_decay_with_warmup,
    sqrt_decay_with_inverse_warmup,
    constant_with_warmup,
    constant_with_inverse_warmup,
    constant_schedule,
)

__all__ = [
    "arbitrary_schedule_factory",
    "SynchronousSchedule",
    "extend_optimizer",
    "get_param_groups_regrouped_by_key",
    "cosine_annealing_with_warmup",
    "cosine_annealing_with_inverse_warmup",
    "polynomial_decay_with_warmup",
    "polynomial_decay_with_inverse_warmup",
    "linear_decay_with_warmup",
    "linear_decay_with_inverse_warmup",
    "quadratic_decay_with_warmup",
    "quadratic_decay_with_inverse_warmup",
    "sqrt_decay_with_warmup",
    "sqrt_decay_with_inverse_warmup",
    "constant_with_warmup",
    "constant_with_inverse_warmup",
    "constant_schedule",
    "__version__",
]

try:
    __version__ = version("torch-schedule-anything")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
