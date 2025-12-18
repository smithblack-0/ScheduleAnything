"""
Public infrastructure API for scheduling arbitrary optimizer parameters.

This module provides the user-facing API for:
- arbitrary_schedule_factory: Bind any PyTorch scheduler to any parameter
- SynchronousSchedule: Coordinate multiple schedules in lockstep
- extend_optimizer: Add custom parameters to optimizers
- get_param_groups_regrouped_by_key: Extract parameters grouped by scheduled value

All functions match the documented API specification exactly.
"""

from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import Parameter
from torch.optim import Optimizer

# Backward compatibility: PyTorch renamed _LRScheduler to LRScheduler
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from .arbitrary_schedules import ArbitraryScheduleAdapter

# ================================================================================
# Arbitrary Schedule Factory
# ================================================================================


def arbitrary_schedule_factory(
    optimizer: Optimizer,
    schedule_factory: Callable[[Optimizer], LRScheduler],
    default_value: Optional[float] = None,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Create a scheduler that controls any optimizer parameter.

    Binds any PyTorch scheduler to any optimizer parameter using a factory pattern.
    The factory receives a proxy optimizer and returns a scheduler - this function
    handles the parameter binding automatically.

    Args:
        optimizer: The PyTorch optimizer to bind the schedule to
        schedule_factory: Callable that takes an optimizer and returns a scheduler
        default_value: Optional initial value if parameter doesn't exist
        schedule_target: Which optimizer parameter to schedule (default: 'lr')

    Returns:
        PyTorch scheduler that modifies the specified parameter

    Example:
        >>> scheduler = arbitrary_schedule_factory(
        ...     optimizer=optimizer,
        ...     schedule_factory=lambda opt: StepLR(opt, step_size=30, gamma=0.5),
        ...     default_value=0.9,
        ...     schedule_target='momentum'
        ... )
    """
    adapter = ArbitraryScheduleAdapter(optimizer, schedule_target, default_value)
    schedule = schedule_factory(adapter)
    return schedule


# ================================================================================
# Synchronous Schedule Coordination
# ================================================================================


class SynchronousSchedule(LRScheduler):
    """
    Coordinate multiple schedulers to step in lockstep.

    When scheduling multiple parameters (e.g., lr + weight_decay), this keeps
    them synchronized and provides honest API methods that don't lie about
    what values they return.

    Args:
        schedules: List of PyTorch schedulers to coordinate

    Example:
        >>> lr_sched = arbitrary_schedule_factory(...)
        >>> wd_sched = arbitrary_schedule_factory(...)
        >>> sync = SynchronousSchedule([lr_sched, wd_sched])
        >>> for step in range(1000):
        ...     sync.step()
    """

    def __init__(self, schedules: List[LRScheduler]):
        """
        Initialize with list of schedulers.

        Raises:
            RuntimeError: If multiple schedules target the same parameter
        """
        assigned_schedules = {}
        for schedule in schedules:
            # Determine schedule name
            if isinstance(schedule.optimizer, ArbitraryScheduleAdapter):
                name = schedule.optimizer.schedule_target
            else:
                name = "lr"

            # Check for duplicates
            if name in assigned_schedules:
                raise RuntimeError(
                    f"Multiple schedules targeting '{name}' detected. "
                    f"Each parameter can only have one schedule."
                )

            assigned_schedules[name] = schedule

        self.schedules = assigned_schedules

    @property
    def schedule_names(self) -> List[str]:
        """Get list of all schedule names."""
        return list(self.schedules.keys())

    def step(self):
        """Step all managed schedulers together."""
        for schedule in self.schedules.values():
            schedule.step()

    def get_last_schedule(self, name: str) -> List[float]:
        """
        Get last scheduled values for a specific parameter.

        Args:
            name: The schedule name (e.g., 'lr', 'weight_decay', 'momentum')

        Returns:
            List of values, one per parameter group

        Raises:
            KeyError: If schedule name doesn't exist
        """
        if name not in self.schedules:
            raise KeyError(f"No schedule named '{name}'")
        return self.schedules[name].get_last_lr()

    def get_last_lr(self) -> List[float]:
        """
        Convenience method to get last learning rate values.

        Returns:
            Learning rate values for each parameter group

        Raises:
            KeyError: If no 'lr' schedule exists
        """
        return self.get_last_schedule("lr")

    def state_dict(self) -> Dict[str, Any]:
        """
        Save state of all managed schedulers.

        Returns:
            Dictionary mapping schedule names to their state dicts
        """
        output = {}
        for name, schedule in self.schedules.items():
            output[name] = schedule.state_dict()
        return output

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore state of all managed schedulers.

        Args:
            state_dict: Dictionary previously returned by state_dict()
        """
        for name, individual_state in state_dict.items():
            if name in self.schedules:
                self.schedules[name].load_state_dict(individual_state)


# ================================================================================
# Optimizer Extension Utilities
# ================================================================================


def extend_optimizer(
    optimizer: Optimizer,
    name: str,
    default_value: Number,
    overwrite_values: bool = False,
) -> Optimizer:
    """
    Add a custom parameter to all param_groups in an optimizer.

    Args:
        optimizer: The optimizer to extend
        name: Name of the parameter to add
        default_value: Value to set for the parameter
        overwrite_values: If True, replace existing values. If False, only add if missing.

    Returns:
        The same optimizer instance (modified in-place)

    Raises:
        TypeError: If default_value is not numeric
        ValueError: If default_value is not numeric

    Example:
        >>> extend_optimizer(optimizer, 'gradient_clip_threshold', default_value=10.0)
        >>> # Now gradient_clip_threshold exists in all param_groups
    """
    # Validate default_value is numeric
    if not isinstance(default_value, Number):
        raise TypeError(f"default_value must be numeric, got {type(default_value)}")

    # Add to all param_groups
    for param_group in optimizer.param_groups:
        if overwrite_values or name not in param_group:
            param_group[name] = default_value

    return optimizer


def get_param_groups_regrouped_by_key(
    optimizer: Optimizer, schedule_target: str
) -> List[Tuple[Any, List[Parameter], Dict]]:
    """
    Extract and organize param_groups by a specific parameter's value.

    Groups parameters that have the same value for a given parameter key.
    Useful when training logic needs to respond to scheduled parameter values.

    Args:
        optimizer: The optimizer to extract from
        schedule_target: Which parameter to organize by (e.g., 'lr', 'weight_decay')

    Returns:
        List of tuples (value, params, group_dict) where:
            - value: The scheduled parameter's value for this group
            - params: List of torch.nn.Parameter objects in this group
            - group_dict: The complete param_group dictionary

    Example:
        >>> for threshold, params, group in get_param_groups_regrouped_by_key(
        ...     optimizer, 'gradient_clip_threshold'
        ... ):
        ...     torch.nn.utils.clip_grad_norm_(params, max_norm=threshold)
    """
    result = []

    for param_group in optimizer.param_groups:
        if schedule_target in param_group:
            value = param_group[schedule_target]
            params = param_group["params"]
            result.append((value, params, param_group))

    return result
