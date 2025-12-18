"""
Internal machinery for arbitrary schedule adapters.

This module provides the proxy mechanism that allows PyTorch schedulers to control
arbitrary optimizer parameters (not just learning rate) by presenting a "fake"
optimizer where setting lr actually sets the target parameter in the real optimizer.

This is internal infrastructure - users interact with the public API in infrastructure.py.
"""

import warnings
from collections import UserDict
from typing import Optional, Any, Callable, Dict

from torch.optim import Optimizer
from torch.optim.optimizer import StateDict


# ================================================================================
# Error Control
# ================================================================================

THROW_ERROR_ON_DESYNC = True


def throw_errors_on_desync(flag: bool):
    """
    Control whether to throw errors when proxy and backend become desynced.

    Args:
        flag: If True, throw RuntimeError on desync. If False, emit warning.
    """
    global THROW_ERROR_ON_DESYNC
    THROW_ERROR_ON_DESYNC = flag


# ================================================================================
# Proxy Dictionary with Per-Schedule Namespaces
# ================================================================================


class ProxyDictByLR(UserDict):
    """
    Specialized proxy dictionary that presents a target parameter as 'lr'.

    When a scheduler sets 'lr', this proxy actually sets the target parameter
    in the backing dictionary. This allows any PyTorch scheduler to control
    any optimizer parameter.

    Handles namespace collision: Multiple schedules want to set 'initial_lr',
    but there can only be one in the dict. Solution: Per-schedule namespaces
    stored in 'schedule_namespaces' in the backing dict.

    Args:
        proxy_key: The parameter to proxy as 'lr' (e.g., 'weight_decay')
        dictionary: The backing optimizer param_group dict
    """

    def __init__(self, proxy_key: str, dictionary: Dict[str, Any]):
        # Validate proxy key exists
        if proxy_key not in dictionary:
            raise KeyError(f"Proxy key '{proxy_key}' not found in dictionary")

        # Initialize with lr pointing to target parameter
        proxy_dictionary = {"lr": dictionary[proxy_key]}
        super().__init__(proxy_dictionary)

        # Backend references
        self.dictionary = dictionary
        self.proxy_key = proxy_key
        self.existing_keys = list(dictionary.keys())
        self._initialized = True

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set item with namespace routing.

        - If key is 'lr': Update target parameter in backing dict
        - If key existed originally: Error (can't overwrite original params)
        - Otherwise: Store in per-schedule namespace to avoid collisions
        """
        # Handle initialization phase
        if not hasattr(self, "_initialized"):
            return super().__setitem__(key, value)

        # Main logic
        if key == "lr":
            # Check for desync
            if self["lr"] != self.dictionary[self.proxy_key]:
                if THROW_ERROR_ON_DESYNC:
                    msg = (
                        "Proxy and backend have become desynced. "
                        "This indicates schedules were set up incorrectly. "
                        "To disable this error, use throw_errors_on_desync(False)"
                    )
                    raise RuntimeError(msg)
                else:
                    msg = (
                        "Backend optimizer state modified without going through proxy. "
                        "Are schedulers hooked up correctly?"
                    )
                    warnings.warn(msg)

            # Update both proxy and backend
            self.dictionary[self.proxy_key] = value
            super().__setitem__(key, value)

        elif key in self.existing_keys:
            # Can't overwrite original optimizer parameters
            raise KeyError(
                f"Key '{key}' existed in original optimizer dict and cannot be overwritten by schedule"
            )

        else:
            # Store in per-schedule namespace to avoid collisions
            # Multiple schedules each want to set 'initial_lr', etc.
            if "schedule_namespaces" not in self.dictionary:
                self.dictionary["schedule_namespaces"] = {}
            if self.proxy_key not in self.dictionary["schedule_namespaces"]:
                self.dictionary["schedule_namespaces"][self.proxy_key] = {}

            self.dictionary["schedule_namespaces"][self.proxy_key][key] = value
            super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        """
        Get item with namespace routing.

        - If key is 'lr': Return current target parameter value
        - If key existed originally: Get from backing dict
        - Otherwise: Get from per-schedule namespace
        """
        if not hasattr(self, "_initialized"):
            return super().__getitem__(key)

        if key == "lr":
            # Always get latest value from backend
            return self.dictionary[self.proxy_key]

        elif key in self.existing_keys:
            # Get from backing dict
            return self.dictionary[key]

        else:
            # Get from per-schedule namespace
            if "schedule_namespaces" in self.dictionary:
                if self.proxy_key in self.dictionary["schedule_namespaces"]:
                    if key in self.dictionary["schedule_namespaces"][self.proxy_key]:
                        return self.dictionary["schedule_namespaces"][self.proxy_key][key]

            # Fallback to proxy dict
            return super().__getitem__(key)


# ================================================================================
# Arbitrary Schedule Adapter
# ================================================================================


class ArbitraryScheduleAdapter(Optimizer):
    """
    Adapter that makes an optimizer schedulable on arbitrary parameters.

    This is NOT a real optimizer - it's a stub that PyTorch schedulers can use.
    When a scheduler sets 'lr' on this adapter, it actually sets the target
    parameter on the real optimizer via ProxyDictByLR.

    Users should not instantiate this directly - use arbitrary_schedule_factory instead.

    Args:
        optimizer: The real optimizer to wrap
        schedule_target: The parameter to make schedulable (e.g., 'weight_decay')
        default_value: If parameter doesn't exist, initialize to this value
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedule_target: str,
        default_value: Optional[float] = None,
    ) -> None:
        self.optimizer = optimizer
        self.schedule_target = schedule_target

        # Ensure parameter exists (extend if needed)
        if default_value is not None:
            from .infrastructure import extend_optimizer

            self.optimizer = extend_optimizer(
                optimizer, schedule_target, default_value, overwrite_values=False
            )

        # Create proxy param_groups
        self.param_groups = [
            ProxyDictByLR(schedule_target, param_dict)
            for param_dict in self.optimizer.param_groups
        ]

    # Stub methods - this is not a real optimizer
    def step(self, closure: Optional[Callable[[], float]] = None):
        raise NotImplementedError(
            "ArbitraryScheduleAdapter is a stub. Use arbitrary_schedule_factory to create schedules."
        )

    def load_state_dict(self, state_dict: StateDict) -> None:
        raise NotImplementedError(
            "ArbitraryScheduleAdapter is a stub. Use arbitrary_schedule_factory to create schedules."
        )

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "ArbitraryScheduleAdapter is a stub. Use arbitrary_schedule_factory to create schedules."
        )
