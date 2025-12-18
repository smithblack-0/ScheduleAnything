"""
Internal machinery for arbitrary schedule adapters.

WHAT THIS PROVIDES:
===================
ArbitraryScheduleAdapter - A stub optimizer that allows PyTorch schedulers to control
arbitrary optimizer parameters (not just 'lr').

OUTPUT:
  ArbitraryScheduleAdapter(optimizer, schedule_target, default_value) -> Stub Optimizer

USAGE:
  adapter = ArbitraryScheduleAdapter(optimizer, "weight_decay")
  scheduler = StepLR(adapter, step_size=30, gamma=0.5)
  scheduler.step()  # Now schedules weight_decay, not lr!

Users don't typically call this directly - they use arbitrary_schedule_factory() from
infrastructure.py, which handles the adapter creation and scheduler factory pattern.

WHAT'S INSIDE:
==============
- ArbitraryScheduleAdapter: Stub optimizer with param_groups that redirect 'lr' access
- ProxyDictByLR: Dictionary proxy that intercepts and redirects 'lr' operations
- set_throw_error_on_desync(): Global control for error vs warning on desync detection

INVARIANTS (Problems This Module Must Handle):
===============================================
- PyTorch schedulers are hardcoded to look for 'lr' - we must intercept and redirect
- Multiple schedules on same optimizer would collide on auxiliary keys (e.g., 'initial_lr')
- Backend modifications bypass proxy - we must detect when cache and backend diverge
- Proxy initialization could recurse infinitely - we must guard against this

See class docstrings (especially ProxyDictByLR) for implementation details.

MAINTAINER NOTES:
=================
The core complexity is in ProxyDictByLR - start there when debugging.
The adapter is just a thin wrapper that creates proxies for each param_group.
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


def set_throw_error_on_desync(flag: bool):
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
    Proxy dictionary that intercepts 'lr' access and redirects to a target parameter.

    This is the core mechanism that allows PyTorch schedulers to control arbitrary
    optimizer parameters. It works by:

    1. Inheriting from UserDict (provides dict-like interface)
    2. Overriding __getitem__ and __setitem__ to intercept access
    3. Redirecting 'lr' operations to a different parameter in the backing dict

    PROXY BEHAVIOR:
    ===============
    Reading:
      proxy["lr"]              → returns CACHED value (checks desync first!)
      proxy["existing_param"]  → returns backend["existing_param"]
      proxy["new_key"]         → returns schedule_namespaces[proxy_key]["new_key"]

    Writing:
      proxy["lr"] = X          → sets BOTH backend[proxy_key] = X AND cache = X
      proxy["existing_param"]  → RAISES KeyError (can't overwrite original params)
      proxy["new_key"] = Y     → sets schedule_namespaces[proxy_key]["new_key"] = Y

    NAMESPACE ISOLATION:
    ====================
    When PyTorch schedulers set auxiliary keys (like 'initial_lr', 'base_lr'),
    they're routed to per-schedule namespaces to prevent collision:

      param_group['schedule_namespaces'][proxy_key][auxiliary_key] = value

    This allows multiple schedules to coexist without interfering with each other.

    DESYNC DETECTION:
    =================
    The proxy caches 'lr' value in self.data["lr"]. On each READ, it checks:
      - Cached value (self.data["lr"]) == Backend value (backend[proxy_key])
      - If they differ, someone bypassed the proxy → RuntimeError

    This catches (on first access after modification):
      - Direct modifications to optimizer.param_groups[i][target]
      - Incorrect scheduler initialization
      - Manual state modifications

    Fails fast: error raised on read, not on next write.
    Disable with: set_throw_error_on_desync(False) to get warnings instead.

    INITIALIZATION GUARD:
    =====================
    The _initialized flag prevents infinite recursion:
      - During __init__, we call super().__setitem__ which may trigger __setitem__
      - Check for _initialized prevents re-entering proxy logic during setup
      - After init completes, all access goes through proxy logic

    BACKEND REFERENCE:
    ==================
    self.dictionary is the ACTUAL optimizer param_group dict (not a copy).
    All modifications are reflected immediately in the optimizer state.
    This is crucial - the proxy doesn't duplicate data, it redirects access.

    Args:
        proxy_key: The parameter name to proxy as 'lr' (e.g., 'weight_decay', 'momentum')
        dictionary: The backing optimizer param_group dict (modified in-place)

    Raises:
        KeyError: If proxy_key doesn't exist in dictionary
        RuntimeError: If desync detected between proxy cache and backend

    Example:
        >>> param_group = {"lr": 0.001, "weight_decay": 0.01}
        >>> proxy = ProxyDictByLR("weight_decay", param_group)
        >>> proxy["lr"] = 0.005  # Actually sets param_group["weight_decay"] = 0.005
        >>> assert param_group["weight_decay"] == 0.005
        >>> assert proxy["lr"] == 0.005
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
        Intercept dictionary writes and route based on key type.

        This method implements the core proxy logic. When a PyTorch scheduler
        writes to the proxy, we route the operation based on what's being set:

        ROUTING LOGIC:
        ==============
        1. key == "lr":
           - Update backend[proxy_key] = value
           - Update cache: self.data["lr"] = value
           - Both must stay in sync for correct operation

        2. key in existing_keys (original param_group keys):
           - FORBIDDEN: Can't overwrite original optimizer parameters
           - Raises KeyError immediately
           - Prevents schedulers from corrupting optimizer state

        3. key is new (auxiliary scheduler state):
           - Route to per-schedule namespace
           - Ensures isolation between multiple schedules
           - Also update cache for consistency

        NOTE: Desync detection happens in __getitem__, not here.
        When reading 'lr', we check if cache != backend and raise.
        This catches backend modifications on first access (fail fast).

        INITIALIZATION GUARD:
        =====================
        During __init__, we haven't set _initialized yet. In that case,
        bypass all proxy logic and use default UserDict behavior.
        This prevents infinite recursion during object construction.

        Args:
            key: The dictionary key being set
            value: The value to set

        Raises:
            KeyError: If trying to overwrite an original param_group key
        """
        # Handle initialization phase
        if not hasattr(self, "_initialized"):
            return super().__setitem__(key, value)

        # Main logic
        if key == "lr":
            # Update both proxy cache and backend
            # Desync detection happens in __getitem__, not here
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
        Intercept dictionary reads and route based on key type.

        This method complements __setitem__ by routing reads appropriately.
        PyTorch schedulers read from the proxy to get current parameter values.

        ROUTING LOGIC:
        ==============
        1. key == "lr":
           - Return CACHED value: super().__getitem__("lr")
           - But first: check for desync (cached != backend)
           - If desynced: raise RuntimeError or warn
           - Proxy presents the cached view, not backend view

        2. key in existing_keys (original param_group keys):
           - Return from backend: self.dictionary[key]
           - These are shared across all schedules
           - Examples: 'params', 'eps', 'betas'

        3. key is new (auxiliary scheduler state):
           - Look in per-schedule namespace first
           - Check: schedule_namespaces[proxy_key][key]
           - Fallback to cached value if not found
           - This isolates scheduler-specific state

        DESYNC DETECTION ON READ:
        ==========================
        Before returning cached 'lr', we check if backend was modified:
          cached = super().__getitem__("lr")
          if cached != self.dictionary[self.proxy_key]:
              raise RuntimeError(...)  # Backend was modified directly!

        This catches backend modifications immediately on first read.
        Fails fast - better than returning stale data or catching on write.

        WHY RETURN CACHED, NOT BACKEND:
        ================================
        The proxy OWNS the value. When scheduler sets lr, we update:
          - Backend (so optimizer sees it)
          - Cache (so proxy remembers it)

        Returning cached value means:
          - Proxy presents consistent view of what it last set
          - Desync check catches when backend diverges
          - Scheduler gets the value it expects (what it set)

        INITIALIZATION GUARD:
        =====================
        During __init__, bypass proxy logic and use UserDict default.
        Prevents infinite recursion during construction.

        Args:
            key: The dictionary key being accessed

        Returns:
            The value associated with the key

        Raises:
            KeyError: If key doesn't exist anywhere (proxy, backend, namespaces)
            RuntimeError: If desync detected and THROW_ERROR_ON_DESYNC is True
        """
        if not hasattr(self, "_initialized"):
            return super().__getitem__(key)

        if key == "lr":
            # Get cached value
            cached = super().__getitem__("lr")

            # Check for desync: has backend been modified directly?
            if cached != self.dictionary[self.proxy_key]:
                if THROW_ERROR_ON_DESYNC:
                    msg = (
                        "Proxy and backend have become desynced. "
                        "Backend was modified directly without going through proxy. "
                        "This indicates schedules were set up incorrectly. "
                        "To disable this error, use set_throw_error_on_desync(False)"
                    )
                    raise RuntimeError(msg)
                else:
                    msg = (
                        "Backend optimizer state modified without going through proxy. "
                        "Are schedulers hooked up correctly?"
                    )
                    warnings.warn(msg)

            return cached

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
