"""
Tests for low-level adapter/proxy infrastructure.

Tests the adapter and proxy mechanisms that enable scheduling arbitrary parameters:
- ArbitraryScheduleAdapter
- ProxyDictByLR
- Namespace routing
- State independence

These are internal infrastructure tests for the adapter layer.
IMPORTANT: These tests CANNOT use arbitrary_schedule_factory (it's higher-level).
They must use ArbitraryScheduleAdapter directly.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, StepLR

import src.torch_schedule_anything as sa
from src.torch_schedule_anything.arbitrary_schedules import ArbitraryScheduleAdapter

# =============================================================================
# Adapter Contract Tests
# =============================================================================


def test_adapter_cannot_be_used_as_real_optimizer(optimizer):
    """
    Contract: Adapter raises NotImplementedError if used as optimizer.
    Why: Prevents misuse - adapter is a stub for schedulers, not for training loops.
    How: Attempt adapter.step(), verify it raises NotImplementedError.
    """
    # Create adapter
    adapter = ArbitraryScheduleAdapter(optimizer, "weight_decay")

    # Observable: Calling adapter.step() raises NotImplementedError
    # (It's a stub, not a real optimizer)
    with pytest.raises(NotImplementedError, match="stub"):
        adapter.step()


def test_adapter_provides_schedule_target_field(optimizer):
    """
    Contract: Adapter exposes schedule_target as a field.
    Why: Allows inspection of what parameter the adapter is controlling.
    How: Create adapter, verify schedule_target attribute exists and is correct.
    """
    adapter = ArbitraryScheduleAdapter(optimizer, "weight_decay")

    # Observable: schedule_target field exists
    assert hasattr(adapter, "schedule_target")
    assert adapter.schedule_target == "weight_decay"


# =============================================================================
# Namespace Routing Tests
# =============================================================================


def test_namespace_routing_isolates_scheduler_state(optimizer):
    """
    Contract: Scheduler internal state routes to schedule_namespaces, not main param_group.
    Why: Prevents multiple schedules from clobbering each other's internal state
         (e.g., 'initial_lr').
    How: Create adapter+scheduler, verify 'initial_lr' in schedule_namespaces[weight_decay],
         NOT main dict.
    """
    # Create adapter - this sets up the namespace routing
    adapter = ArbitraryScheduleAdapter(optimizer, "weight_decay")

    # Create scheduler on adapter - StepLR will set 'initial_lr' internally
    scheduler = StepLR(adapter, step_size=10, gamma=0.5)

    # Step the scheduler (this causes StepLR to set 'initial_lr')
    scheduler.step()

    param_group = optimizer.param_groups[0]

    # Observable: schedule_namespaces exists and contains the schedule's namespace
    assert "schedule_namespaces" in param_group
    assert "weight_decay" in param_group["schedule_namespaces"]

    # Observable: 'initial_lr' is routed to namespace, not main dict keys
    main_keys = set(param_group.keys()) - {"schedule_namespaces"}
    assert "initial_lr" not in main_keys  # Not polluting main dict

    # Observable: 'initial_lr' is in the weight_decay schedule's namespace
    assert "initial_lr" in param_group["schedule_namespaces"]["weight_decay"]


# =============================================================================
# State Independence Tests
# =============================================================================


def test_multiple_adapters_maintain_separate_state(optimizer):
    """
    Contract: Multiple adapters coexist without state collision.
    Why: Each adapter sets internal state - must not clobber each other.
    How: Create two adapters+schedulers, run 20 steps, verify both parameters evolved independently.
    """
    # Create two adapters on different parameters
    wd_adapter = ArbitraryScheduleAdapter(optimizer, "weight_decay")

    # Add momentum parameter for second adapter
    sa.extend_optimizer(optimizer, "momentum", default_value=0.9)
    mom_adapter = ArbitraryScheduleAdapter(optimizer, "momentum")

    # Create schedulers on each adapter
    wd_scheduler = StepLR(wd_adapter, step_size=10, gamma=0.5)
    mom_scheduler = StepLR(mom_adapter, step_size=5, gamma=0.8)

    # Step both schedulers 20 times
    for _ in range(20):
        wd_scheduler.step()
        mom_scheduler.step()

    # Observable: Each parameter evolved independently per its schedule
    # weight_decay: step_size=10, so changed at step 10, 20
    # momentum: step_size=5, so changed at steps 5, 10, 15, 20
    final_wd = optimizer.param_groups[0]["weight_decay"]
    final_mom = optimizer.param_groups[0]["momentum"]

    assert final_wd != 0.01  # Changed from initial
    assert final_mom != 0.9  # Changed from initial


def test_three_adapters_coexist(optimizer):
    """
    Contract: System supports multiple simultaneous adapters without conflicts.
    Why: Ensures namespace isolation scales beyond two adapters.
    How: Create three adapters+schedulers, run 10 steps, verify all parameters evolved.
    """
    # Three different adapters
    lr_adapter = ArbitraryScheduleAdapter(optimizer, "lr")
    wd_adapter = ArbitraryScheduleAdapter(optimizer, "weight_decay")

    sa.extend_optimizer(optimizer, "custom_param", default_value=5.0)
    custom_adapter = ArbitraryScheduleAdapter(optimizer, "custom_param")

    # Create schedulers
    lr_scheduler = ExponentialLR(lr_adapter, gamma=0.95)
    wd_scheduler = ExponentialLR(wd_adapter, gamma=0.9)
    custom_scheduler = ExponentialLR(custom_adapter, gamma=0.85)

    # Step all schedulers
    for _ in range(10):
        lr_scheduler.step()
        wd_scheduler.step()
        custom_scheduler.step()

    # Observable: All three parameters evolved
    assert optimizer.param_groups[0]["lr"] != 0.001
    assert optimizer.param_groups[0]["weight_decay"] != 0.01
    assert optimizer.param_groups[0]["custom_param"] != 5.0


def test_adapters_with_different_step_sizes_dont_interfere(optimizer):
    """
    Contract: Adapters with different stepping behavior work independently.
    Why: Ensures that different step_size parameters don't interfere across adapters.
    How: Create two adapters with StepLR step_size=3 and step_size=7, verify both apply correctly.
    """
    # Create adapters with different step sizes
    wd_adapter = ArbitraryScheduleAdapter(optimizer, "weight_decay")

    sa.extend_optimizer(optimizer, "momentum", default_value=1.0)
    mom_adapter = ArbitraryScheduleAdapter(optimizer, "momentum")

    # Create schedulers with different step sizes
    wd_scheduler = StepLR(wd_adapter, step_size=3, gamma=0.5)
    mom_scheduler = StepLR(mom_adapter, step_size=7, gamma=0.7)

    initial_wd = optimizer.param_groups[0]["weight_decay"]
    initial_mom = optimizer.param_groups[0]["momentum"]

    # Step to 7 (wd should change at 3 and 6, mom at 7)
    for _ in range(7):
        wd_scheduler.step()
        mom_scheduler.step()

    wd_at_7 = optimizer.param_groups[0]["weight_decay"]
    mom_at_7 = optimizer.param_groups[0]["momentum"]

    # Observable: weight_decay changed (stepped at 3 and 6)
    assert wd_at_7 != initial_wd
    # Observable: momentum changed (stepped at 7)
    assert mom_at_7 != initial_mom

    # Continue to 14
    for _ in range(7):
        wd_scheduler.step()
        mom_scheduler.step()

    # Observable: Both continue evolving independently
    assert optimizer.param_groups[0]["weight_decay"] != wd_at_7
    assert optimizer.param_groups[0]["momentum"] != mom_at_7
