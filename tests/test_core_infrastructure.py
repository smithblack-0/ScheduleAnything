"""
Black box tests for core infrastructure behavior.

Tests the observable behavior of the complete infrastructure system:
- Namespace collision prevention (multiple schedules coexist)
- Multi-schedule state independence
- Error detection and handling
- Factory + SynchronousSchedule integration

All tests verify observable behavior only - no internal state checking.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR

import src.torch_schedule_anything as sa


# =============================================================================
# Namespace Collision Prevention Tests
# =============================================================================


def test_schedule_namespaces_routing_contract(optimizer):
    """
    Contract: Scheduler internal state routes to schedule_namespaces, not main param_group.
    Why: Prevents multiple schedules from clobbering each other's internal state (e.g., 'initial_lr').
    How: Verify 'initial_lr' exists in schedule_namespaces[weight_decay], NOT in main dict keys.
    """
    # Create scheduler - StepLR will set 'initial_lr' internally
    scheduler = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=10, gamma=0.5),
        schedule_target="weight_decay",
    )

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


def test_multiple_schedules_maintain_separate_state(optimizer, setup_schedule):
    """
    Contract: Multiple schedules coexist without state collision.
    Why: Each schedule sets internal state (e.g., 'initial_lr') - must not clobber each other.
    How: Run two schedules for 20 steps, verify both parameters evolved independently.
    """
    # Create two schedules on different parameters
    wd_scheduler = setup_schedule(optimizer, "weight_decay", StepLR, step_size=10, gamma=0.5)
    mom_scheduler = setup_schedule(optimizer, "momentum", StepLR, default_value=0.9, step_size=5, gamma=0.8)

    sync = sa.SynchronousSchedule([wd_scheduler, mom_scheduler])

    # Step 20 times
    for _ in range(20):
        sync.step()

    # Observable: Each parameter evolved independently per its schedule
    # weight_decay: step_size=10, so changed at step 10, 20
    # momentum: step_size=5, so changed at steps 5, 10, 15, 20
    final_wd = optimizer.param_groups[0]["weight_decay"]
    final_mom = optimizer.param_groups[0]["momentum"]

    assert final_wd != 0.01  # Changed from initial
    assert final_mom != 0.9  # Changed from initial


def test_three_schedules_coexist(optimizer, setup_schedule):
    """
    Contract: System supports multiple simultaneous schedules without conflicts.
    Why: Ensures namespace isolation scales beyond two schedules.
    How: Run three schedules together for 10 steps, verify all parameters evolved.
    """
    # Three different schedules
    lr_sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: ExponentialLR(opt, gamma=0.95),
        schedule_target="lr",
    )

    wd_sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: ExponentialLR(opt, gamma=0.9),
        schedule_target="weight_decay",
    )

    custom_sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: ExponentialLR(opt, gamma=0.85),
        schedule_target="custom_param",
    )

    # Observable: All three coordinate successfully
    sync = sa.SynchronousSchedule([lr_sched, wd_sched, custom_sched])

    for _ in range(10):
        sync.step()

    # Observable: All three parameters evolved
    assert optimizer.param_groups[0]["lr"] != 0.001
    assert optimizer.param_groups[0]["weight_decay"] != 0.01
    assert optimizer.param_groups[0]["custom_param"] != 5.0


def test_schedules_with_different_step_sizes_dont_interfere():
    """
    Contract: Schedules with different stepping behavior work independently.
    Observable: Each schedule's step logic applies correctly without interference.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    sa.extend_optimizer(optimizer, "momentum", default_value=1.0)

    # Different step sizes
    wd_sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=3, gamma=0.5),
        schedule_target="weight_decay",
    )

    mom_sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=7, gamma=0.7),
        schedule_target="momentum",
    )

    sync = sa.SynchronousSchedule([wd_sched, mom_sched])

    initial_wd = optimizer.param_groups[0]["weight_decay"]
    initial_mom = optimizer.param_groups[0]["momentum"]

    # Step to 7 (wd should change at 3 and 6, mom at 7)
    for _ in range(7):
        sync.step()

    wd_at_7 = optimizer.param_groups[0]["weight_decay"]
    mom_at_7 = optimizer.param_groups[0]["momentum"]

    # Observable: weight_decay changed (stepped at 3 and 6)
    assert wd_at_7 != initial_wd
    # Observable: momentum changed (stepped at 7)
    assert mom_at_7 != initial_mom

    # Continue to 14
    for _ in range(7):
        sync.step()

    # Observable: Both continue evolving independently
    assert optimizer.param_groups[0]["weight_decay"] != wd_at_7
    assert optimizer.param_groups[0]["momentum"] != mom_at_7


# =============================================================================
# State Independence Tests
# =============================================================================


def test_state_dict_preserves_separate_schedule_states():
    """
    Contract: State dict save/load preserves each schedule's independent state.
    Observable: After load, stepping produces correct values for each schedule.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    sa.extend_optimizer(optimizer, "momentum", default_value=0.9)

    wd_sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=5, gamma=0.5),
        schedule_target="weight_decay",
    )

    mom_sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=3, gamma=0.7),
        schedule_target="momentum",
    )

    sync = sa.SynchronousSchedule([wd_sched, mom_sched])

    # Step to 10
    for _ in range(10):
        sync.step()

    # Save checkpoint (both optimizer and scheduler)
    checkpoint = {
        'optimizer': optimizer.state_dict(),
        'scheduler': sync.state_dict()
    }

    # Continue to get reference value at step 11
    sync.step()
    wd_at_11 = optimizer.param_groups[0]["weight_decay"]
    mom_at_11 = optimizer.param_groups[0]["momentum"]

    # Create new optimizer and schedulers
    new_model = nn.Linear(10, 1)
    new_optimizer = AdamW(new_model.parameters(), lr=0.001, weight_decay=0.01)
    sa.extend_optimizer(new_optimizer, "momentum", default_value=0.9)

    new_wd_sched = sa.arbitrary_schedule_factory(
        new_optimizer,
        lambda opt: StepLR(opt, step_size=5, gamma=0.5),
        schedule_target="weight_decay",
    )

    new_mom_sched = sa.arbitrary_schedule_factory(
        new_optimizer,
        lambda opt: StepLR(opt, step_size=3, gamma=0.7),
        schedule_target="momentum",
    )

    new_sync = sa.SynchronousSchedule([new_wd_sched, new_mom_sched])

    # Load checkpoint
    new_optimizer.load_state_dict(checkpoint['optimizer'])
    new_sync.load_state_dict(checkpoint['scheduler'])

    # Step once from checkpoint (should be step 11)
    new_sync.step()

    # Observable: Values match step 11 from original run
    assert abs(new_optimizer.param_groups[0]["weight_decay"] - wd_at_11) < 1e-6
    assert abs(new_optimizer.param_groups[0]["momentum"] - mom_at_11) < 1e-6


# =============================================================================
# Factory Integration Tests
# =============================================================================


def test_factory_with_lambda_scheduler():
    """
    Contract: Factory works with LambdaLR for custom curves.
    Observable: Formula is initial_hyperparameter_value * lambda(t).
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    base_wd = optimizer.param_groups[0]["weight_decay"]  # 0.01

    # Custom lambda: constant multiplier of 2.0
    scheduler = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda=lambda t: 2.0),
        schedule_target="weight_decay",
    )

    # Observable: After init, value = base_wd * lambda(0) = 0.01 * 2.0 = 0.02
    assert abs(optimizer.param_groups[0]["weight_decay"] - base_wd * 2.0) < 1e-6

    scheduler.step()

    # Observable: After step, value = base_wd * lambda(1) = 0.01 * 2.0 = 0.02 (constant lambda)
    assert abs(optimizer.param_groups[0]["weight_decay"] - base_wd * 2.0) < 1e-6


def test_factory_default_schedule_target_is_lr():
    """
    Contract: factory's schedule_target defaults to 'lr'.
    Observable: Omitting schedule_target controls learning rate.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Omit schedule_target - should default to 'lr'
    scheduler = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=1, gamma=0.5),
        # No schedule_target specified
    )

    initial_lr = optimizer.param_groups[0]["lr"]
    initial_wd = optimizer.param_groups[0]["weight_decay"]

    scheduler.step()

    # Observable: lr changed, weight_decay didn't
    assert optimizer.param_groups[0]["lr"] == initial_lr * 0.5
    assert optimizer.param_groups[0]["weight_decay"] == initial_wd


def test_factory_creates_parameter_with_default_value():
    """
    Contract: Factory creates missing parameter if default_value provided.
    Observable: Parameter exists after factory call with default value.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Parameter doesn't exist yet
    assert "custom_param" not in optimizer.param_groups[0]

    scheduler = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=10, gamma=0.5),
        default_value=7.5,
        schedule_target="custom_param",
    )

    # Observable: Parameter now exists with default value
    assert "custom_param" in optimizer.param_groups[0]
    assert optimizer.param_groups[0]["custom_param"] == 7.5


# =============================================================================
# Error Detection Tests
# =============================================================================


def test_synchronous_schedule_detects_duplicate_lr_schedules():
    """
    Contract: SynchronousSchedule raises error for duplicate schedule names.
    Observable: Two 'lr' schedules cause RuntimeError.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Two raw torch schedulers = both named 'lr'
    sched1 = StepLR(optimizer, step_size=10)
    sched2 = StepLR(optimizer, step_size=20)

    # Observable: Raises RuntimeError about duplicate
    with pytest.raises(RuntimeError, match="lr"):
        sa.SynchronousSchedule([sched1, sched2])


def test_synchronous_schedule_allows_different_named_schedules():
    """
    Contract: Different schedule names can coexist.
    Observable: 'lr' + 'weight_decay' schedules work together.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # One for lr, one for weight_decay - different names
    lr_sched = StepLR(optimizer, step_size=10)

    wd_sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=5),
        schedule_target="weight_decay",
    )

    # Observable: No error - different names
    sync = sa.SynchronousSchedule([lr_sched, wd_sched])

    # Observable: Works correctly
    sync.step()
    assert True


def test_extend_optimizer_validates_numeric_types():
    """
    Contract: extend_optimizer validates default_value is numeric.
    Observable: Non-numeric values raise TypeError/ValueError.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001)

    # Observable: String raises error
    with pytest.raises((TypeError, ValueError)):
        sa.extend_optimizer(optimizer, "param", default_value="invalid")

    # Observable: List raises error
    with pytest.raises((TypeError, ValueError)):
        sa.extend_optimizer(optimizer, "param", default_value=[1, 2, 3])


# =============================================================================
# Multiple Param Groups Tests
# =============================================================================


def test_factory_works_with_multiple_param_groups():
    """
    Contract: Factory handles optimizers with multiple param groups.
    Observable: All param groups are scheduled correctly.
    """
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
    optimizer = AdamW(
        [
            {"params": model[0].parameters(), "lr": 0.001, "weight_decay": 0.01},
            {"params": model[1].parameters(), "lr": 0.01, "weight_decay": 0.1},
        ]
    )

    scheduler = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=1, gamma=0.5),
        schedule_target="weight_decay",
    )

    initial_wd_0 = optimizer.param_groups[0]["weight_decay"]
    initial_wd_1 = optimizer.param_groups[1]["weight_decay"]

    scheduler.step()

    # Observable: Both param groups updated
    assert optimizer.param_groups[0]["weight_decay"] == initial_wd_0 * 0.5
    assert optimizer.param_groups[1]["weight_decay"] == initial_wd_1 * 0.5


def test_get_param_groups_regrouped_with_multiple_values():
    """
    Contract: get_param_groups_regrouped_by_key groups by parameter value.
    Observable: Different values create separate groups.
    """
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
    optimizer = AdamW(
        [
            {"params": model[0].parameters(), "lr": 0.001, "weight_decay": 0.01},
            {"params": model[1].parameters(), "lr": 0.01, "weight_decay": 0.1},
        ]
    )

    result = sa.get_param_groups_regrouped_by_key(optimizer, "weight_decay")

    # Observable: Two groups (different weight_decay values)
    assert len(result) == 2

    values = [item[0] for item in result]
    assert 0.01 in values
    assert 0.1 in values


# =============================================================================
# Edge Cases
# =============================================================================


def test_factory_works_with_existing_parameter_no_default():
    """
    Contract: Factory works with existing params without default_value.
    Observable: Can schedule existing parameter without providing default.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # weight_decay already exists - no default_value needed
    scheduler = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=1, gamma=0.5),
        # No default_value
        schedule_target="weight_decay",
    )

    initial_wd = optimizer.param_groups[0]["weight_decay"]

    scheduler.step()

    # Observable: Works correctly
    assert optimizer.param_groups[0]["weight_decay"] == initial_wd * 0.5


def test_extend_optimizer_with_overwrite_values():
    """
    Contract: extend_optimizer overwrites when flag is True.
    Observable: Existing values replaced when overwrite_values=True.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Add parameter
    sa.extend_optimizer(optimizer, "custom", default_value=5.0)
    assert optimizer.param_groups[0]["custom"] == 5.0

    # Overwrite with new value
    sa.extend_optimizer(optimizer, "custom", default_value=10.0, overwrite_values=True)

    # Observable: Value changed
    assert optimizer.param_groups[0]["custom"] == 10.0


def test_extend_optimizer_without_overwrite_preserves():
    """
    Contract: extend_optimizer preserves existing values when overwrite_values=False.
    Observable: Existing value unchanged.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    sa.extend_optimizer(optimizer, "custom", default_value=5.0)
    assert optimizer.param_groups[0]["custom"] == 5.0

    # Try to overwrite with overwrite_values=False
    sa.extend_optimizer(optimizer, "custom", default_value=10.0, overwrite_values=False)

    # Observable: Value preserved
    assert optimizer.param_groups[0]["custom"] == 5.0


def test_desync_detection_raises_error():
    """
    Contract: Desync detection catches when backend is modified directly.
    Observable: RuntimeError raised when proxy and backend are out of sync.

    This test verifies that the internal desync detection works by:
    1. Creating a scheduler with proxy
    2. Modifying the backend parameter directly (bypassing proxy)
    3. Attempting to use the scheduler, which should detect desync
    """
    from src.torch_schedule_anything.arbitrary_schedules import ArbitraryScheduleAdapter

    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Create adapter (internal machinery)
    adapter = ArbitraryScheduleAdapter(optimizer, "weight_decay")

    # Create scheduler on the adapter
    scheduler = LambdaLR(adapter, lambda epoch: 0.95 ** epoch)

    # Step once to establish baseline
    scheduler.step()

    # BYPASS the proxy: Modify backend directly
    # This creates a desync between proxy cache and backend
    optimizer.param_groups[0]["weight_decay"] = 0.999

    # Observable: Next step should detect desync and raise
    with pytest.raises(RuntimeError, match="Proxy and backend have become desynced"):
        scheduler.step()
