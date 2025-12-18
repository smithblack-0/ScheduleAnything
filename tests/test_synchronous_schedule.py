"""
Black box API tests for SynchronousSchedule coordination.

Tests the documented contract from infrastructure.md:
- SynchronousSchedule constructor and properties
- Stepping and synchronization
- Value retrieval methods
- State dict save/load

All tests verify observable behavior only, not implementation details.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

import src.torch_schedule_anything as sa


# =============================================================================
# Constructor & Properties Tests
# =============================================================================


def test_synchronous_accepts_scheduler_list(optimizer):
    """
    Contract: SynchronousSchedule accepts list of schedulers.
    Observable: Constructs successfully with multiple schedulers.
    """
    sched1 = sa.arbitrary_schedule_factory(
        optimizer, lambda opt: StepLR(opt, step_size=10), schedule_target="weight_decay"
    )
    sched2 = sa.arbitrary_schedule_factory(
        optimizer, lambda opt: ExponentialLR(opt, gamma=0.9), schedule_target="momentum"
    )

    sync = sa.SynchronousSchedule([sched1, sched2])

    # Observable: Constructed successfully
    assert sync is not None


def test_schedule_names_property(optimizer):
    """
    Contract: schedule_names property returns list of schedule names.
    Observable: Names match schedule_target values.
    """
    sched1 = sa.arbitrary_schedule_factory(
        optimizer, lambda opt: StepLR(opt, step_size=10), schedule_target="weight_decay"
    )
    sched2 = sa.arbitrary_schedule_factory(
        optimizer, lambda opt: StepLR(opt, step_size=10), schedule_target="momentum"
    )
    sched3 = sa.arbitrary_schedule_factory(
        optimizer, lambda opt: StepLR(opt, step_size=10), schedule_target="lr"
    )

    sync = sa.SynchronousSchedule([sched1, sched2, sched3])

    # Observable: schedule_names contains expected names
    names = sync.schedule_names
    assert isinstance(names, list)
    assert "weight_decay" in names
    assert "momentum" in names
    assert "lr" in names


def test_schedule_names_lr_for_standard_pytorch(optimizer):
    """
    Contract: Standard PyTorch schedulers (not from factory) are named 'lr'.
    Observable: Raw torch scheduler gets name 'lr'.
    """
    # Standard PyTorch scheduler (not wrapped by factory)
    raw_scheduler = StepLR(optimizer, step_size=10)

    sync = sa.SynchronousSchedule([raw_scheduler])

    # Observable: Name is 'lr'
    assert "lr" in sync.schedule_names


def test_synchronous_rejects_duplicate_lr_schedules(optimizer):
    """
    Contract: SynchronousSchedule raises error when multiple 'lr' schedules detected.
    Observable: Two raw torch schedulers (both 'lr') causes runtime error.

    Note: Factory-created schedules use schedule_target name, avoiding this issue.
    """
    # Two raw torch schedulers = both named 'lr' = conflict
    sched1 = StepLR(optimizer, step_size=10)
    sched2 = CosineAnnealingLR(optimizer, T_max=100)

    # Observable: Raises error about duplicate 'lr'
    with pytest.raises(RuntimeError, match="lr"):
        sa.SynchronousSchedule([sched1, sched2])


# =============================================================================
# Stepping & Synchronization Tests
# =============================================================================


def test_step_updates_all_schedulers(optimizer):
    """
    Contract: Single step() call updates all schedulers.
    Observable: All scheduled parameters change after sync.step().
    """
    # Add parameters for scheduling
    sa.extend_optimizer(optimizer, "custom_param_1", default_value=10.0)
    sa.extend_optimizer(optimizer, "custom_param_2", default_value=20.0)

    sched1 = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=1, gamma=0.5),
        schedule_target="custom_param_1",
    )
    sched2 = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=1, gamma=0.8),
        schedule_target="custom_param_2",
    )

    sync = sa.SynchronousSchedule([sched1, sched2])

    initial_val1 = optimizer.param_groups[0]["custom_param_1"]
    initial_val2 = optimizer.param_groups[0]["custom_param_2"]

    # Step once
    sync.step()

    # Observable: Both parameters changed
    assert optimizer.param_groups[0]["custom_param_1"] == initial_val1 * 0.5
    assert optimizer.param_groups[0]["custom_param_2"] == initial_val2 * 0.8


# =============================================================================
# Value Retrieval Tests
# =============================================================================


def test_get_last_schedule_returns_correct_values(optimizer):
    """
    Contract: get_last_schedule returns actual current values.
    Observable: Returned values match optimizer param_groups.
    """
    sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=1, gamma=0.5),
        schedule_target="weight_decay",
    )

    sync = sa.SynchronousSchedule([sched])

    # Step multiple times
    for _ in range(3):
        sync.step()

    # Observable: get_last_schedule matches param_groups
    last_values = sync.get_last_schedule("weight_decay")
    actual_value = optimizer.param_groups[0]["weight_decay"]

    assert last_values[0] == actual_value


def test_get_last_schedule_returns_list_per_param_group(optimizer_with_multiple_param_groups):
    """
    Contract: Returns one value per param_group.
    Observable: List length equals number of param_groups.
    """
    opt = optimizer_with_multiple_param_groups

    sched = sa.arbitrary_schedule_factory(
        opt, lambda o: StepLR(o, step_size=10), schedule_target="weight_decay"
    )

    sync = sa.SynchronousSchedule([sched])
    sync.step()

    last_values = sync.get_last_schedule("weight_decay")

    # Observable: List has 2 elements (2 param_groups)
    assert len(last_values) == 2
    assert last_values[0] == opt.param_groups[0]["weight_decay"]
    assert last_values[1] == opt.param_groups[1]["weight_decay"]


def test_get_last_lr_convenience_method(optimizer):
    """
    Contract: get_last_lr() is convenience for get_last_schedule('lr').
    Observable: Both methods return same values.
    """
    sched = sa.arbitrary_schedule_factory(
        optimizer, lambda opt: StepLR(opt, step_size=1, gamma=0.9), schedule_target="lr"
    )

    sync = sa.SynchronousSchedule([sched])
    sync.step()

    # Observable: Both methods return same values
    lr_via_convenience = sync.get_last_lr()
    lr_via_generic = sync.get_last_schedule("lr")

    assert lr_via_convenience == lr_via_generic


def test_get_last_schedule_with_invalid_name(optimizer):
    """
    Contract: Validates schedule name exists.
    Observable: Raises error for non-existent schedule.
    """
    sched = sa.arbitrary_schedule_factory(
        optimizer, lambda opt: StepLR(opt, step_size=10), schedule_target="weight_decay"
    )

    sync = sa.SynchronousSchedule([sched])

    # Observable: Error for invalid name
    with pytest.raises((KeyError, ValueError)):
        sync.get_last_schedule("nonexistent_param")


# =============================================================================
# State Dict Tests
# =============================================================================


def test_state_dict_returns_dict(optimizer):
    """
    Contract: state_dict() returns dictionary.
    Observable: Return type is dict.
    """
    sched = sa.arbitrary_schedule_factory(
        optimizer, lambda opt: StepLR(opt, step_size=10), schedule_target="weight_decay"
    )

    sync = sa.SynchronousSchedule([sched])

    state = sync.state_dict()

    # Observable: Returns dict
    assert isinstance(state, dict)


def test_load_state_dict_restores_state(optimizer):
    """
    Contract: State restoration works correctly.
    Observable: Loaded scheduler continues from saved step, values match expected formula.

    Black box approach: Test that resuming from step 500 produces correct value at step 501.
    """
    # Create scheduler with known behavior (StepLR)
    sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=100, gamma=0.5),
        schedule_target="weight_decay",
    )

    sync = sa.SynchronousSchedule([sched])

    # Step to 500
    for _ in range(500):
        sync.step()

    value_at_500 = optimizer.param_groups[0]["weight_decay"]

    # Save state
    state = sync.state_dict()

    # Create new scheduler and optimizer
    new_model = nn.Linear(10, 1)
    new_optimizer = AdamW(new_model.parameters(), lr=0.001, weight_decay=0.01)
    new_sched = sa.arbitrary_schedule_factory(
        new_optimizer,
        lambda opt: StepLR(opt, step_size=100, gamma=0.5),
        schedule_target="weight_decay",
    )
    new_sync = sa.SynchronousSchedule([new_sched])

    # Load state
    new_sync.load_state_dict(state)

    # Observable: Value at step 500 matches
    loaded_value = new_optimizer.param_groups[0]["weight_decay"]
    assert abs(loaded_value - value_at_500) < 1e-6

    # Step once more to 501
    new_sync.step()

    # Observable: Continues correctly (no change yet, step_size=100)
    value_at_501 = new_optimizer.param_groups[0]["weight_decay"]
    assert abs(value_at_501 - value_at_500) < 1e-6


def test_state_dict_roundtrip(optimizer):
    """
    Contract: Roundtrip serialization is stable.
    Observable: save → load → save produces equivalent behavior.
    """
    sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: ExponentialLR(opt, gamma=0.95),
        schedule_target="weight_decay",
    )

    sync = sa.SynchronousSchedule([sched])

    # Step to some point
    for _ in range(50):
        sync.step()

    # First save
    state1 = sync.state_dict()
    value_before_load = optimizer.param_groups[0]["weight_decay"]

    # Load back into same scheduler
    sync.load_state_dict(state1)

    # Observable: Value unchanged after load
    value_after_load = optimizer.param_groups[0]["weight_decay"]
    assert abs(value_after_load - value_before_load) < 1e-6

    # Second save
    state2 = sync.state_dict()

    # Step both forward
    sync.step()
    value_after_step = optimizer.param_groups[0]["weight_decay"]

    # Observable: Behavior continues correctly
    expected_value = value_before_load * 0.95
    assert abs(value_after_step - expected_value) < 1e-6


def test_state_dict_with_multiple_schedules(optimizer):
    """
    Contract: State dict saves all schedulers.
    Observable: Multiple schedulers all resume correctly.
    """
    sa.extend_optimizer(optimizer, "custom_param", default_value=5.0)

    sched1 = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: StepLR(opt, step_size=10, gamma=0.5),
        schedule_target="weight_decay",
    )
    sched2 = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: ExponentialLR(opt, gamma=0.9),
        schedule_target="custom_param",
    )

    sync = sa.SynchronousSchedule([sched1, sched2])

    # Step to 25
    for _ in range(25):
        sync.step()

    wd_at_25 = optimizer.param_groups[0]["weight_decay"]
    cp_at_25 = optimizer.param_groups[0]["custom_param"]

    # Save state
    state = sync.state_dict()

    # New schedulers
    new_model = nn.Linear(10, 1)
    new_optimizer = AdamW(new_model.parameters(), lr=0.001, weight_decay=0.01)
    sa.extend_optimizer(new_optimizer, "custom_param", default_value=5.0)

    new_sched1 = sa.arbitrary_schedule_factory(
        new_optimizer,
        lambda opt: StepLR(opt, step_size=10, gamma=0.5),
        schedule_target="weight_decay",
    )
    new_sched2 = sa.arbitrary_schedule_factory(
        new_optimizer,
        lambda opt: ExponentialLR(opt, gamma=0.9),
        schedule_target="custom_param",
    )

    new_sync = sa.SynchronousSchedule([new_sched1, new_sched2])
    new_sync.load_state_dict(state)

    # Observable: Both parameters match at step 25
    assert abs(new_optimizer.param_groups[0]["weight_decay"] - wd_at_25) < 1e-6
    assert abs(new_optimizer.param_groups[0]["custom_param"] - cp_at_25) < 1e-6


def test_load_state_from_earlier_step(optimizer):
    """
    Contract: State from earlier step can be loaded into scheduler at later step.
    Observable: After loading earlier state, behavior resumes from that earlier point.

    This tests that we can "rewind" a scheduler by loading an earlier checkpoint.
    """
    sched = sa.arbitrary_schedule_factory(
        optimizer,
        lambda opt: ExponentialLR(opt, gamma=0.9),
        schedule_target="weight_decay",
    )

    sync = sa.SynchronousSchedule([sched])

    # Step to 10 and save
    for _ in range(10):
        sync.step()

    state_at_10 = sync.state_dict()
    value_at_10 = optimizer.param_groups[0]["weight_decay"]

    # Continue stepping to 20
    for _ in range(10):
        sync.step()

    value_at_20 = optimizer.param_groups[0]["weight_decay"]

    # Observable: Value changed from step 10 to 20
    assert value_at_20 != value_at_10

    # Load earlier state (step 10)
    sync.load_state_dict(state_at_10)

    # Observable: Value reverted to step 10
    value_after_load = optimizer.param_groups[0]["weight_decay"]
    assert abs(value_after_load - value_at_10) < 1e-6

    # Step once more
    sync.step()

    # Observable: Continues from step 10 (step 11 now)
    value_at_11 = optimizer.param_groups[0]["weight_decay"]
    expected_at_11 = value_at_10 * 0.9
    assert abs(value_at_11 - expected_at_11) < 1e-6
