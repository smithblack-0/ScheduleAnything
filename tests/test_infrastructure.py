"""
Black box API tests for ScheduleAnything infrastructure utilities.

Tests the documented contract from infrastructure.md:
- arbitrary_schedule_factory
- extend_optimizer
- get_param_groups_regrouped_by_key

All tests verify observable behavior only, not implementation details.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LambdaLR,
    StepLR,
)

# Backward compatibility: PyTorch renamed _LRScheduler to LRScheduler
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

import src.torch_schedule_anything as sa

# =============================================================================
# arbitrary_schedule_factory Tests
# =============================================================================


def test_factory_returns_scheduler(optimizer):
    """
    Contract: arbitrary_schedule_factory returns a PyTorch LRScheduler instance.
    """
    scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: StepLR(opt, step_size=10),
        schedule_target="weight_decay",
    )
    assert isinstance(scheduler, LRScheduler)


def test_factory_with_existing_parameter(optimizer):
    """
    Contract: Factory works with existing parameters without requiring default_value.
    Observable: weight_decay is scheduled and changes when stepped.
    """
    initial_wd = optimizer.param_groups[0]["weight_decay"]

    scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: StepLR(opt, step_size=1, gamma=0.5),
        schedule_target="weight_decay",
    )

    # Step the scheduler
    scheduler.step()

    # Observable: weight_decay changed
    new_wd = optimizer.param_groups[0]["weight_decay"]
    assert new_wd != initial_wd
    assert new_wd == initial_wd * 0.5  # StepLR with gamma=0.5


def test_factory_creates_new_parameter(optimizer):
    """
    Contract: Factory creates parameter when missing if default_value provided.
    Observable: Parameter exists in param_groups with default_value.
    """
    # Verify parameter doesn't exist initially
    assert "gradient_clip_threshold" not in optimizer.param_groups[0]

    sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: StepLR(opt, step_size=10),
        default_value=10.0,
        schedule_target="gradient_clip_threshold",
    )

    # Observable: Parameter now exists with default value
    assert "gradient_clip_threshold" in optimizer.param_groups[0]
    assert optimizer.param_groups[0]["gradient_clip_threshold"] == 10.0


def test_factory_with_step_lr(optimizer):
    """
    Contract: Compatible with PyTorch StepLR scheduler.
    Observable: Parameter updates according to StepLR schedule.
    """
    scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: StepLR(opt, step_size=2, gamma=0.1),
        schedule_target="weight_decay",
    )

    initial_wd = optimizer.param_groups[0]["weight_decay"]

    # Step once - no change yet (step_size=2)
    scheduler.step()
    assert optimizer.param_groups[0]["weight_decay"] == initial_wd

    # Step twice - now should decay
    scheduler.step()
    expected_wd = initial_wd * 0.1
    assert abs(optimizer.param_groups[0]["weight_decay"] - expected_wd) < 1e-6


def test_factory_with_cosine_annealing_lr(optimizer):
    """
    Contract: Compatible with PyTorch CosineAnnealingLR scheduler.
    Observable: Parameter follows cosine curve.
    """
    scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: CosineAnnealingLR(opt, T_max=10),
        schedule_target="weight_decay",
    )

    initial_wd = optimizer.param_groups[0]["weight_decay"]

    # Step to middle of cosine curve
    for _ in range(5):
        scheduler.step()

    # Observable: Value changed (exact formula not under test here, just that it works)
    assert optimizer.param_groups[0]["weight_decay"] != initial_wd


def test_factory_with_exponential_lr(optimizer):
    """
    Contract: Compatible with PyTorch ExponentialLR scheduler.
    Observable: Parameter decays exponentially.
    """
    scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: ExponentialLR(opt, gamma=0.9),
        schedule_target="weight_decay",
    )

    initial_wd = optimizer.param_groups[0]["weight_decay"]

    scheduler.step()
    expected_wd = initial_wd * 0.9
    assert abs(optimizer.param_groups[0]["weight_decay"] - expected_wd) < 1e-6


def test_factory_with_lambda_lr(optimizer):
    """
    Contract: Compatible with PyTorch LambdaLR for custom curves.
    Observable: Formula is initial_hyperparameter_value * lambda(t).
    """
    base_wd = optimizer.param_groups[0]["weight_decay"]  # 0.01

    scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: LambdaLR(opt, lr_lambda=lambda t: 0.5),
        schedule_target="weight_decay",
    )

    # Observable: After init, value = base_wd * lambda(0) = 0.01 * 0.5 = 0.005
    assert abs(optimizer.param_groups[0]["weight_decay"] - base_wd * 0.5) < 1e-6

    scheduler.step()

    # Observable: After step, value = base_wd * lambda(1) = 0.01 * 0.5 = 0.005 (constant lambda)
    assert abs(optimizer.param_groups[0]["weight_decay"] - base_wd * 0.5) < 1e-6


def test_factory_schedule_actually_updates_parameter(optimizer):
    """
    Contract: Scheduler actually modifies the target parameter over time.
    Observable: Multiple steps cause parameter to change.
    """
    scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: ExponentialLR(opt, gamma=0.9),
        schedule_target="weight_decay",
    )

    values = []
    for _ in range(5):
        values.append(optimizer.param_groups[0]["weight_decay"])
        scheduler.step()

    # Observable: Values are changing
    assert len(set(values)) > 1  # Not all the same


def test_factory_with_lr_target_is_default(optimizer):
    """
    Contract: Default schedule_target is 'lr' when omitted.
    Observable: Learning rate is modified when schedule_target not specified.
    """
    scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: StepLR(opt, step_size=1, gamma=0.5),
        # schedule_target omitted - should default to 'lr'
    )

    initial_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    new_lr = optimizer.param_groups[0]["lr"]

    # Observable: lr changed, weight_decay didn't
    assert new_lr == initial_lr * 0.5
    assert optimizer.param_groups[0]["weight_decay"] == 0.01  # Unchanged


def test_factory_with_multiple_param_groups(optimizer_with_multiple_param_groups):
    """
    Contract: Factory works with optimizers containing multiple param_groups.
    Observable: All param_groups are affected.
    """
    opt = optimizer_with_multiple_param_groups

    scheduler = sa.arbitrary_schedule_factory(
        optimizer=opt,
        schedule_factory=lambda o: StepLR(o, step_size=1, gamma=0.5),
        schedule_target="weight_decay",
    )

    initial_wd_0 = opt.param_groups[0]["weight_decay"]
    initial_wd_1 = opt.param_groups[1]["weight_decay"]

    scheduler.step()

    # Observable: Both param_groups updated
    assert opt.param_groups[0]["weight_decay"] == initial_wd_0 * 0.5
    assert opt.param_groups[1]["weight_decay"] == initial_wd_1 * 0.5


def test_factory_throws_invalid_target(optimizer):
    """Test when trying to construct on an invalid target we throw"""

    with pytest.raises(KeyError):
        sa.arbitrary_schedule_factory(
            optimizer=optimizer,
            schedule_factory=lambda opt: StepLR(opt, step_size=1, gamma=0.5),
            schedule_target="burble",
        )


# =============================================================================
# extend_optimizer Tests
# =============================================================================


def test_extend_adds_parameter_to_all_param_groups(optimizer_with_multiple_param_groups):
    """
    Contract: extend_optimizer adds parameter to all param_groups.
    Observable: Parameter exists in every param_group with default_value.
    """
    opt = optimizer_with_multiple_param_groups

    sa.extend_optimizer(opt, "custom_param", default_value=5.0)

    # Observable: All param_groups have the parameter
    assert opt.param_groups[0]["custom_param"] == 5.0
    assert opt.param_groups[1]["custom_param"] == 5.0


def test_extend_does_not_overwrite_existing_by_default(optimizer):
    """
    Contract: extend_optimizer preserves existing values when overwrite_values=False.
    Observable: Existing parameter value unchanged.
    """
    # Manually add parameter to first group
    optimizer.param_groups[0]["custom_param"] = 3.0

    sa.extend_optimizer(optimizer, "custom_param", default_value=5.0, overwrite_values=False)

    # Observable: Existing value preserved
    assert optimizer.param_groups[0]["custom_param"] == 3.0


def test_extend_overwrites_when_flag_true(optimizer):
    """
    Contract: extend_optimizer overwrites existing values when overwrite_values=True.
    Observable: Existing parameter replaced with default_value.
    """
    # Manually add parameter
    optimizer.param_groups[0]["custom_param"] = 3.0

    sa.extend_optimizer(optimizer, "custom_param", default_value=5.0, overwrite_values=True)

    # Observable: Value overwritten
    assert optimizer.param_groups[0]["custom_param"] == 5.0


def test_extend_returns_optimizer(optimizer):
    """
    Contract: extend_optimizer returns the optimizer instance for chaining.
    Observable: Returned object is the same optimizer.
    """
    result = sa.extend_optimizer(optimizer, "custom_param", default_value=1.0)

    # Observable: Same optimizer instance
    assert result is optimizer


def test_extend_validates_default_value_is_number(optimizer):
    """
    Contract: extend_optimizer validates default_value is numeric.
    Observable: Raises error for non-numeric values.
    """
    with pytest.raises((TypeError, ValueError)):
        sa.extend_optimizer(optimizer, "custom_param", default_value="invalid")

    with pytest.raises((TypeError, ValueError)):
        sa.extend_optimizer(optimizer, "custom_param", default_value=[1, 2, 3])


def test_extend_with_different_numeric_types(optimizer):
    """
    Contract: extend_optimizer accepts various numeric types.
    Observable: int, float work correctly.
    """
    # Test with int
    sa.extend_optimizer(optimizer, "param_int", default_value=5)
    assert optimizer.param_groups[0]["param_int"] == 5

    # Test with float
    sa.extend_optimizer(optimizer, "param_float", default_value=3.14)
    assert abs(optimizer.param_groups[0]["param_float"] - 3.14) < 1e-6


# =============================================================================
# get_param_groups_regrouped_by_key Tests
# =============================================================================


def test_regrouped_returns_correct_structure(optimizer):
    """
    Contract: get_param_groups_regrouped_by_key returns list of (value, params, group_dict) tuples.
    Observable: Return type matches documented signature.
    """
    result = sa.get_param_groups_regrouped_by_key(optimizer, "lr")

    # Observable: Returns list
    assert isinstance(result, list)

    # Observable: Each element is a tuple
    assert all(isinstance(item, tuple) for item in result)

    # Observable: Each tuple has 3 elements (value, params, group_dict)
    assert all(len(item) == 3 for item in result)

    # Observable: params is a list, group_dict is a dict
    for value, params, group_dict in result:
        assert isinstance(params, list)
        assert isinstance(group_dict, dict)


def test_regrouped_with_uniform_values(optimizer):
    """
    Contract: Groups parameters by value - uniform values create single group.
    Observable: All param_groups have same lr, returns single tuple with all parameters.
    """
    # All param_groups have lr=0.001
    result = sa.get_param_groups_regrouped_by_key(optimizer, "lr")

    # Observable: Single group (all have same lr)
    assert len(result) == 1

    value, params, group_dict = result[0]
    assert value == 0.001
    assert len(params) > 0  # Has parameters


def test_regrouped_with_different_values(optimizer_with_multiple_param_groups):
    """
    Contract: Separates parameters by different values.
    Observable: Different lr values create separate groups.
    """
    opt = optimizer_with_multiple_param_groups

    result = sa.get_param_groups_regrouped_by_key(opt, "lr")

    # Observable: Two groups (lr=0.001 and lr=0.01)
    assert len(result) == 2

    values = [item[0] for item in result]
    assert 0.001 in values
    assert 0.01 in values


def test_regrouped_with_custom_parameter(optimizer):
    """
    Contract: Works with custom (non-standard) parameters.
    Observable: Custom parameter can be regrouped.
    """
    # Add custom parameter
    sa.extend_optimizer(optimizer, "custom_param", default_value=7.5)

    result = sa.get_param_groups_regrouped_by_key(optimizer, "custom_param")

    # Observable: Returns results for custom parameter
    assert len(result) >= 1
    value, params, group_dict = result[0]
    assert value == 7.5


def test_regrouped_params_are_actual_parameters(optimizer):
    """
    Contract: Returned params are actual torch.nn.Parameter instances.
    Observable: Can be used directly with PyTorch functions.
    """
    result = sa.get_param_groups_regrouped_by_key(optimizer, "lr")

    value, params, group_dict = result[0]

    # Observable: All items in params are Parameters
    assert all(isinstance(p, torch.nn.Parameter) for p in params)


def test_regrouped_group_dict_contains_original_params(optimizer):
    """
    Contract: group_dict contains the complete param_group dictionary.
    Observable: Can access other param_group fields through group_dict.
    """
    result = sa.get_param_groups_regrouped_by_key(optimizer, "lr")

    value, params, group_dict = result[0]

    # Observable: group_dict has expected keys
    assert "lr" in group_dict
    assert "weight_decay" in group_dict
    assert group_dict["lr"] == value


def test_regrouped_with_weight_decay_parameter(optimizer_with_multiple_param_groups):
    """
    Contract: Works with standard PyTorch parameters like weight_decay.
    Observable: Can regroup by weight_decay.
    """
    opt = optimizer_with_multiple_param_groups

    result = sa.get_param_groups_regrouped_by_key(opt, "weight_decay")

    # Observable: Two groups (weight_decay=0.01 and 0.1)
    assert len(result) == 2

    values = [item[0] for item in result]
    assert 0.01 in values
    assert 0.1 in values


def test_regrouped_preserves_all_parameters(optimizer_with_multiple_param_groups):
    """
    Contract: All parameters are included in regrouped output.
    Observable: Total parameter count matches original.
    """
    opt = optimizer_with_multiple_param_groups

    # Count original parameters
    original_param_count = sum(len(g["params"]) for g in opt.param_groups)

    result = sa.get_param_groups_regrouped_by_key(opt, "lr")

    # Count regrouped parameters
    regrouped_param_count = sum(len(params) for _, params, _ in result)

    # Observable: All parameters accounted for
    assert regrouped_param_count == original_param_count
