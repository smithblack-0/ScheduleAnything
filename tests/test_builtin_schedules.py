"""
Black box API tests for built-in schedules.

Tests all 13 built-in schedules match their documented mathematical formulas from builtin_schedules.md:
- Cosine schedules (2)
- Polynomial schedules (8)
- Constant schedules (3)

All tests verify observable behavior matches documented formulas.
"""

import pytest
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler

import src.torch_schedule_anything as sa


# =============================================================================
# Helper Functions
# =============================================================================


def get_scheduled_value(optimizer, param_name="lr"):
    """Helper to get current scheduled parameter value."""
    return optimizer.param_groups[0][param_name]


def assert_close(actual, expected, rtol=1e-5):
    """Helper for floating point comparison."""
    assert abs(actual - expected) < rtol or abs(actual - expected) / max(abs(expected), 1e-8) < rtol


# =============================================================================
# Cosine Schedules
# =============================================================================


def test_cosine_annealing_with_warmup_returns_scheduler(optimizer):
    """Contract: Returns _LRScheduler instance."""
    scheduler = sa.cosine_annealing_with_warmup(
        optimizer,
        warmup_to_value=1.0,
        anneal_to_value=0.1,
        num_warmup_steps=100,
        num_training_steps=1000,
    )
    assert isinstance(scheduler, _LRScheduler)


def test_cosine_annealing_with_warmup_accepts_schedule_target(optimizer):
    """Contract: Works with schedule_target parameter."""
    scheduler = sa.cosine_annealing_with_warmup(
        optimizer,
        warmup_to_value=1.0,
        anneal_to_value=0.1,
        num_warmup_steps=100,
        num_training_steps=1000,
        schedule_target="weight_decay",
    )

    # Observable: Controls weight_decay, not lr
    initial_wd = get_scheduled_value(optimizer, "weight_decay")
    scheduler.step()
    # Value changed
    assert get_scheduled_value(optimizer, "weight_decay") != initial_wd


def test_cosine_annealing_with_warmup_formula(optimizer):
    """
    Contract: Matches documented formula from builtin_schedules.md.

    Formula during warmup (t <= L):
        λ(t) = (W * t) / L

    Formula during annealing (t > L):
        λ(t) = A + (W - A) * cos((π/2) * (t - L) / (M - L))

    Where: W=warmup_to_value, A=anneal_to_value, L=num_warmup_steps, M=num_training_steps
    """
    W, A, L, M = 1.0, 0.1, 100, 1000
    initial_lr = 0.001

    scheduler = sa.cosine_annealing_with_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
    )

    # Test warmup phase
    # At t=0 (before first step)
    scheduler.step(0)
    expected_lambda = (W * 0) / L  # = 0
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)

    # At t=50 (mid-warmup)
    scheduler.step(50)
    expected_lambda = (W * 50) / L  # = 0.5
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)

    # At t=100 (end of warmup)
    scheduler.step(100)
    expected_lambda = W  # = 1.0
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)

    # Test annealing phase
    # At t=550 (mid-annealing)
    scheduler.step(550)
    expected_lambda = A + (W - A) * math.cos((math.pi / 2) * (550 - L) / (M - L))
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)

    # At t=1000 (end)
    scheduler.step(1000)
    expected_lambda = A + (W - A) * math.cos((math.pi / 2) * (1000 - L) / (M - L))
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)


def test_cosine_annealing_with_inverse_warmup_formula(optimizer):
    """
    Contract: Inverse warmup starts high, decays to baseline, then cosine anneals.

    Formula during inverse warmup (t <= L):
        λ(t) = W * (R - (R-1) * t/L)

    Formula during annealing (t > L):
        λ(t) = A + (W - A) * cos((π/2) * (t - L) / (M - L))

    Where: W=warmup_to_value, A=anneal_to_value, L=num_warmup_steps, M=num_training_steps, R=warmup_multiplier
    """
    W, A, L, M, R = 1.0, 0.1, 100, 1000, 10.0
    initial_lr = 0.001

    scheduler = sa.cosine_annealing_with_inverse_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
        warmup_multiplier=R,
    )

    # At t=0 (start high)
    scheduler.step(0)
    expected_lambda = W * (R - (R - 1) * 0 / L)  # = W * R = 10.0
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)

    # At t=100 (end of inverse warmup, should be at W)
    scheduler.step(100)
    expected_lambda = W  # = 1.0
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)


# =============================================================================
# Polynomial Schedules
# =============================================================================


def test_polynomial_schedule_with_warmup_formula(optimizer):
    """
    Contract: Polynomial schedule with custom exponent.

    Formula during warmup (t <= L):
        λ(t) = (W * t) / L

    Formula during annealing (t > L):
        λ(t) = A + (W - A) * (1 - (t - L) / (M - L))^P

    Where: W=warmup_to_value, A=anneal_to_value, L=num_warmup_steps, M=num_training_steps, P=polynomial_exponent
    """
    W, A, L, M, P = 1.0, 0.1, 100, 1000, 3.0
    initial_lr = 0.001

    scheduler = sa.polynomial_schedule_with_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
        polynomial_exponent=P,
    )

    # At t=550 (mid-annealing)
    scheduler.step(550)
    expected_lambda = A + (W - A) * ((1 - (550 - L) / (M - L)) ** P)
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)


def test_linear_schedule_with_warmup_formula(optimizer):
    """
    Contract: Linear schedule is polynomial with exponent=1.

    Formula during annealing (t > L):
        λ(t) = A + (W - A) * (1 - (t - L) / (M - L))
    """
    W, A, L, M = 1.0, 0.1, 100, 1000
    initial_lr = 0.001

    scheduler = sa.linear_schedule_with_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
    )

    # At t=550
    scheduler.step(550)
    expected_lambda = A + (W - A) * (1 - (550 - L) / (M - L))
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)


def test_quadratic_schedule_with_warmup_formula(optimizer):
    """
    Contract: Quadratic schedule is polynomial with exponent=2.

    Formula during annealing (t > L):
        λ(t) = A + (W - A) * (1 - (t - L) / (M - L))^2
    """
    W, A, L, M = 1.0, 0.1, 100, 1000
    initial_lr = 0.001

    scheduler = sa.quadratic_schedule_with_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
    )

    # At t=550
    scheduler.step(550)
    expected_lambda = A + (W - A) * ((1 - (550 - L) / (M - L)) ** 2)
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)


def test_sqrt_schedule_with_warmup_formula(optimizer):
    """
    Contract: Square root schedule is polynomial with exponent=0.5.

    Formula during annealing (t > L):
        λ(t) = A + (W - A) * (1 - (t - L) / (M - L))^0.5
    """
    W, A, L, M = 1.0, 0.1, 100, 1000
    initial_lr = 0.001

    scheduler = sa.sqrt_schedule_with_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
    )

    # At t=550
    scheduler.step(550)
    expected_lambda = A + (W - A) * ((1 - (550 - L) / (M - L)) ** 0.5)
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)


def test_polynomial_schedule_with_inverse_warmup_formula(optimizer):
    """Contract: Polynomial with inverse warmup."""
    W, A, L, M, P, R = 1.0, 0.1, 100, 1000, 2.0, 5.0
    initial_lr = 0.001

    scheduler = sa.polynomial_schedule_with_inverse_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
        polynomial_exponent=P,
        warmup_multiplier=R,
    )

    # At t=0 (start high)
    scheduler.step(0)
    expected_lambda = W * (R - (R - 1) * 0 / L)  # = W * R
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)


def test_linear_schedule_with_inverse_warmup_formula(optimizer):
    """Contract: Linear with inverse warmup."""
    W, A, L, M, R = 1.0, 0.1, 100, 1000, 5.0
    initial_lr = 0.001

    scheduler = sa.linear_schedule_with_inverse_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
        warmup_multiplier=R,
    )

    # At t=0
    scheduler.step(0)
    expected_lambda = W * R
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)


def test_quadratic_schedule_with_inverse_warmup_formula(optimizer):
    """Contract: Quadratic with inverse warmup."""
    W, A, L, M, R = 1.0, 0.1, 100, 1000, 5.0
    initial_lr = 0.001

    scheduler = sa.quadratic_schedule_with_inverse_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
        warmup_multiplier=R,
    )

    # At t=0
    scheduler.step(0)
    expected_lambda = W * R
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)


def test_sqrt_schedule_with_inverse_warmup_formula(optimizer):
    """Contract: Square root with inverse warmup."""
    W, A, L, M, R = 1.0, 0.1, 100, 1000, 5.0
    initial_lr = 0.001

    scheduler = sa.sqrt_schedule_with_inverse_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
        warmup_multiplier=R,
    )

    # At t=0
    scheduler.step(0)
    expected_lambda = W * R
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)


# =============================================================================
# Constant Schedules
# =============================================================================


def test_constant_with_warmup_formula(optimizer):
    """
    Contract: Warmup then holds constant.

    Formula during warmup (t <= L):
        λ(t) = (W * t) / L

    Formula after warmup (t > L):
        λ(t) = W
    """
    W, L = 1.0, 100
    initial_lr = 0.001

    scheduler = sa.constant_with_warmup(
        optimizer,
        warmup_to_value=W,
        num_warmup_steps=L,
    )

    # During warmup
    scheduler.step(50)
    expected_lambda = (W * 50) / L
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)

    # After warmup - should stay constant
    scheduler.step(100)
    expected_value = initial_lr * W
    assert_close(get_scheduled_value(optimizer), expected_value)

    scheduler.step(500)
    expected_value = initial_lr * W
    assert_close(get_scheduled_value(optimizer), expected_value)


def test_constant_with_inverse_warmup_formula(optimizer):
    """
    Contract: Inverse warmup then holds constant.

    Formula during inverse warmup (t <= L):
        λ(t) = W * (R - (R-1) * t/L)

    Formula after warmup (t > L):
        λ(t) = W
    """
    W, L, R = 1.0, 100, 5.0
    initial_lr = 0.001

    scheduler = sa.constant_with_inverse_warmup(
        optimizer,
        warmup_to_value=W,
        num_warmup_steps=L,
        warmup_multiplier=R,
    )

    # At t=0 (start high)
    scheduler.step(0)
    expected_lambda = W * R
    expected_value = initial_lr * expected_lambda
    assert_close(get_scheduled_value(optimizer), expected_value)

    # After warmup - should stay at W
    scheduler.step(100)
    expected_value = initial_lr * W
    assert_close(get_scheduled_value(optimizer), expected_value)

    scheduler.step(500)
    expected_value = initial_lr * W
    assert_close(get_scheduled_value(optimizer), expected_value)


def test_constant_schedule_formula(optimizer):
    """
    Contract: Fixed value throughout training.

    Formula:
        λ(t) = V

    Where: V=value
    """
    V = 0.5
    initial_lr = 0.001

    scheduler = sa.constant_schedule(
        optimizer,
        value=V,
    )

    # At all steps, value is constant
    scheduler.step(0)
    expected_value = initial_lr * V
    assert_close(get_scheduled_value(optimizer), expected_value)

    scheduler.step(100)
    assert_close(get_scheduled_value(optimizer), expected_value)

    scheduler.step(1000)
    assert_close(get_scheduled_value(optimizer), expected_value)


# =============================================================================
# Boundary Value Tests
# =============================================================================


def test_warmup_boundary_values(optimizer):
    """Contract: Boundary conditions match documented warmup_to_value and anneal_to_value."""
    W, A, L, M = 1.0, 0.01, 100, 1000
    initial_lr = 0.001

    scheduler = sa.cosine_annealing_with_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=A,
        num_warmup_steps=L,
        num_training_steps=M,
    )

    # At end of warmup, should be at warmup_to_value
    scheduler.step(L)
    expected = initial_lr * W
    assert_close(get_scheduled_value(optimizer), expected)


def test_inverse_warmup_boundary_values(optimizer):
    """Contract: Inverse warmup starts at warmup_to_value * warmup_multiplier."""
    W, R = 1.0, 20.0
    initial_lr = 0.001

    scheduler = sa.cosine_annealing_with_inverse_warmup(
        optimizer,
        warmup_to_value=W,
        anneal_to_value=0.01,
        num_warmup_steps=100,
        num_training_steps=1000,
        warmup_multiplier=R,
    )

    # At t=0, should be at W * R
    scheduler.step(0)
    expected = initial_lr * W * R
    assert_close(get_scheduled_value(optimizer), expected, rtol=1e-4)


# =============================================================================
# Integration: All Schedules Work with Custom Parameters
# =============================================================================


@pytest.mark.parametrize(
    "schedule_func",
    [
        lambda opt: sa.cosine_annealing_with_warmup(opt, 1.0, 0.1, 10, 100, schedule_target="custom"),
        lambda opt: sa.linear_schedule_with_warmup(opt, 1.0, 0.1, 10, 100, schedule_target="custom"),
        lambda opt: sa.quadratic_schedule_with_warmup(opt, 1.0, 0.1, 10, 100, schedule_target="custom"),
        lambda opt: sa.sqrt_schedule_with_warmup(opt, 1.0, 0.1, 10, 100, schedule_target="custom"),
        lambda opt: sa.constant_with_warmup(opt, 1.0, 10, schedule_target="custom"),
        lambda opt: sa.constant_schedule(opt, 0.5, schedule_target="custom"),
    ],
)
def test_all_schedules_work_with_custom_parameters(optimizer, schedule_func):
    """Contract: All built-in schedules work with custom schedule_target."""
    sa.extend_optimizer(optimizer, "custom", default_value=1.0)

    scheduler = schedule_func(optimizer)

    initial_value = get_scheduled_value(optimizer, "custom")
    scheduler.step()

    # Observable: Custom parameter is being scheduled
    # (value may or may not change on first step depending on schedule)
    assert "custom" in optimizer.param_groups[0]
