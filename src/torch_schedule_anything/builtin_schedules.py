"""
Built-in schedules for ScheduleAnything.

Provides 13 pre-configured schedule functions covering common training patterns.
All schedules work on any optimizer parameter via the schedule_target argument.

Schedules use PyTorch's multiplier convention: λ(t) is applied to initial parameter value.
value(t) = initial_value × λ(t)

All formulas match the documented API specification in builtin_schedules.md.
"""

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# Backward compatibility: PyTorch renamed _LRScheduler to LRScheduler
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from .infrastructure import arbitrary_schedule_factory

# =============================================================================
# Cosine Schedules
# =============================================================================


def cosine_annealing_with_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Cosine annealing with linear warmup.

    Formula:
        Warmup (t ≤ L): λ(t) = (W * t) / L
        Annealing (t > L): λ(t) = A + (W - A) * cos((π/2) * (t - L) / (M - L))

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Target multiplier at end of warmup
        anneal_to_value: Final multiplier at end of training
        num_warmup_steps: Steps for warmup phase
        num_training_steps: Total training steps
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    W, A, L, M = warmup_to_value, anneal_to_value, num_warmup_steps, num_training_steps

    def lr_lambda(step):
        if step < L:
            # Linear warmup
            return (W * step) / L if L > 0 else W
        else:
            # Cosine annealing
            progress = (step - L) / (M - L) if M > L else 1.0
            return A + (W - A) * math.cos((math.pi / 2) * progress)

    return arbitrary_schedule_factory(
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda),
        schedule_target=schedule_target,
    )


def cosine_annealing_with_inverse_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    warmup_multiplier: float = 20.0,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Cosine annealing with inverse warmup (high to baseline).

    Formula:
        Inverse warmup (t ≤ L): λ(t) = W * (R - (R-1) * t/L)
        Annealing (t > L): λ(t) = A + (W - A) * cos((π/2) * (t - L) / (M - L))

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Baseline multiplier after inverse warmup
        anneal_to_value: Final multiplier at end of training
        num_warmup_steps: Steps for inverse warmup phase
        num_training_steps: Total training steps
        warmup_multiplier: Starting multiplier (starts at W * R, default: 20.0)
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    W, A, L, M, R = (
        warmup_to_value,
        anneal_to_value,
        num_warmup_steps,
        num_training_steps,
        warmup_multiplier,
    )

    def lr_lambda(step):
        if step < L:
            # Inverse warmup: Start high, decay to baseline
            return W * (R - (R - 1) * step / L) if L > 0 else W
        else:
            # Cosine annealing
            progress = (step - L) / (M - L) if M > L else 1.0
            return A + (W - A) * math.cos((math.pi / 2) * progress)

    return arbitrary_schedule_factory(
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda),
        schedule_target=schedule_target,
    )


# =============================================================================
# Polynomial Schedules
# =============================================================================


def polynomial_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    polynomial_exponent: float = 2.0,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Polynomial decay with linear warmup.

    Formula:
        Warmup (t ≤ L): λ(t) = (W * t) / L
        Annealing (t > L): λ(t) = A + (W - A) * (1 - (t - L) / (M - L))^P

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Target multiplier at end of warmup
        anneal_to_value: Final multiplier at end of training
        num_warmup_steps: Steps for warmup phase
        num_training_steps: Total training steps
        polynomial_exponent: Exponent for polynomial decay (default: 2.0)
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    W, A, L, M, P = (
        warmup_to_value,
        anneal_to_value,
        num_warmup_steps,
        num_training_steps,
        polynomial_exponent,
    )

    def lr_lambda(step):
        if step < L:
            # Linear warmup
            return (W * step) / L if L > 0 else W
        else:
            # Polynomial decay
            progress = (step - L) / (M - L) if M > L else 1.0
            return A + (W - A) * ((1 - progress) ** P)

    return arbitrary_schedule_factory(
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda),
        schedule_target=schedule_target,
    )


def polynomial_schedule_with_inverse_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    polynomial_exponent: float = 2.0,
    warmup_multiplier: float = 20.0,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Polynomial decay with inverse warmup.

    Formula:
        Inverse warmup (t ≤ L): λ(t) = W * (R - (R-1) * t/L)
        Annealing (t > L): λ(t) = A + (W - A) * (1 - (t - L) / (M - L))^P

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Baseline multiplier after inverse warmup
        anneal_to_value: Final multiplier at end of training
        num_warmup_steps: Steps for inverse warmup phase
        num_training_steps: Total training steps
        polynomial_exponent: Exponent for polynomial decay (default: 2.0)
        warmup_multiplier: Starting multiplier (starts at W * R, default: 20.0)
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    W, A, L, M, P, R = (
        warmup_to_value,
        anneal_to_value,
        num_warmup_steps,
        num_training_steps,
        polynomial_exponent,
        warmup_multiplier,
    )

    def lr_lambda(step):
        if step < L:
            # Inverse warmup
            return W * (R - (R - 1) * step / L) if L > 0 else W
        else:
            # Polynomial decay
            progress = (step - L) / (M - L) if M > L else 1.0
            return A + (W - A) * ((1 - progress) ** P)

    return arbitrary_schedule_factory(
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda),
        schedule_target=schedule_target,
    )


def linear_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Linear decay with warmup. Equivalent to polynomial_schedule_with_warmup with exponent=1.0.

    Formula:
        Warmup (t ≤ L): λ(t) = (W * t) / L
        Annealing (t > L): λ(t) = A + (W - A) * (1 - (t - L) / (M - L))

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Target multiplier at end of warmup
        anneal_to_value: Final multiplier at end of training
        num_warmup_steps: Steps for warmup phase
        num_training_steps: Total training steps
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    return polynomial_schedule_with_warmup(
        optimizer,
        warmup_to_value,
        anneal_to_value,
        num_warmup_steps,
        num_training_steps,
        polynomial_exponent=1.0,
        schedule_target=schedule_target,
    )


def linear_schedule_with_inverse_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    warmup_multiplier: float = 20.0,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Linear decay with inverse warmup.

    Formula:
        Inverse warmup (t ≤ L): λ(t) = W * (R - (R-1) * t/L)
        Annealing (t > L): λ(t) = A + (W - A) * (1 - (t - L) / (M - L))

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Baseline multiplier after inverse warmup
        anneal_to_value: Final multiplier at end of training
        num_warmup_steps: Steps for inverse warmup phase
        num_training_steps: Total training steps
        warmup_multiplier: Starting multiplier (starts at W * R, default: 20.0)
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    return polynomial_schedule_with_inverse_warmup(
        optimizer,
        warmup_to_value,
        anneal_to_value,
        num_warmup_steps,
        num_training_steps,
        polynomial_exponent=1.0,
        warmup_multiplier=warmup_multiplier,
        schedule_target=schedule_target,
    )


def quadratic_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Quadratic decay with warmup. Equivalent to polynomial_schedule_with_warmup with exponent=2.0.

    Formula:
        Warmup (t ≤ L): λ(t) = (W * t) / L
        Annealing (t > L): λ(t) = A + (W - A) * (1 - (t - L) / (M - L))^2

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Target multiplier at end of warmup
        anneal_to_value: Final multiplier at end of training
        num_warmup_steps: Steps for warmup phase
        num_training_steps: Total training steps
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    return polynomial_schedule_with_warmup(
        optimizer,
        warmup_to_value,
        anneal_to_value,
        num_warmup_steps,
        num_training_steps,
        polynomial_exponent=2.0,
        schedule_target=schedule_target,
    )


def quadratic_schedule_with_inverse_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    warmup_multiplier: float = 20.0,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Quadratic decay with inverse warmup.

    Formula:
        Inverse warmup (t ≤ L): λ(t) = W * (R - (R-1) * t/L)
        Annealing (t > L): λ(t) = A + (W - A) * (1 - (t - L) / (M - L))^2

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Baseline multiplier after inverse warmup
        anneal_to_value: Final multiplier at end of training
        num_warmup_steps: Steps for inverse warmup phase
        num_training_steps: Total training steps
        warmup_multiplier: Starting multiplier (starts at W * R, default: 20.0)
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    return polynomial_schedule_with_inverse_warmup(
        optimizer,
        warmup_to_value,
        anneal_to_value,
        num_warmup_steps,
        num_training_steps,
        polynomial_exponent=2.0,
        warmup_multiplier=warmup_multiplier,
        schedule_target=schedule_target,
    )


def sqrt_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Square root decay with warmup. Equivalent to polynomial_schedule_with_warmup with exponent=0.5.

    Formula:
        Warmup (t ≤ L): λ(t) = (W * t) / L
        Annealing (t > L): λ(t) = A + (W - A) * (1 - (t - L) / (M - L))^0.5

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Target multiplier at end of warmup
        anneal_to_value: Final multiplier at end of training
        num_warmup_steps: Steps for warmup phase
        num_training_steps: Total training steps
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    return polynomial_schedule_with_warmup(
        optimizer,
        warmup_to_value,
        anneal_to_value,
        num_warmup_steps,
        num_training_steps,
        polynomial_exponent=0.5,
        schedule_target=schedule_target,
    )


def sqrt_schedule_with_inverse_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    warmup_multiplier: float = 20.0,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Square root decay with inverse warmup.

    Formula:
        Inverse warmup (t ≤ L): λ(t) = W * (R - (R-1) * t/L)
        Annealing (t > L): λ(t) = A + (W - A) * (1 - (t - L) / (M - L))^0.5

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Baseline multiplier after inverse warmup
        anneal_to_value: Final multiplier at end of training
        num_warmup_steps: Steps for inverse warmup phase
        num_training_steps: Total training steps
        warmup_multiplier: Starting multiplier (starts at W * R, default: 20.0)
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    return polynomial_schedule_with_inverse_warmup(
        optimizer,
        warmup_to_value,
        anneal_to_value,
        num_warmup_steps,
        num_training_steps,
        polynomial_exponent=0.5,
        warmup_multiplier=warmup_multiplier,
        schedule_target=schedule_target,
    )


# =============================================================================
# Constant Schedules
# =============================================================================


def constant_with_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    num_warmup_steps: int,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Constant schedule with linear warmup.

    Formula:
        Warmup (t ≤ L): λ(t) = (W * t) / L
        After warmup (t > L): λ(t) = W

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Target multiplier (holds after warmup)
        num_warmup_steps: Steps for warmup phase
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    W, L = warmup_to_value, num_warmup_steps

    def lr_lambda(step):
        if step < L:
            # Linear warmup
            return (W * step) / L if L > 0 else W
        else:
            # Hold constant
            return W

    return arbitrary_schedule_factory(
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda),
        schedule_target=schedule_target,
    )


def constant_with_inverse_warmup(
    optimizer: Optimizer,
    warmup_to_value: float,
    num_warmup_steps: int,
    warmup_multiplier: float = 20.0,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Constant schedule with inverse warmup.

    Formula:
        Inverse warmup (t ≤ L): λ(t) = W * (R - (R-1) * t/L)
        After warmup (t > L): λ(t) = W

    Args:
        optimizer: Optimizer to schedule
        warmup_to_value: Target multiplier (holds after warmup)
        num_warmup_steps: Steps for inverse warmup phase
        warmup_multiplier: Starting multiplier (starts at W * R, default: 20.0)
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    W, L, R = warmup_to_value, num_warmup_steps, warmup_multiplier

    def lr_lambda(step):
        if step < L:
            # Inverse warmup
            return W * (R - (R - 1) * step / L) if L > 0 else W
        else:
            # Hold constant
            return W

    return arbitrary_schedule_factory(
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda),
        schedule_target=schedule_target,
    )


def constant_schedule(
    optimizer: Optimizer,
    value: float,
    schedule_target: str = "lr",
) -> LRScheduler:
    """
    Pure constant schedule (no warmup).

    Formula:
        λ(t) = V (for all t)

    Args:
        optimizer: Optimizer to schedule
        value: Constant multiplier value
        schedule_target: Parameter to schedule (default: 'lr')

    Returns:
        Scheduler that implements the documented formula
    """
    V = value

    def lr_lambda(step):
        return V

    return arbitrary_schedule_factory(
        optimizer,
        lambda opt: LambdaLR(opt, lr_lambda),
        schedule_target=schedule_target,
    )
