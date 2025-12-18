"""
Shared fixtures for ScheduleAnything test suite.

Provides common test setup following black box testing principles:
- Test observable behavior, not implementation details
- Use actual PyTorch components, not mocks
- Verify side effects and return values
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW

import src.torch_schedule_anything as sa


@pytest.fixture
def simple_model():
    """Create a simple linear model for testing."""
    return nn.Linear(10, 1)


@pytest.fixture
def optimizer(simple_model):
    """Create a basic AdamW optimizer with standard parameters."""
    return AdamW(simple_model.parameters(), lr=0.001, weight_decay=0.01)


@pytest.fixture
def optimizer_with_multiple_param_groups(simple_model):
    """Create optimizer with multiple parameter groups for testing."""
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
    return AdamW(
        [
            {"params": model[0].parameters(), "lr": 0.001, "weight_decay": 0.01},
            {"params": model[1].parameters(), "lr": 0.01, "weight_decay": 0.1},
        ]
    )


@pytest.fixture
def sgd_optimizer(simple_model):
    """Create SGD optimizer for testing different optimizer types."""
    return SGD(simple_model.parameters(), lr=0.1, momentum=0.9)


@pytest.fixture
def setup_schedule():
    """
    Helper to create scheduler with automatic parameter extension.

    Simplifies common pattern of:
    1. Extend optimizer with parameter (if needed)
    2. Create scheduler factory
    3. Wrap with arbitrary_schedule_factory

    Returns:
        Callable that creates and returns a scheduler

    Example:
        sched = setup_schedule(optimizer, "momentum", StepLR,
                              default_value=0.9, step_size=10, gamma=0.5)
    """

    def _setup(optimizer, param_name, scheduler_class, default_value=None, **scheduler_kwargs):
        """
        Args:
            optimizer: PyTorch optimizer to schedule
            param_name: Parameter name to schedule (e.g., 'lr', 'weight_decay', 'momentum')
            scheduler_class: PyTorch scheduler class (e.g., StepLR, ExponentialLR)
            default_value: If param doesn't exist, create with this value
            **scheduler_kwargs: Arguments to pass to scheduler_class constructor

        Returns:
            Configured scheduler instance
        """
        if default_value is not None:
            sa.extend_optimizer(optimizer, param_name, default_value=default_value)
        return sa.arbitrary_schedule_factory(
            optimizer,
            lambda opt: scheduler_class(opt, **scheduler_kwargs),
            schedule_target=param_name,
        )

    return _setup


@pytest.fixture
def setup_model_and_optimizer():
    """
    Helper to create model and optimizer with standard configuration.

    Returns:
        Callable that creates (model, optimizer) tuple

    Example:
        model, opt = setup_model_and_optimizer()
    """

    def _setup(lr=0.001, weight_decay=0.01):
        """
        Args:
            lr: Learning rate (default: 0.001)
            weight_decay: Weight decay (default: 0.01)

        Returns:
            Tuple of (model, optimizer)
        """
        model = nn.Linear(10, 1)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        return model, optimizer

    return _setup
