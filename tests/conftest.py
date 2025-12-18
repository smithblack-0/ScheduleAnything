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
from torch.optim import AdamW, SGD


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
