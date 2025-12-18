"""
Tests for low-level adapter/proxy infrastructure.

Tests the adapter and proxy mechanisms that enable scheduling arbitrary parameters:
- ArbitraryScheduleAdapter
- ProxyDictByLR
- Desync detection

These are internal infrastructure tests for the adapter layer.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import src.torch_schedule_anything as sa


def test_desync_detection_raises_error(optimizer):
    """
    Contract: Desync detection catches when backend is modified directly.
    Why: Prevents silent bugs from bypassing the scheduler proxy.
    How: Create proxy, modify backend directly, attempt to use proxy, verify RuntimeError.

    This test verifies internal desync detection by:
    1. Creating a scheduler with proxy
    2. Modifying the backend parameter directly (bypassing proxy)
    3. Attempting to use the scheduler, which should detect desync
    """
    from src.torch_schedule_anything.arbitrary_schedules import ArbitraryScheduleAdapter

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
