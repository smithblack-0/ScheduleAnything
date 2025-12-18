# ScheduleAnything

ScheduleAnything is foundation infrastructure designed to be composable that works with pytorch to allow attachment of schedules to any optimizer hyperparameter, not just learning rate. It allows extension of existing optimizer, and otherwise supports complex scenarios where the optimization thresholds need to change as training proceeds.

## Why ScheduleAnything?

PyTorch's built-in schedulers only work with learning rate. ScheduleAnything extends scheduling to **any optimizer parameter** - weight decay, momentum, gradient clipping thresholds, custom parameters - using the same familiar PyTorch scheduler interface.

This is a lightweight, focused tool following the Unix philosophy: do one thing well. That thing is support building tools and implementations around arbitrary hyperparameter scheduling, to be composed as part of a broader project.

## Who Needs This?

For researchers scheduling novel parameters or developers needing lightweight scheduling components to integrate into their projects. Not for standard model training with typical hyperparameter configurations. The important thing to keep in mind is **ScheduleAnything is Infrastructure**, not a prebuilt solution.

## Installation

```bash
pip install torch-schedule-anything
```

Canonically imported as:
```python
import torch_schedule_anything as sa
```

## Quick Start

```python
import torch.nn as nn
from torch.optim import AdamW
import torch_schedule_anything as sa

# Standard PyTorch setup
model = nn.Linear(10, 1)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Schedule weight decay with cosine annealing
scheduler = sa.cosine_annealing_with_warmup(
    optimizer,
    warmup_to_value=1.0,
    anneal_to_value=0.01,
    num_warmup_steps=100,
    num_training_steps=1000,
    schedule_target='weight_decay'
)
scheduler = sa.SynchronousSchedule([scheduler])

# Training loop
for step in range(1000):
    # Your training code here
    scheduler.step()
```

Want to schedule learning rate instead? Change `schedule_target='lr'` (or omit it - 'lr' is the default)

## What Can You Schedule?

Anything in your optimizer's `param_groups`:
- Learning rate (`lr`)
- Weight decay (`weight_decay`)
- Momentum (`momentum`)
- Dampening (`dampening`)
- Gradient clipping thresholds
- Custom parameters you define

The library works by proxying PyTorch's learning rate scheduling mechanism to arbitrary parameters. It can also extend and insert new parameters if you want as well.

---
## Technical Highlights

### Built-In Schedules

13 pre-configured curve primitives covering common training patterns. Only the classes of schedules are currently shown

| Schedule Type | Description |
|--------------|-------------|
| **Cosine** | Smooth S-shaped transitions (standard and inverse warmup) |
| **Polynomial** | Customizable curve shapes with arbitrary exponents |
| **Linear** | Constant-rate decay |
| **Quadratic** | Accelerating decay (slow start, fast finish) |
| **Square Root** | Decelerating decay (fast start, slow finish) |
| **Constant** | Flat after warmup |

All schedules work on any optimizer parameter via `schedule_target`. Each includes standard and inverse warmup variants.

**→** See [Built-In Schedules API Reference](documentation/builtin_schedules.md) for complete documentation including mathematical formulas.

### Custom Schedules

Use **any PyTorch scheduler** on **any parameter** via the factory pattern. Compatible with:
- `StepLR`, `MultiStepLR`, `ExponentialLR`
- `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`
- `LambdaLR` for custom curves
- Any other PyTorch `_LRScheduler`

The factory handles parameter creation, initialization, and binding automatically. Perfect for extending optimizer behavior with custom parameters that your training code can read and respond to.

**→** See [User Guide - Custom Schedules](documentation/user_guide.md#custom-schedules-with-the-factory) for usage details.

### Parallel Schedule Coordination

`SynchronousSchedule` coordinates multiple schedules:
- Keeps schedulers in lockstep (no desynchronization)
- Provides honest API methods (no lying `get_lr()`)
- Supports state dict save/load
- Handles arbitrary numbers of schedules

Essential when scheduling multiple parameters simultaneously.

**→** See [User Guide - Parallel Schedules](documentation/user_guide.md#parallel-schedules) for patterns and best practices.

### Case Study: Complete Training Setup

Combining all three capabilities - built-in schedules, custom schedules via factory, and parallel coordination.

**Scenario:** You have a custom gradient clipping function that reads per-parameter-group thresholds:

```python
def my_custom_gradient_clipping(optimizer):
    """
    Apply gradient clipping per parameter group based on scheduled thresholds.
    Reads 'gradient_clip_threshold' from each param_group.
    """
    for threshold, parameters, group in sa.get_param_groups_regrouped_by_key(optimizer, "gradient_clip_threshold"):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=threshold)
```

You also have `MyCustomSchedule` that needs the number of training steps.

You want to:
1. Schedule the gradient clipping threshold from 10 → 0 over training
2. Schedule weight decay to strengthen from 0.01 → 0.1 (quadratic curve)
3. Schedule learning rate with standard cosine annealing

**Here's how ScheduleAnything achieves this:**

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch_schedule_anything as sa

optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Custom parameter: Gradient clipping threshold
# Use your custom scheduler via factory - starts at 10, anneals to 0
clip_scheduler = sa.arbitrary_schedule_factory(
    optimizer=optimizer,
    schedule_factory=lambda opt: MyCustomSchedule(opt, train_steps=10000),
    default_value=10.0,  # Initialize the custom parameter
    schedule_target='gradient_clip_threshold',

)

# Built-in: Weight decay strengthening over training
# Quadratic curve - starts loose (0.01), tightens near end (1.0)
wd_scheduler = sa.quadratic_schedule_with_warmup(
    optimizer,
    warmup_to_value=0.01,
    anneal_to_value=1.0,
    num_warmup_steps=500,
    num_training_steps=10000,
    schedule_target='weight_decay'
)

# Built-in: Standard learning rate with cosine annealing
lr_scheduler = sa.cosine_annealing_with_warmup(
    optimizer,
    warmup_to_value=1.0,
    anneal_to_value=0.01,
    num_warmup_steps=500,
    num_training_steps=10000,
    schedule_target='lr'
)

# Coordinate all three schedules
sync = sa.SynchronousSchedule([clip_scheduler, wd_scheduler, lr_scheduler])

# Training loop (step-based)
for step in range(10000):
    # Forward pass, backward pass
    loss.backward()
    
    # Apply gradient clipping using scheduled threshold
    # This function reads gradient_clip_threshold from optimizer.param_groups
    my_custom_gradient_clipping(optimizer)
    
    optimizer.step()
    optimizer.zero_grad()
    
    # Step all schedules together
    sync.step()
```

This demonstrates the full power: custom parameters created via factory, built-in curves for standard parameters, and synchronous coordination keeping everything aligned.

---

## Documentation

- **[User Guide](documentation/user_guide.md)** - Complete usage guide with examples and best practices
- **[Built-In Schedules](documentation/builtin_schedules.md)** - API reference with mathematical formulas for all schedules
- **[Infrastructure](documentation/infrastructure.md)** for complete API references of the utilities.


## License

MIT

## Contributing

Issues and PRs welcome at [Github](https://github.com/smithblack-0/ScheduleAnything)