# ScheduleAnything

Schedule any optimizer hyperparameter in PyTorch, not just learning rate.

PyTorch's built-in schedulers only work with learning rate. Want to schedule weight decay, momentum, or any custom parameter? Now you can.

## Installation

```bash
pip install torch-schedule-anything
```

Conventionally imported as:
```python
import torch_schedule_anything as sa
```

## Quick Start

Schedule weight decay with cosine annealing and warmup:

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch_schedule_anything as sa

# Standard PyTorch setup
model = nn.Linear(10, 1)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Schedule weight_decay with cosine annealing + warmup
wd_scheduler = sa.cosine_annealing_with_warmup(
    optimizer,
    warmup_to_value=0.01,
    anneal_to_value=0.0001,
    num_warmup_steps=100,
    num_training_steps=1000,
    schedule_target='weight_decay'  # That's it!
)

# Training loop
for step in range(1000):
    # Your training code here
    wd_scheduler.step()
```

Want to schedule learning rate instead? Just change `schedule_target='lr'` (or omit it, since 'lr' is the default).

## Built-in Schedules

### Common Parameters
- **warmup_to_value**: Target value at end of warmup phase
- **anneal_to_value**: Final value at end of training
- **num_warmup_steps**: Steps for warmup phase
- **num_training_steps**: Total training steps
- **warmup_multiplier**: For inverse warmup - starts at `warmup_to_value * warmup_multiplier`
- **polynomial_exponent**: Controls curve shape (higher = slower initial change)
- **schedule_target**: Which parameter to schedule (default: `'lr'`)

### Available Schedules

| Schedule | When to Use |
|----------|-------------|
| `cosine_annealing_with_warmup` | Smooth warmup then cosine decay - most popular choice |
| `cosine_annealing_with_inverse_warmup` | Start high, decay to baseline, then cosine anneal |
| `linear_schedule_with_warmup` | Simple linear decay after warmup |
| `polynomial_schedule_with_warmup` | Customizable curve shape (quadratic, cubic, etc.) |
| `quadratic_schedule_with_warmup` | Slow early decay, faster later |
| `sqrt_schedule_with_warmup` | Fast early decay, slower later |
| `constant_with_warmup` | Warmup to value then hold constant |
| `constant_schedule` | Hold one value throughout training |

*Inverse warmup variants available for all schedules (except constant).*

See [Builtin Schedule API Reference](documentation/builtin_schedules.md) for complete documentation with examples. See 

## Custom Schedules

Need something specific? Use any PyTorch scheduler with the factory pattern:

```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Schedule momentum with step decay  
momentum_scheduler = sa.arbitrary_schedule_factory(
    feature_name='momentum',
    optimizer=optimizer,
    schedule_factory=lambda opt: StepLR(opt, step_size=30, gamma=0.1),
    default_value=0.9  # Sets initial value if not present
)

# Schedule weight decay with cosine annealing  
wd_scheduler = sa.arbitrary_schedule_factory(
    feature_name='weight_decay',
    optimizer=optimizer,
    schedule_factory=lambda opt: CosineAnnealingLR(opt, T_max=100)
)
```

Any standard PyTorch scheduler works: `StepLR`, `ExponentialLR`, `ReduceLROnPlateau`, `LambdaLR`, etc.

## Schedule Multiple Parameters

Synchronize learning rate AND weight decay schedules:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

# Schedule learning rate
lr_scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Schedule weight decay  
wd_scheduler = sa.cosine_annealing_with_warmup(
    optimizer,
    warmup_to_value=0.01,
    anneal_to_value=0.0001,
    num_warmup_steps=10,
    num_training_steps=100,
    schedule_target='weight_decay'
)

# Synchronize them
sync_scheduler = sa.SynchronousSchedule([lr_scheduler, wd_scheduler])

# Step both together
for epoch in range(100):
    # Training loop here
    sync_scheduler.step()
```

## What Can You Schedule?

Anything in your optimizer's `param_groups`:
- `weight_decay` - L2 regularization strength
- `momentum` - SGD momentum  
- Custom parameters you've added

## Why ScheduleAnything?

Dynamic hyperparameter scheduling is a powerful training technique, but PyTorch only supports it for learning rate out of the box. Options like PyTorch-Ignite support other possibilities, but require full commitment to heavy frameworks. This is a lightweight solution that instead builds on top of what torch already provides you.

Perfect for:
- Research experiments with novel scheduling strategies
- Reproducing papers that schedule multiple hyperparameters
- Fine-grained training control

## License

MIT

## Contributing

Issues and PRs welcome at [github.com/smithblack-0/ScheduleAnything](https://github.com/yourusername/ScheduleAnything)