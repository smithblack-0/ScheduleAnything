# ScheduleAnything User Guide

Complete guide to using `torch-schedule-anything` for flexible hyperparameter scheduling in PyTorch.

---

## Who This Library Is For

This library is a **research tool** for people who understand optimization and need precise control over hyperparameter curves during training. It is also a **production component** for those who want to hook up torch optimizers in python as part of a broader project.

**You want this library if:**
- You need to schedule parameters beyond learning rate (weight decay, momentum, gradient clipping thresholds, etc.)
- You want specific curve shapes (cosine, polynomial, custom) for your hyperparameters
- You're implementing novel training strategies from papers.
- You understand your optimizer's update equations and how parameters interact
- You want a lightweight arbitrary scheduling tool in a familiar form factor without having to commit to a framework.

**You don't need this library if:**
- You just want standard learning rate scheduling (use PyTorch's built-in schedulers).
- You're following a standard training recipe for common architectures.
- You're unsure how your optimizer's hyperparameters interact.
- You do not know what a scheduling curve is.

This is professional infrastructure for researchers and developers, not a convenience wrapper for beginners. It follows the unix philsophy of doing one thing, and one thing well. That thing is arbitrary schedule support.

---

## Canonical Import

Throughout this guide and in your code, use:

```python
import torch_schedule_anything as sa
```

All examples follow this convention.

---

## Built-In Schedules

The library provides 13 pre-configured schedules covering common curve shapes. All work on any optimizer parameter via the `schedule_target` argument (defaults to `'lr'`).

### Quick Example

```python
import torch.nn as nn
from torch.optim import AdamW
import torch_schedule_anything as sa

model = nn.Linear(10, 1)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Schedule weight decay with cosine annealing
scheduler = sa.cosine_annealing_with_warmup(
    optimizer,
    warmup_to_value=0.01,
    anneal_to_value=0.001,
    num_warmup_steps=100,
    num_training_steps=1000,
    schedule_target='weight_decay'
)

# Training loop
for step in range(1000):
    # ... training code ...
    scheduler.step()
```

### Available Schedules

The following built-in schedules are available:

**Cosine curves** - Smooth half-pi cosine transitions:
- `cosine_annealing_with_warmup` - Standard warmup then cosine decay
- `cosine_annealing_with_inverse_warmup` - Start high, decay to baseline, then cosine

**Polynomial curves** - Customizable curve shapes:
- `polynomial_schedule_with_warmup` - Arbitrary polynomial exponent
- `linear_schedule_with_warmup` - Constant-rate linear decay
- `quadratic_schedule_with_warmup` - Accelerating decay (slow start, fast finish)
- `sqrt_schedule_with_warmup` - Decelerating decay (fast start, slow finish)
- Inverse warmup variants available for all

**Constant schedules** - Flat after warmup:
- `constant_with_warmup` - Ramp up then hold
- `constant_with_inverse_warmup` - Decay down then hold
- `constant_schedule` - Fixed value throughout

For complete API documentation including mathematical formulas, parameter specifications, and detailed examples, see [Schedules](builtin_schedules.md). Most users should be covered by these use cases.

---

## Custom Schedules with the Factory

For maximum flexibility, use `arbitrary_schedule_factory` to apply any PyTorch scheduler to any parameter.

### Basic Usage

The basic idea is the user passes in a factory that can be invoked with an optimizer to produce a schedule attached to that optimizer. ScheduleAnything then constructs an optimizer that will in fact bind to and schedule the targetted feature name.

### When to Use `default_value`

**If the parameter already exists** in your optimizer (like `weight_decay` in Adam), you don't need `default_value`:

```python
# Adam already has weight_decay - no default needed
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = sa.arbitrary_schedule_factory(
    feature_name='weight_decay',
    optimizer=optimizer,
    schedule_factory=lambda opt: CosineAnnealingLR(opt, T_max=100)
)
```

**If you're creating a new parameter**, provide `default_value`. It will be added to param_groups if missing:

```python
# Creating a custom parameter for the first time
scheduler = sa.arbitrary_schedule_factory(
    feature_name='my_custom_parameter',
    optimizer=optimizer,
    schedule_factory=lambda opt: ExponentialLR(opt, gamma=0.95),
    default_value=1.0  # Sets initial value if not present
)
```

### Power User: Direct Optimizer Extension

For advanced use cases, you can extend the optimizer directly before creating schedules. Note you will then have to look up the set schedules yourself downstream:

```python
# Add a custom parameter to all param groups
sa.extend_optimizer(optimizer, 'gradient_clip_threshold', default_value=10.0)

# Now schedule it
scheduler = sa.arbitrary_schedule_factory(
    feature_name='gradient_clip_threshold',
    optimizer=optimizer,
    schedule_factory=lambda opt: StepLR(opt, step_size=100, gamma=0.5)
)
```

This is particularly useful when building optimizer wrappers or extending optimizer behavior. The Gradient Quality Control library uses this pattern to implement scheduled gradient norm thresholds and many other behaviors.

---

## Parallel Schedules

Schedule Anything allows you to schedule multiple parameters independently. This provides fantastic flexibility. Unfortunately, it also forms a new problem. How do you keep it all organized?

### The Problem

If you create multiple schedulers and step them separately, they can desync:

```python
lr_scheduler = sa.cosine_annealing_with_warmup(...)
wd_scheduler = sa.quadratic_schedule_with_warmup(...)

# Don't do this!
for step in range(1000):
    # training...
    lr_scheduler.step()
    # Oops, forgot wd_scheduler! Now they're out of sync.
```

Furthermore, some methods are not true to what they actually do. In reality, the torch schedule is being attached to an optimizer that passes off learning rate changes to the main dictionary in the main optimizer. However, this makes some methods lie.

```python
lr_scheduler = sa.cosine_annealing_with_warmup(...)
wd_scheduler = sa.quadratic_schedule_with_warmup(...)

# Don't do this!
for step in range(1000):
    # training...
    lr_scheduler.step()
    wd_scheduler.step()

    lr = lr_scheduler.last_lr()
    wd = lr_scheduler.last_lr() # But it is actually the weight decay!
```

Neither of these are good things, but they are necessary to use raw torch schedules directly.

### The Solution: SynchronousSchedule

`SynchronousSchedule` steps multiple schedulers together in lockstep and provides sane access methods.

```python
lr_scheduler = sa.cosine_annealing_with_warmup(
    optimizer,
    warmup_to_value=0.001,
    anneal_to_value=0.00001,
    num_warmup_steps=100,
    num_training_steps=1000,
    schedule_target='lr'
)

wd_scheduler = sa.quadratic_schedule_with_warmup(
    optimizer,
    warmup_to_value=0.01,
    anneal_to_value=0.1,
    num_warmup_steps=100,
    num_training_steps=1000,
    schedule_target='weight_decay'
)

# Synchronize them
sync = sa.SynchronousSchedule([lr_scheduler, wd_scheduler])

# Single step() call updates both
for step in range(1000):
    # training...
    sync.step()
```

You can then retrieve individual schedule values:

```python
current_lr = sync.get_last_schedule('lr')
current_wd = sync.get_last_schedule('weight_decay')

# Or just get learning rate specifically
current_lr = sync.get_last_lr()
```

As such, it is generally recommended to use a SyncronousSchedule object.

---

## Tips and Best Practices

**Understand your optimizer's formulas.** Make sure you understand how your optimizer's hyperparameters interact before scheduling them. For example, using both weight decay scheduling and learning rate scheduling would be inappropriate in AdamW, as weight decay already depends on learning rate. Don't schedule parameters blindly.

**Use this library to extend optimizer behavior.** It's possible to extend optimizer behavior using an optimizer wrapper with this system. Have the optimizer wrapper inject initial hyperparameters into the optimizer, and have the wrapper respond to them. Then attach schedules. This is how the Gradient Quality Control library implements gradient norm thresholds, and generally this library works excellently as a lightweight optimizer extender for other projects.

**Initialize schedules through this system whenever possible** It's generally a better idea to initialize your learning rate schedule through this system rather than independently, as the system prevents schedule namespace collisions. If you need to know what schedule namespace collisions are, you're having a bad day.

**The method name lies!** While calling `get_lr()` on individual schedules works, keep in mind the method name lies! It actually returns the bound schedule feature value, not necessarily learning rate. Getting the feature through `SynchronousSchedule` with `get_last_schedule(name)` doesn't have this problem - it's honest about what it returns.

**Check your serialization.** Make sure your optimizer serialization algorithms are up to snuff. This library injects custom elements into optimizer dictionaries. If you're just serializing the dictionary (like `torch.save(optimizer.state_dict())`), all is well. If you're constructing custom tuples by element name, you may have problems. For normal PyTorch optimizers, this should be no issue.

---

## Next Steps

- See [builtin schedules.md](builtin_schedules.md) for complete API reference with mathematical formulas
- Read the [README](README.md) for installation and quick start.