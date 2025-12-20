# ScheduleAnything User Guide

Complete guide to using `torch-schedule-anything` for flexible hyperparameter scheduling in PyTorch and extension of optimizers.


## Who This Library Is For

This library is a **research tool** for people who understand optimization and need precise control over hyperparameter curves during training. It is also a **production component** for those who want to hook up torch optimizers in python as part of a broader project. It is professional infrastructure for researchers and developers, not a convenience wrapper for beginners. It follows the unix philosophy of doing one thing, and one thing well. That thing is an arbitrary optimizer hyperparameter scheduling extension in the PyTorch ecosystem. 

**You want this library if your requirements are:**

- A need to arbitrarily schedule hyperparameters in the optimizer, and a lack of existing knowledge on exactly what needs scheduling, 
- A need to write arbitrary schedules that extend the torch optimizer that are hooked up to, for instance, a gradient clip threshold.
- Clear documentation, simplicity, and flexibility; you could walk back in 6 months and make any change you need
- The confidence and simplicity you can know this has been done right, and that failures fail fast rather than corrupt silently.
- A refusal to an existing framework for only one little part, or a situation where you are developing a framework yourself.
- You want specific curve shapes (cosine, polynomial, custom) for your hyperparameters and want reasonable support for common defaults, but the ability to define your own curves as well.

Then this tool is likely for you. It is designed for compositon as part of a broader whole This covers primarily framework developers and researchers, but also may cover experimenters who have an unusual needed schedule case, like batch size. 

**You don't need this library if:**

- You just want standard learning rate scheduling (use PyTorch's built-in schedulers).
- You're following a standard training recipe for common architectures.
- You're unsure how your optimizer's hyperparameters interact.
- You do not know what a scheduling curve is or how to define it.
- You are not willing to reason about the internals of optimizers.

This project is deliberately documented using Document-Driven Development, append-only once released, and simple enough to be confident no test case is missed. **Correctness, Composability, and Flexibility at a minor and intuitive Verbosity increase is the trade we make.**

## Canonical Import

Throughout this guide and in your code, use:

```python
import torch_schedule_anything as tsa
```

All examples follow this convention.

## Navigation

- See [builtin schedules.md](builtin_schedules.md) for complete API reference of builtin schedules with mathematical formulas
- See [Infrastructure](infrastructure.md) for complete API references of the syncronous schedules and helpers.
- Read the [README](README.md) for installation and quick start.

---
## What is a Schedule?

The term 'Schedule' as defined in this library is the same as in Canonical PyTorch: A multiplier that is applied to an initial hyperparameter. PyTorch schedulers work by computing a **multiplier** λ(t) that gets applied to initial parameter values:

$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

To maintain maximal compatibility, we adopt PyTorch's conventions.

**This means schedules set λ(t), NOT absolute values.**

When you specify `warmup_to_value=1.0` and `anneal_to_value=0.1`, you're defining the multiplier curve. If your parameter starts at 0.001, the actual values will be 0.001 → 0.0001.

**If you want absolute behavior, make sure to initialize new schedule hyperparameters to 1.0**

---
## Built-In Schedules

The library provides 13 pre-configured schedules covering common curve shapes. All work on any optimizer parameter via the `schedule_target` argument (defaults to `'lr'`). Adding more schedules to the synchronous
schedule list allows scheduling multiple things at once on different parameters. For reasons that shall
be explained shortly, you should still use a SynchronousSchedule even when only scheduling one object.

### Quick Example
```python
import torch.nn as nn
from torch.optim import AdamW
import torch_schedule_anything as tsa

model = nn.Linear(10, 1)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Schedule weight decay with cosine annealing
scheduler = tsa.cosine_annealing_with_warmup(
    optimizer,
    warmup_to_value=1.0,
    anneal_to_value=0.01,
    num_warmup_steps=100,
    num_training_steps=1000,
    schedule_target='weight_decay'
)
scheduler = tsa.SynchronousSchedule([scheduler])

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

### Basic Understanding

The basic idea is the user passes in a factory that can be invoked with an optimizer to produce a schedule attached to that optimizer. 

Behind the scenes, ScheduleAnything then constructs a proxy optimizer with a learning rate that in fact represents the scheduled parameters - for example lr=0.1 might actually mean weight_decay=0.1. Changes to this rate are then passed on, through a transform, to the actual value in the optimizer. Since torch schedules know how to change learning rates, they can change this as well.

For the most part, the user never needs to interact with these fake optimizers, but you may occasionally encounter them in error messages.

### When to Use `default_value`

**If the parameter already exists** in your optimizer (like `weight_decay` in Adam), you don't need `default_value`:

```python
# Adam already has weight_decay - no default needed
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = tsa.arbitrary_schedule_factory(
    optimizer=optimizer,
    schedule_factory=lambda opt: CosineAnnealingLR(opt, T_max=100),
    schedule_target='weight_decay',
)
```

**If you're creating a new parameter**, provide `default_value`. It will be added to param_groups if missing:

```python
# Creating a custom parameter for the first time
scheduler = tsa.arbitrary_schedule_factory(
    optimizer=optimizer,
    schedule_factory=lambda opt: ExponentialLR(opt, gamma=0.95),
    default_value=1.0,  # Base value multiplier applies to
    schedule_target='gradient_clip_value',

)

```


Advanced users can, of course, just manually include it with separate values when initializing the optimizers param groups instead. Consult torch's optimizer documentation for details on this. 

For complete API documentation including mathematical formulas, parameter specifications, and detailed examples, see [Utilities](infrastructure.md).

### Power User: Direct Optimizer Extension

For advanced use cases, you can extend the optimizer directly before creating schedules. Note you will then have to look up the set schedules yourself downstream:

```python
# Add a custom parameter to all param groups
tsa.extend_optimizer(optimizer, 'gradient_clip_threshold', default_value=10.0)

# Now schedule it
scheduler = tsa.arbitrary_schedule_factory(
    optimizer=optimizer,
    schedule_factory=lambda opt: StepLR(opt, step_size=100, gamma=0.5),
    schedule_target='gradient_clip_threshold',
)
```

This is particularly useful when building optimizer wrappers or extending optimizer behavior. The Gradient Quality Control library uses this pattern to implement scheduled gradient norm thresholds and many other behaviors.

---

For complete API documentation including mathematical formulas, parameter specifications, and detailed examples, see [Utilities](infrastructure.md). Power users are covered.

## Parallel Schedules

Schedule Anything allows you to schedule multiple parameters independently. This provides fantastic flexibility. Unfortunately, it also forms a new problem. How do you keep it all organized?

### The Problem

If you create multiple schedulers and step them separately, they can desync:

```python
lr_scheduler = tsa.cosine_annealing_with_warmup(...)
wd_scheduler = tsa.quadratic_schedule_with_warmup(...)

# Don't do this!
for step in range(1000):
    # training...
    lr_scheduler.step()
    # Oops, forgot wd_scheduler! Now they're out of sync.
```

Furthermore, some methods are not true to what they actually do. In reality, the torch schedule is being attached to an optimizer that passes off learning rate changes to the main dictionary in the main optimizer. However, this makes some methods lie.

```python
lr_scheduler = tsa.cosine_annealing_with_warmup(...)
wd_scheduler = tsa.quadratic_schedule_with_warmup(...)

# Don't do this!
for step in range(1000):
    # training...
    lr_scheduler.step()
    wd_scheduler.step()

    lr = lr_scheduler.last_lr()
    wd = wd_scheduler.last_lr() # But it is actually the weight decay!
```

Neither of these are good things, but they are necessary to use raw torch schedules directly.

### The Solution: SynchronousSchedule

`SynchronousSchedule` steps multiple schedulers together in lockstep and provides sane access methods.

```python
lr_scheduler = tsa.cosine_annealing_with_warmup(
    optimizer,
    warmup_to_value=1.0,
    anneal_to_value=0.001,
    num_warmup_steps=100,
    num_training_steps=1000,
    schedule_target='lr'
)

wd_scheduler = tsa.quadratic_schedule_with_warmup(
    optimizer,
    warmup_to_value=1.0,
    anneal_to_value=0.01,
    num_warmup_steps=100,
    num_training_steps=1000,
    schedule_target='weight_decay'
)

# Synchronize them
sync = tsa.SynchronousSchedule([lr_scheduler, wd_scheduler])

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

As such, it is generally recommended to use a SynchronousSchedule object.

For complete API documentation including mathematical formulas, parameter specifications, and detailed examples, see [Utilities](infrastructure.md). Power users are covered.

---

## Tips and Best Practices

**Understand your optimizer's formulas.** Make sure you understand how your optimizer's hyperparameters interact before scheduling them. For example, using both weight decay scheduling and learning rate scheduling would be inappropriate in AdamW, as weight decay already depends on learning rate. Don't schedule parameters blindly.

**Use this library to extend optimizer behavior.** It's possible to extend optimizer behavior using an optimizer wrapper with this system. Have the optimizer wrapper inject initial hyperparameters into the optimizer, and have the wrapper respond to them. Then attach schedules. This is how the Gradient Quality Control library implements gradient norm thresholds, and generally this library works excellently as a lightweight optimizer extender for other projects.

**Initialize schedules through this system whenever possible** It's generally a better idea to initialize your learning rate schedule through this system rather than independently, as the system prevents schedule namespace collisions. If you need to know what schedule namespace collisions are, you're having a bad day.

**The method name lies!** While calling `get_last_lr()` on individual `_LRSchedule` schedules works, keep in mind the method name lies! It actually returns the bound schedule feature value, not necessarily learning rate. Getting the feature through `SynchronousSchedule` with `get_last_schedule(name)` doesn't have this problem - it's honest about what it returns.

**Check your serialization.** Make sure your optimizer serialization algorithms are up to snuff. This library injects custom elements into optimizer dictionaries. If you're just serializing the dictionary (like `torch.save(optimizer.state_dict())`), all is well. If you're constructing custom tuples by element name inside an overloaded `.state_dict()', and do not include the new terms, you may have problems. For normal PyTorch optimizers, this should be no issue.

---

## Next Steps

- See [Built-In Schedules](builtin_schedules.md) for complete API reference of builtin schedules with mathematical formulas
- See [Infrastructure](infrastructure.md) for complete API references of the syncronous schedules and helpers.
- Read the [README](README.md) for installation and quick start.