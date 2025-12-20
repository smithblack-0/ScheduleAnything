# Infrastructure API Reference

This document covers the low-level infrastructure classes and utilities that power ScheduleAnything. These are the building blocks used internally by the built-in schedules, but are also exposed for advanced use cases.

**When to use these:**

- **`arbitrary_schedule_factory`** - When you need to use a custom PyTorch scheduler on any parameter
- **`SynchronousSchedule`** - When coordinating multiple schedules (recommended for all use cases)
- **`extend_optimizer`** - When manually adding custom parameters before scheduling
- **`get_param_groups_regrouped_by_key`** - When your training logic needs to respond to scheduled parameter values

Most users will primarily use `SynchronousSchedule` for coordination and occasionally `arbitrary_schedule_factory` for custom schedulers. The other utilities are for power users building optimizer extensions or custom training loops.

## Navigation

- **[Examples & Tutorials](examples_and_tutorials.md)** - Hands-on Colab notebooks with step-by-step tutorials
- **[User Guide](user_guide.md)** - Complete usage guide and concepts
- **[Built-In Schedules](builtin_schedules.md)** - API reference for all builtin schedules
- **[README](../README.md)** - Installation and quick start
---

## The arbitrary_schedule_factory

The `arbitrary_schedule_factory` binds any PyTorch scheduler to any optimizer parameter using a factory callback pattern. It is up to the user to define a correct factory:

```python
arbitrary_schedule_factory(
    optimizer: Optimizer,
    schedule_factory: Callable[[Optimizer], _LRScheduler],
    default_value: Optional[float] = None,
    schedule_target: str = 'lr'
) -> _LRScheduler
```

**Parameters:**
- **`optimizer`** - The PyTorch optimizer to bind the schedule to
- **`schedule_factory`** - A callable that takes an optimizer and returns a PyTorch scheduler. This allows you to configure the scheduler with any parameters you need.
- **`default_value`** - Optional initial value if the parameter doesn't exist in the optimizer. If the parameter already exists, this is ignored.
- **`schedule_target`** - Which optimizer parameter to schedule. Default: `'lr'`

**Returns:**
- `_LRScheduler` - A PyTorch scheduler that modifies the specified parameter

**Behavior:**
- If `schedule_target` doesn't exist in param_groups and `default_value` is provided, creates the parameter with that value
- Wraps the optimizer in an adapter that makes the target parameter schedulable
- Returns a standard PyTorch scheduler interface that you step normally

**Example:**

```python
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch_schedule_anything as tsa

optimizer = Adam(model.parameters(), lr=0.001)

# Schedule momentum using StepLR
momentum_scheduler = tsa.arbitrary_schedule_factory(
    optimizer=optimizer,
    schedule_factory=lambda opt: StepLR(opt, step_size=30, gamma=0.5),
    default_value=0.9,  # Initialize momentum to 0.9
    schedule_target='momentum'
)

# Use it like any PyTorch scheduler
for epoch in range(100):
    # training...
    momentum_scheduler.step()
```

---

## The SynchronousSchedule Object

The `SynchronousSchedule` object has two primary responsibilities:

* Coordinate multiple schedules to step in lockstep, passing on epoch as needed
* Remap their apis to be accessed in a sane way that does not lie

In exchange, the user has to run their schedules through it. 

```python
SynchronousSchedule(schedules: List[_LRScheduler])
```

**Constructor Parameters:**
- **`schedules`** - List of PyTorch schedulers to coordinate. Can mix built-in ScheduleAnything schedules with raw PyTorch schedulers.

**Properties:**
- **`schedule_names`** (`List[str]`) - Names of all managed schedules. Standard PyTorch schedulers are named `'lr'`, arbitrary schedules use their `schedule_target` name.

**Methods:**

### step

Step is invoked as

`step(epoch: Optional[int] = None)`

Step all managed schedulers together.

**Parameters:**
- **`epoch`** - Optional epoch number to pass to schedulers

**Example:**
```python
sync = tsa.SynchronousSchedule([lr_scheduler, wd_scheduler])
for epoch in range(100):
    # training...
    sync.step()
```

### get_last_schedule

Get last schedule is a more generic version of `get_last_lr` as in torch. It is invoked as:

`get_last_schedule(name: str) -> List[float]`

Get the last scheduled values for a specific parameter. The system is smart enough to 
infer what that was.

**Parameters:**
- **`name`** - The schedule name (e.g., `'lr'`, `'weight_decay'`, `'momentum'`)

**Returns:**
- `List[float]` - List of values, one per parameter group

**Example:**
```python
current_lr = sync.get_last_schedule('lr')
current_wd = sync.get_last_schedule('weight_decay')
```

### get_last_lr

Convenience method to get the last learning rate values specifically.
Helps prevent breaking your existing logging system. Used as:

`get_last_lr() -> List[float]`

**Returns:**
- `List[float]` - Learning rate values for each parameter group

**Example:**
```python
current_lr = sync.get_last_lr()
```

### state_dict

Saves the state of all managed schedulers. 

`state_dict() -> Dict[str, Any]`


**Returns:**
- `Dict[str, Any]` - Dictionary mapping schedule names to their state dicts with scheduler aux information.

### load_state_dict

Resumes state. Used as:

`load_state_dict(state_dict: Dict[str, Any])`

Restore the state of all managed schedulers.

**Parameters:**
- **`state_dict`** - Dictionary previously returned by `state_dict()`

**Key Behaviors:**
- Automatically assigns names to schedules: `'lr'` for standard PyTorch schedulers, `schedule_target` for arbitrary schedules
- Provides honest API - `get_last_schedule('weight_decay')` actually returns weight decay, not a value pretending to be learning rate
- Ensures state dict save/load works correctly for all managed schedules

**Complete Example:**

```python
import torch_schedule_anything as tsa

# Create multiple schedules
lr_scheduler = tsa.cosine_annealing_with_warmup(
    optimizer, 1, 0.01, 100, 1000, schedule_target='lr'
)
wd_scheduler = tsa.quadratic_schedule_with_warmup(
    optimizer, 0.1, 1.0, 100, 1000, schedule_target='weight_decay'
)

# Coordinate them
sync = tsa.SynchronousSchedule([lr_scheduler, wd_scheduler])

# Training loop
for step in range(1000):
    # training code...
    sync.step()
    
    # Access current values
    if step % 100 == 0:
        print(f"LR: {sync.get_last_lr()}")
        print(f"WD: {sync.get_last_schedule('weight_decay')}")

# Save state
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': sync.state_dict()
}
torch.save(checkpoint, 'checkpoint.pt')

# Load state
checkpoint = torch.load('checkpoint.pt')
sync.load_state_dict(checkpoint['scheduler'])
```

---

## extend_optimizer

A tool for real power users. Add a custom parameter to all param_groups in an optimizer.

```python
extend_optimizer(
    optimizer: Optimizer,
    name: str,
    default_value: Number,
    overwrite_values: bool = False
) -> Optimizer
```

**Parameters:**
- **`optimizer`** - The optimizer to extend
- **`name`** - Name of the parameter to add
- **`default_value`** - Value to set for the parameter
- **`overwrite_values`** - If `True`, replaces existing values with `default_value`. If `False` (default), only adds if parameter is missing.

**Returns:**
- `Optimizer` - The same optimizer instance (modified in-place)

**Behavior:**
- Iterates through all param_groups in the optimizer
- If `overwrite_values=False`: only sets the parameter if it doesn't exist
- If `overwrite_values=True`: always sets the parameter to `default_value`
- Validates that `default_value` is a `Number` type

**Example:**

```python
from torch.optim import SGD
import torch_schedule_anything as tsa

optimizer = SGD(model.parameters(), lr=0.1)

# Add custom parameter
tsa.extend_optimizer(optimizer, 'gradient_clip_threshold', default_value=10.0)

# Now gradient_clip_threshold exists in all param_groups
print(optimizer.param_groups[0]['gradient_clip_threshold'])  # 10.0

# Can now schedule it
clip_scheduler = tsa.arbitrary_schedule_factory(
    optimizer,
    lambda opt: CosineAnnealingLR(opt, T_max=1000),
    schedule_target='gradient_clip_threshold'
)
```

**Use Cases:**
- Pre-initializing custom parameters before creating schedules
- Building optimizer wrappers that inject custom behavior
- Resetting parameter values across all param_groups

---

## get_param_groups_regrouped_by_key
Extract and organize param_groups by a specific parameter's value. Usable to reduce the pain of writing custom logic 
that interacts and uses extensions.

```python
get_param_groups_regrouped_by_key(
    optimizer: Optimizer,
    schedule_target: str
) -> List[Tuple[Any, List[Parameter], Dict]]
```

**Parameters:**
- **`optimizer`** - The optimizer to extract from
- **`schedule_target`** - Which parameter to organize by (e.g., `'lr'`, `'weight_decay'`, `'gradient_clip_threshold'`)

**Returns:**
- `List[Tuple[Any, List[Parameter], Dict]]` - List of tuples, where each tuple contains:
  - **value** - The scheduled parameter's value for this group
  - **params** - List of `nn.Parameter` objects in this group
  - **group** - The complete param_group dictionary (in case you need other fields)

**Behavior:**
- Works with both native PyTorch parameters (`'lr'`, `'weight_decay'`, `'momentum'`) and custom extended parameters
- Returns data organized for easy iteration by parameter value
- Useful when training logic needs to respond to scheduled values

**Example: Gradient Clipping**

```python
def my_custom_gradient_clipping(optimizer):
    """Apply gradient clipping with per-group scheduled thresholds."""
    for threshold, params, group in tsa.get_param_groups_regrouped_by_key(
        optimizer, 'gradient_clip_threshold'
    ):
        torch.nn.utils.clip_grad_norm_(params, max_norm=threshold)

# In training loop
optimizer.zero_grad()
loss.backward()
my_custom_gradient_clipping(optimizer)  # Uses scheduled thresholds
optimizer.step()
```

**Example: Logging Parameter Values**

```python
def log_scheduled_parameters(optimizer, step):
    """Log all scheduled parameter values."""
    # Learning rates
    for lr, params, group in tsa.get_param_groups_regrouped_by_key(optimizer, 'lr'):
        print(f"Step {step}, LR: {lr}, Params: {len(params)}")
    
    # Weight decays
    for wd, params, group in tsa.get_param_groups_regrouped_by_key(optimizer, 'weight_decay'):
        print(f"Step {step}, WD: {wd}, Params: {len(params)}")
```

**Example: Conditional Operations**

```python
def apply_conditional_regularization(optimizer):
    """Apply different regularization based on weight decay value."""
    for wd, params, group in tsa.get_param_groups_regrouped_by_key(
        optimizer, 'weight_decay'
    ):
        if wd > 0.05:
            # Strong regularization regime
            apply_additional_constraints(params)
        else:
            # Weak regularization regime
            pass
```

---

## See Also

- **[Examples & Tutorials](examples_and_tutorials.md)** - Hands-on Colab notebooks with step-by-step tutorials
- **[User Guide](user_guide.md)** - Complete usage guide and concepts
- **[Built-In Schedules](builtin_schedules.md)** - API reference for all builtin schedules
- **[README](../README.md)** - Installation and quick start