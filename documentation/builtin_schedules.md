# Schedule API Reference

Complete guide to all built-in schedules in `torch-schedule-anything`.

This library provides curve primitives for scheduling any optimizer hyperparameter. These are research tools - you choose the curve shape based on your optimization problem, not based on prescriptive "use X for Y" rules.


## What is a Schedule?

The term 'Schedule' as defined in this library is the same as in canonical PyTorch: A multiplier that is applied to an initial hyperparameter. PyTorch schedulers work by computing a **multiplier** λ(t) that gets applied to initial parameter values:

$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

To maintain maximal compatibility, we adopt PyTorch's conventions.

**This means schedules set λ(t), NOT absolute values.**

When you specify `warmup_to_value=1.0` and `anneal_to_value=0.1`, you're defining the multiplier curve. If your parameter starts at 0.001, the actual values will be 0.001 → 0.0001.

**If you want absolute behavior, make sure to initialize new schedule hyperparameters to 1.0**


## Navigation

- **[Examples & Tutorials](examples_and_tutorials.md)** - Hands-on Colab notebooks with step-by-step tutorials
- **[User Guide](user_guide.md)** - Complete usage guide and concepts
- **[Infrastructure](infrastructure.md)** - API reference for utilities and arbitrary schedule factories
- **[README](../README.md)** - Installation and quick start


---
## Parameter Glossary

All values that sound absolute are setting the multiplier **lambda(t)** not the value **value(t)**

**Function Parameters:**
- **`optimizer`**: The PyTorch optimizer to schedule
- **`warmup_to_value`**: Target value at the end of the warmup phase
- **`anneal_to_value`**: Final value at the end of training
- **`num_warmup_steps`**: Number of steps for the warmup phase
- **`num_training_steps`**: Total number of training steps
- **`warmup_multiplier`**: (Inverse warmup only) Starting multiplier - begins at `warmup_to_value * warmup_multiplier`. Default: 20.0
- **`polynomial_exponent`**: Controls curve shape - higher values create slower initial change followed by rapid change; lower values create rapid initial decay followed by slower change
- **`value`**: (Constant schedule) Fixed value throughout training
- **`schedule_target`**: Which optimizer parameter to schedule. Default: `'lr'`

**Mathematical Notation (used in formulas below):**
- **`t`**: Current step, whatever the torch scheduler or user decides it is. WARNING: PyTorch initializes this at 1.
- **`initial_hyperparameter_value`**: The initial value of the target parameter in the optimizer's param_groups before the schedule is attached
- **`num_training_steps`**: Number of training batches or epochs.

Schedules are stepped once per training step, as defined by the caller. Whether a training step corresponds to a batch, an epoch, or something else entirely is determined by the user, not this library.

---
## Understanding Inverse Warmup

Standard warmup starts at zero and increases to a target value. **Inverse warmup** does the opposite: starts at a high value (`warmup_to_value * warmup_multiplier`) and linearly decreases to the baseline (`warmup_to_value`).

Inverse warmup is useful when **low values are more constraining**. Early in training, models are unstable and benefit from being "permissive." As training progresses, you tighten constraints.

**Examples where inverse warmup makes sense:**
- **Gradient clipping**: Start with high threshold (permissive, let gradients through) → decay to low threshold (restrictive, clip more aggressively)
- **Noise schedules**: Start with high noise (permissive exploration) → reduce noise (exploit learned patterns)
- **Certain regularization parameters**: Where lower values impose stronger constraints

After the inverse warmup phase completes, the schedule continues with its normal behavor. Usually this means following the annealing/decay behavior from `warmup_to_value` down to `anneal_to_value`.

---

## Curve Primitives

The built-in schedules provide different curve shapes. Choose based on the decay profile you need:

**Cosine curves** - Smooth S-shaped transitions with no sharp changes. The cosine function naturally provides smooth acceleration and deceleration.

**Linear curves** - Constant rate of change. Predictable, uniform decay.

**Quadratic curves** - Accelerating decay. Changes slowly early, rapidly later. Good when you want stability followed by aggressive adjustment.

**Square root curves** - Decelerating decay. Changes rapidly early, slowly later. Good when you want quick initial adjustment that stabilizes.

**Polynomial curves** - Arbitrary exponents for custom curve shapes. Use when you need a much more aggressive or much slower curve than the standard options provide. 

**Constant curves** - Flat after warmup. Use when you want a parameter inactive initially, then locked at a fixed value. There is also a variant that is just constant.

If you need a different primitive, read the [User Guide](user_guide.md) on how to make your own schedule

---

## Schedules API Reference

### Cosine Schedules

#### `cosine_annealing_with_warmup`

Smooth warmup from zero followed by smooth cosine decay.

```python
cosine_annealing_with_warmup(
    optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- A = anneal_to_value
- L = num_warmup_steps
- M = num_training_steps

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
\frac{W \cdot t}{L} & \text{if } t \leq L \\
A + (W - A) \cdot \cos\left(\frac{\pi}{2} \cdot \frac{t - L}{M - L}\right) & \text{otherwise}
\end{cases}
$$

**Example:**
```python
scheduler = tsa.cosine_annealing_with_warmup(
    optimizer,
    warmup_to_value=1.0,
    anneal_to_value=0.001,
    num_warmup_steps=100,
    num_training_steps=1000,
    schedule_target='lr'
)
```

---

#### `cosine_annealing_with_inverse_warmup`

Inverse warmup (high to baseline) followed by smooth cosine decay.

```python
cosine_annealing_with_inverse_warmup(
    optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    warmup_multiplier: float = 20.0,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- A = anneal_to_value
- L = num_warmup_steps
- M = num_training_steps
- R = warmup_multiplier

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
W \cdot \left(R - (R-1) \cdot \frac{t}{L}\right) & \text{if } t \leq L \\
A + (W - A) \cdot \cos\left(\frac{\pi}{2} \cdot \frac{t - L}{M - L}\right) & \text{otherwise}
\end{cases}
$$

**Example:**
```python
# Gradient clipping: start permissive, tighten over training
scheduler = tsa.cosine_annealing_with_inverse_warmup(
    optimizer,
    warmup_to_value=5.0,
    anneal_to_value=0.5,
    num_warmup_steps=100,
    num_training_steps=1000,
    warmup_multiplier=10.0,  # Starts at 50.0
    schedule_target='gradient_clip_threshold'
)
```

---

### Polynomial Schedules

#### `polynomial_schedule_with_warmup`

Warmup followed by polynomial decay with customizable exponent.

```python
polynomial_schedule_with_warmup(
    optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    polynomial_exponent: float = 2.0,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- A = anneal_to_value
- L = num_warmup_steps
- M = num_training_steps
- P = polynomial_exponent

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
\frac{W \cdot t}{L} & \text{if } t \leq L \\
A + (W - A) \cdot \left(1 - \frac{t - L}{M - L}\right)^P & \text{otherwise}
\end{cases}
$$

---

#### `polynomial_schedule_with_inverse_warmup`

Inverse warmup followed by polynomial decay.

```python
polynomial_schedule_with_inverse_warmup(
    optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    polynomial_exponent: float = 2.0,
    warmup_multiplier: float = 20.0,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- A = anneal_to_value
- L = num_warmup_steps
- M = num_training_steps
- P = polynomial_exponent
- R = warmup_multiplier

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
W \cdot \left(R - (R-1) \cdot \frac{t}{L}\right) & \text{if } t \leq L \\
A + (W - A) \cdot \left(1 - \frac{t - L}{M - L}\right)^P & \text{otherwise}
\end{cases}
$$

---

#### `linear_schedule_with_warmup`

Warmup followed by constant-rate linear decay. Equivalent to `polynomial_schedule_with_warmup` with `polynomial_exponent=1.0`.

```python
linear_schedule_with_warmup(
    optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- A = anneal_to_value
- L = num_warmup_steps
- M = num_training_steps

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
\frac{W \cdot t}{L} & \text{if } t \leq L \\
A + (W - A) \cdot \left(1 - \frac{t - L}{M - L}\right) & \text{otherwise}
\end{cases}
$$

---

#### `linear_schedule_with_inverse_warmup`

Inverse warmup followed by linear decay.

```python
linear_schedule_with_inverse_warmup(
    optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    warmup_multiplier: float = 20.0,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- A = anneal_to_value
- L = num_warmup_steps
- M = num_training_steps
- R = warmup_multiplier

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
W \cdot \left(R - (R-1) \cdot \frac{t}{L}\right) & \text{if } t \leq L \\
A + (W - A) \cdot \left(1 - \frac{t - L}{M - L}\right) & \text{otherwise}
\end{cases}
$$

---

#### `quadratic_schedule_with_warmup`

Warmup followed by quadratic decay (accelerating curve). Equivalent to `polynomial_schedule_with_warmup` with `polynomial_exponent=2.0`.

```python
quadratic_schedule_with_warmup(
    optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- A = anneal_to_value
- L = num_warmup_steps
- M = num_training_steps

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
\frac{W \cdot t}{L} & \text{if } t \leq L \\
A + (W - A) \cdot \left(1 - \frac{t - L}{M - L}\right)^2 & \text{otherwise}
\end{cases}
$$

---

#### `quadratic_schedule_with_inverse_warmup`

Inverse warmup followed by quadratic decay.

```python
quadratic_schedule_with_inverse_warmup(
    optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    warmup_multiplier: float = 20.0,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- A = anneal_to_value
- L = num_warmup_steps
- M = num_training_steps
- R = warmup_multiplier

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
W \cdot \left(R - (R-1) \cdot \frac{t}{L}\right) & \text{if } t \leq L \\
A + (W - A) \cdot \left(1 - \frac{t - L}{M - L}\right)^2 & \text{otherwise}
\end{cases}
$$

---

#### `sqrt_schedule_with_warmup`

Warmup followed by square root decay (decelerating curve). Equivalent to `polynomial_schedule_with_warmup` with `polynomial_exponent=0.5`.

```python
sqrt_schedule_with_warmup(
    optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- A = anneal_to_value
- L = num_warmup_steps
- M = num_training_steps

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
\frac{W \cdot t}{L} & \text{if } t \leq L \\
A + (W - A) \cdot \left(1 - \frac{t - L}{M - L}\right)^{0.5} & \text{otherwise}
\end{cases}
$$

---

#### `sqrt_schedule_with_inverse_warmup`

Inverse warmup followed by square root decay.

```python
sqrt_schedule_with_inverse_warmup(
    optimizer,
    warmup_to_value: float,
    anneal_to_value: float,
    num_warmup_steps: int,
    num_training_steps: int,
    warmup_multiplier: float = 20.0,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- A = anneal_to_value
- L = num_warmup_steps
- M = num_training_steps
- R = warmup_multiplier

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
W \cdot \left(R - (R-1) \cdot \frac{t}{L}\right) & \text{if } t \leq L \\
A + (W - A) \cdot \left(1 - \frac{t - L}{M - L}\right)^{0.5} & \text{otherwise}
\end{cases}
$$

---

### Constant Schedules

#### `constant_with_warmup`

Linear warmup to target value, then holds constant.

```python
constant_with_warmup(
    optimizer,
    warmup_to_value: float,
    num_warmup_steps: int,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- L = num_warmup_steps

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
\frac{W \cdot t}{L} & \text{if } t \leq L \\
W & \text{otherwise}
\end{cases}
$$

---

#### `constant_with_inverse_warmup`

Inverse warmup to target value, then holds constant.

```python
constant_with_inverse_warmup(
    optimizer,
    warmup_to_value: float,
    num_warmup_steps: int,
    warmup_multiplier: float = 20.0,
    schedule_target: str = 'lr'
)
```

**Let:**
- W = warmup_to_value
- L = num_warmup_steps
- R = warmup_multiplier

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = \begin{cases}
W \cdot \left(R - (R-1) \cdot \frac{t}{L}\right) & \text{if } t \leq L \\
W & \text{otherwise}
\end{cases}
$$

---

#### `constant_schedule`

Single fixed value throughout training.

```python
constant_schedule(
    optimizer,
    value: float,
    schedule_target: str = 'lr'
)
```

**Let:**
- V = value

**Formula:** 
$$
\text{value}(t) = `\text{initial_hyperparameter_value}` \times \lambda(t)
$$

**where:**
$$
\lambda(t) = V
$$

---

## Next Steps

- See [User Guide](user_guide.md) for how to use the system.
- See [Infrastructure](infrastructure.md) for complete API references of the syncronous schedules and helpers.
- Read the [README](README.md) for installation and quick start.
