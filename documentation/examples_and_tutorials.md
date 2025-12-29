# Examples and Tutorials

This page organizes hands-on examples and tutorials for `torch-schedule-anything`. Examples are being developed gradually, with each example accompanied by a detailed Medium article regarding the concepts and implementation.

## About These Examples

These examples demonstrate real-world usage patterns and are organized by capability. Each example is a complete, runnable Jupyter notebook that you can open in Google Colab and experiment with immediately.

Examples are released incrementally - check back regularly for new tutorials covering advanced scheduling patterns, multi-parameter coordination, and integration strategies.

## Getting Started Examples

### Basic Weight Decay Scheduling

**[Open in Colab](https://colab.research.google.com/github/smithblack-0/ScheduleAnything/blob/master/examples/basic_weight_decay_scheduling.ipynb)**

Learn how to schedule weight decay alongside learning rate in a custom training loop. This example demonstrates:
- Setting up schedules for multiple optimizer parameters
- Building a custom training loop with scheduled hyperparameters
- Coordinating weight decay and learning rate schedules


**Medium Article:** [ScheduleAnything: If you can dream it, it can schedule it.
](https://medium.com/@chris.oquinn2/scheduleanything-if-you-can-dream-it-it-can-schedule-it-574fb3682ea1?postPublishedType=repub)

**What you'll learn:**
- How to use `arbitrary_schedule_factory` to schedule weight decay
- How to coordinate multiple schedules with `SynchronousSchedule`
- How to build a training loop that works with scheduled parameters
- When and why to schedule weight decay

**Prerequisites:** Basic PyTorch knowledge, understanding of optimizers and training loops

---

## Capabilities Series

These examples demonstrate the breadth of what can be scheduled using ScheduleAnything, from standard optimizer parameters to custom extensions.

### Weight Decay and Learning Rate Scheduling

**[Open in Colab](https://colab.research.google.com/github/smithblack-0/ScheduleAnything/blob/master/examples/ScheduleAnything_Weight_Decay_Example.ipynb)**

A comprehensive example showing concurrent scheduling of learning rate and weight decay with different curve types in a custom training loop.

**What you'll learn:**
- Scheduling learning rate with cosine annealing
- Scheduling weight decay with linear inverse warmup
- Coordinating multiple schedules with different curves
- Best practices for multi-parameter scheduling in NLP tasks

**Prerequisites:** PyTorch fundamentals, HuggingFace Transformers basics, understanding of weight decay

---

### Logical Batch Size Scheduling

**[Open in Colab](https://colab.research.google.com/github/smithblack-0/ScheduleAnything/blob/master/examples/ScheduleAnything_Logical_Batch_Size_Example.ipynb)**

Learn to schedule logical batch size through gradient accumulation, demonstrating how to extend optimizers with custom parameters.

**What you'll learn:**
- Using `extend_optimizer` to add custom parameters
- Scheduling logical batch size via gradient accumulation
- Responding to scheduled values in the training loop with `get_param_groups_regrouped_by_key`
- Implementing variable batch size training strategies

**Prerequisites:** Understanding of gradient accumulation, PyTorch training loops

---

### Gradient Norm Threshold Scheduling

**[Open in Colab](https://colab.research.google.com/github/smithblack-0/ScheduleAnything/blob/master/examples/ScheduleAnything_Gradient_Norm_Threshold_Scheduling_Example.ipynb)**

Advanced example implementing Gradient Norm Threshold Scheduling (GNTS) by extending the optimizer with custom scheduling parameters.

**What you'll learn:**
- Creating custom optimizer parameters for advanced training techniques
- Implementing conditional optimizer steps based on scheduled thresholds
- Using `extend_optimizer` for advanced optimizer extensions
- Coordinating multiple schedules including custom parameters

**Prerequisites:** Advanced PyTorch knowledge, understanding of gradient norms and clipping

---

## Framework Integration Series

Learn how to integrate ScheduleAnything with popular training frameworks for production use.

### HuggingFace Trainer Integration

**[Open in Colab](https://colab.research.google.com/github/smithblack-0/ScheduleAnything/blob/master/examples/Medium_Huggingface_Trainer_ScheduleAnything.ipynb)**

Production-ready example showing how to integrate ScheduleAnything with HuggingFace Trainer to schedule arbitrary optimizer parameters.

**Medium Article:** [Using ScheduleAnything with HuggingFace Trainer: Implementing Aggressive Weight Decay Scheduling](https://medium.com/@chris.oquinn2/using-scheduleanything-with-huggingface-train-implementing-aggressive-weight-decay-scheduling-914abc254564)

**What you'll learn:**
- Integrating custom schedulers with HuggingFace Trainer
- Scheduling weight decay in production training pipelines
- Best practices for parameter group management
- Zero-modification integration patterns for existing frameworks

**Prerequisites:** HuggingFace Transformers and Trainer API experience, production training workflow knowledge

---

## Navigation

- [User Guide](user_guide.md) - Complete usage guide and concepts
- [Built-In Schedules](builtin_schedules.md) - API reference for all builtin schedules
- [Infrastructure API](infrastructure.md) - API reference for utilities and arbitrary schedule factories
- [README](../README.md) - Installation and quick start

---

*Have an example request? Open an issue on [GitHub](https://github.com/smithblack-0/ScheduleAnything/issues)!*
