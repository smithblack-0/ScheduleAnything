# Examples and Tutorials

This page organizes hands-on examples and tutorials for `torch-schedule-anything`. Examples are being developed gradually, with each example accompanied by a detailed Medium article explaining the concepts and implementation.

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
- Best practices for checkpoint/resume with scheduled parameters

**Medium Article:** *(Coming soon)*

**What you'll learn:**
- How to use `arbitrary_schedule_factory` to schedule weight decay
- How to coordinate multiple schedules with `SynchronousSchedule`
- How to build a training loop that works with scheduled parameters
- When and why to schedule weight decay

**Prerequisites:** Basic PyTorch knowledge, understanding of optimizers and training loops

---

## Coming Soon

More examples are in development! Future tutorials will cover:
- Advanced multi-parameter scheduling patterns
- Custom schedule curves for research applications
- Integration with existing training frameworks
- Scheduling novel optimizer parameters (momentum, gradient clipping, etc.)
- State persistence and checkpoint strategies

Check back weekly for new examples!

---

## Navigation

- [User Guide](user_guide.md) - Complete usage guide and concepts
- [Built-In Schedules](builtin_schedules.md) - API reference for all built-in schedules
- [Infrastructure API](infrastructure.md) - Low-level infrastructure documentation
- [README](../README.md) - Installation and quick start

---

*Have an example request? Open an issue on [GitHub](https://github.com/smithblack-0/ScheduleAnything/issues)!*
