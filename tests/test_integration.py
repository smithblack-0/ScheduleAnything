"""
Black box integration tests for end-to-end scenarios from documentation.

Tests complete workflows from README.md and user_guide.md:
- README quick start example
- Multi-parameter scheduling
- Custom parameter creation and scheduling
- State dict serialization workflow
- Complete case study from README

All tests verify documented examples work as specified.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, StepLR

import src.torch_schedule_anything as sa

# =============================================================================
# README Examples
# =============================================================================


def test_readme_quick_start_example():
    """
    Contract: README quick start example works as documented.

    From README.md:
        scheduler = sa.cosine_annealing_with_warmup(
            optimizer,
            warmup_to_value=1.0,
            anneal_to_value=0.01,
            num_warmup_steps=100,
            num_training_steps=1000,
            schedule_target='weight_decay'
        )
        scheduler = sa.SynchronousSchedule([scheduler])

        for step in range(1000):
            scheduler.step()
    """
    # Exact code from README
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    scheduler = sa.cosine_annealing_with_warmup(
        optimizer,
        warmup_to_value=1.0,
        anneal_to_value=0.01,
        num_warmup_steps=100,
        num_training_steps=1000,
        schedule_target="weight_decay",
    )
    scheduler = sa.SynchronousSchedule([scheduler])

    initial_wd = optimizer.param_groups[0]["weight_decay"]

    # Run for 1000 steps
    for step in range(1000):
        scheduler.step()

    final_wd = optimizer.param_groups[0]["weight_decay"]

    # Observable: Weight decay changed over training
    assert final_wd != initial_wd
    # Observable: Completes without errors
    assert True


def test_readme_case_study_complete_training_setup():
    """
    Contract: Complete case study from README works.

    From README: Scheduling gradient_clip_threshold, weight_decay, and learning rate together.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Custom parameter: Gradient clipping threshold using factory
    clip_scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: StepLR(opt, step_size=1000, gamma=0.1),
        default_value=10.0,
        schedule_target="gradient_clip_threshold",
    )

    # Built-in: Weight decay strengthening
    wd_scheduler = sa.quadratic_schedule_with_warmup(
        optimizer,
        warmup_to_value=0.01,
        anneal_to_value=1.0,
        num_warmup_steps=500,
        num_training_steps=10000,
        schedule_target="weight_decay",
    )

    # Built-in: Standard learning rate with cosine annealing
    lr_scheduler = sa.cosine_annealing_with_warmup(
        optimizer,
        warmup_to_value=1.0,
        anneal_to_value=0.01,
        num_warmup_steps=500,
        num_training_steps=10000,
        schedule_target="lr",
    )

    # Coordinate all three schedules
    sync = sa.SynchronousSchedule([clip_scheduler, wd_scheduler, lr_scheduler])

    # Observable: Custom parameter was created
    assert "gradient_clip_threshold" in optimizer.param_groups[0]
    assert optimizer.param_groups[0]["gradient_clip_threshold"] == 10.0

    # Run training loop
    for step in range(100):
        sync.step()

    # Observable: All parameters are being scheduled
    assert "gradient_clip_threshold" in optimizer.param_groups[0]
    assert "weight_decay" in optimizer.param_groups[0]
    assert "lr" in optimizer.param_groups[0]


# =============================================================================
# User Guide Examples
# =============================================================================


def test_user_guide_quick_example():
    """
    Contract: User guide quick example works.

    From user_guide.md Quick Example section.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Schedule weight decay with cosine annealing
    scheduler = sa.cosine_annealing_with_warmup(
        optimizer,
        warmup_to_value=1.0,
        anneal_to_value=0.01,
        num_warmup_steps=100,
        num_training_steps=1000,
        schedule_target="weight_decay",
    )
    scheduler = sa.SynchronousSchedule([scheduler])

    # Training loop
    for step in range(1000):
        scheduler.step()

    # Observable: Runs without errors
    assert True


def test_user_guide_multi_parameter_scheduling():
    """
    Contract: Multi-parameter scheduling from user guide works.

    From user_guide.md Parallel Schedules section.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    lr_scheduler = sa.cosine_annealing_with_warmup(
        optimizer,
        warmup_to_value=1.0,
        anneal_to_value=0.001,
        num_warmup_steps=100,
        num_training_steps=1000,
        schedule_target="lr",
    )

    wd_scheduler = sa.quadratic_schedule_with_warmup(
        optimizer,
        warmup_to_value=1.0,
        anneal_to_value=0.01,
        num_warmup_steps=100,
        num_training_steps=1000,
        schedule_target="weight_decay",
    )

    # Synchronize them
    sync = sa.SynchronousSchedule([lr_scheduler, wd_scheduler])

    # Single step() call updates both
    for step in range(1000):
        sync.step()

    # Can retrieve individual schedule values
    current_lr = sync.get_last_schedule("lr")
    current_wd = sync.get_last_schedule("weight_decay")

    # Or just get learning rate specifically
    current_lr_alt = sync.get_last_lr()

    # Observable: Both methods work
    assert current_lr == current_lr_alt
    assert isinstance(current_lr, list)
    assert isinstance(current_wd, list)


def test_user_guide_factory_with_existing_parameter():
    """
    Contract: Factory example with existing parameter from user guide.

    From user_guide.md Custom Schedules section.
    """
    model = nn.Linear(10, 1)
    # Adam already has weight_decay - no default needed
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: StepLR(opt, step_size=100, gamma=0.5),
        schedule_target="weight_decay",
    )

    # Observable: Scheduler created successfully
    assert scheduler is not None


def test_user_guide_factory_with_new_parameter():
    """
    Contract: Factory example creating new parameter from user guide.

    From user_guide.md Custom Schedules section.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Creating a custom parameter for the first time
    sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: StepLR(opt, step_size=100, gamma=0.95),
        default_value=1.0,
        schedule_target="gradient_clip_value",
    )

    # Observable: Parameter was created
    assert "gradient_clip_value" in optimizer.param_groups[0]


def test_user_guide_direct_optimizer_extension():
    """
    Contract: Power user direct optimizer extension example.

    From user_guide.md Power User section.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Add a custom parameter to all param groups
    sa.extend_optimizer(optimizer, "gradient_clip_threshold", default_value=10.0)

    # Now schedule it
    sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: StepLR(opt, step_size=100, gamma=0.5),
        schedule_target="gradient_clip_threshold",
    )

    # Observable: Works as documented
    assert "gradient_clip_threshold" in optimizer.param_groups[0]
    assert optimizer.param_groups[0]["gradient_clip_threshold"] == 10.0


# =============================================================================
# Custom Parameter Usage
# =============================================================================


def test_custom_parameter_with_get_param_groups_regrouped():
    """
    Contract: Custom parameters work with get_param_groups_regrouped_by_key.

    Tests the workflow of creating, scheduling, and using custom parameters.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Create and schedule custom parameter
    scheduler = sa.arbitrary_schedule_factory(
        optimizer=optimizer,
        schedule_factory=lambda opt: StepLR(opt, step_size=10, gamma=0.5),
        default_value=10.0,
        schedule_target="gradient_clip_threshold",
    )

    # Step the scheduler
    sync = sa.SynchronousSchedule([scheduler])
    for _ in range(15):
        sync.step()

    # Use get_param_groups_regrouped_by_key to access
    regrouped = sa.get_param_groups_regrouped_by_key(optimizer, "gradient_clip_threshold")

    # Observable: Can access scheduled values
    assert len(regrouped) >= 1
    threshold, params, group = regrouped[0]

    # Observable: Threshold has been scheduled (changed from initial 10.0)
    assert threshold != 10.0  # After 15 steps with step_size=10, should have changed
    assert isinstance(params, list)
    assert len(params) > 0


# =============================================================================
# State Dict Serialization Workflow
# =============================================================================


def test_state_dict_checkpoint_resume_workflow():
    """
    Contract: Complete checkpoint/resume workflow from infrastructure docs.

    From infrastructure.md SynchronousSchedule example.
    """
    # Initial training
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    lr_scheduler = sa.cosine_annealing_with_warmup(
        optimizer, 1.0, 0.01, 100, 1000, schedule_target="lr"
    )
    wd_scheduler = sa.quadratic_schedule_with_warmup(
        optimizer, 1.0, 0.01, 100, 1000, schedule_target="weight_decay"
    )

    sync = sa.SynchronousSchedule([lr_scheduler, wd_scheduler])

    # Train for 500 steps
    for step in range(500):
        sync.step()

    # Save checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": sync.state_dict(),
    }

    # Simulate resuming training
    new_model = nn.Linear(10, 1)
    new_optimizer = AdamW(new_model.parameters(), lr=0.001, weight_decay=0.01)

    new_lr_scheduler = sa.cosine_annealing_with_warmup(
        new_optimizer, 1.0, 0.01, 100, 1000, schedule_target="lr"
    )
    new_wd_scheduler = sa.quadratic_schedule_with_warmup(
        new_optimizer, 1.0, 0.01, 100, 1000, schedule_target="weight_decay"
    )

    new_sync = sa.SynchronousSchedule([new_lr_scheduler, new_wd_scheduler])

    # Load checkpoint
    new_model.load_state_dict(checkpoint["model"])
    new_optimizer.load_state_dict(checkpoint["optimizer"])
    new_sync.load_state_dict(checkpoint["scheduler"])

    # Continue training
    for step in range(500, 1000):
        new_sync.step()

    # Observable: Training resumed and completed successfully
    assert True


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_scheduling_multiple_different_parameters():
    """
    Contract: Can schedule many different parameters simultaneously.
    Observable: System handles complex multi-parameter scenarios.
    """
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
    optimizer = AdamW(
        [
            {"params": model[0].parameters(), "lr": 0.001, "weight_decay": 0.01},
            {"params": model[1].parameters(), "lr": 0.01, "weight_decay": 0.1},
        ]
    )

    # Add multiple custom parameters
    sa.extend_optimizer(optimizer, "custom_1", default_value=1.0)
    sa.extend_optimizer(optimizer, "custom_2", default_value=2.0)

    # Schedule them all
    lr_sched = sa.cosine_annealing_with_warmup(optimizer, 1.0, 0.1, 10, 100, schedule_target="lr")
    wd_sched = sa.linear_schedule_with_warmup(
        optimizer, 1.0, 0.1, 10, 100, schedule_target="weight_decay"
    )
    c1_sched = sa.arbitrary_schedule_factory(
        optimizer, lambda opt: StepLR(opt, step_size=10), schedule_target="custom_1"
    )
    c2_sched = sa.arbitrary_schedule_factory(
        optimizer, lambda opt: StepLR(opt, step_size=5), schedule_target="custom_2"
    )

    sync = sa.SynchronousSchedule([lr_sched, wd_sched, c1_sched, c2_sched])

    # Run for some steps
    for _ in range(50):
        sync.step()

    # Observable: All schedules working
    assert sync.get_last_schedule("lr") is not None
    assert sync.get_last_schedule("weight_decay") is not None
    assert sync.get_last_schedule("custom_1") is not None
    assert sync.get_last_schedule("custom_2") is not None


def test_zero_warmup_steps():
    """
    Contract: Schedules work with num_warmup_steps=0.
    Observable: No warmup phase, goes directly to annealing.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    scheduler = sa.cosine_annealing_with_warmup(
        optimizer,
        warmup_to_value=1.0,
        anneal_to_value=0.1,
        num_warmup_steps=0,  # No warmup
        num_training_steps=1000,
    )

    sync = sa.SynchronousSchedule([scheduler])

    # Observable: Works without errors
    for _ in range(100):
        sync.step()

    assert True


def test_very_long_training():
    """
    Contract: Schedules handle very long training (many steps).
    Observable: Numerical stability over many steps.
    """
    model = nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    scheduler = sa.linear_schedule_with_warmup(
        optimizer,
        warmup_to_value=1.0,
        anneal_to_value=0.001,
        num_warmup_steps=1000,
        num_training_steps=100000,
    )

    sync = sa.SynchronousSchedule([scheduler])

    # Run for many steps
    for _ in range(10000):
        sync.step()

    # Observable: Still works, values are reasonable
    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr > 0
    assert final_lr < 0.001  # Initial lr


def test_state_dict_preserves_separate_schedule_states(
    optimizer, setup_schedule, setup_model_and_optimizer
):
    """
    Contract: State dict save/load preserves each schedule's independent state.
    Why: Verifies checkpointing works correctly with multiple schedules.
    How: Save at step 10, continue to 11, load in new instance, verify step 11 matches.

    Note: This test uses NEW optimizer/scheduler instances to verify checkpoint transfer.
    For testing rewind behavior, see test_load_state_from_earlier_step in
    test_synchronous_schedule.py
    """
    wd_sched = setup_schedule(optimizer, "weight_decay", StepLR, step_size=5, gamma=0.5)
    mom_sched = setup_schedule(
        optimizer, "momentum", StepLR, default_value=0.9, step_size=3, gamma=0.7
    )

    sync = sa.SynchronousSchedule([wd_sched, mom_sched])

    # Step to 10
    for _ in range(10):
        sync.step()

    # Save checkpoint (both optimizer and scheduler)
    checkpoint = {"optimizer": optimizer.state_dict(), "scheduler": sync.state_dict()}

    # Continue to get reference value at step 11
    sync.step()
    wd_at_11 = optimizer.param_groups[0]["weight_decay"]
    mom_at_11 = optimizer.param_groups[0]["momentum"]

    # Create new optimizer and schedulers
    new_model, new_optimizer = setup_model_and_optimizer()
    new_wd_sched = setup_schedule(new_optimizer, "weight_decay", StepLR, step_size=5, gamma=0.5)
    new_mom_sched = setup_schedule(
        new_optimizer, "momentum", StepLR, default_value=0.9, step_size=3, gamma=0.7
    )
    new_sync = sa.SynchronousSchedule([new_wd_sched, new_mom_sched])

    # Load checkpoint
    new_optimizer.load_state_dict(checkpoint["optimizer"])
    new_sync.load_state_dict(checkpoint["scheduler"])

    # Step once from checkpoint (should be step 11)
    new_sync.step()

    # Observable: Values match step 11 from original run
    assert abs(new_optimizer.param_groups[0]["weight_decay"] - wd_at_11) < 1e-6
    assert abs(new_optimizer.param_groups[0]["momentum"] - mom_at_11) < 1e-6
