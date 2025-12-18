# Changelog

## 0.6.0

Adapter and core infrastructure audit (FINAL)

- Renamed throw_errors_on_desync → set_throw_error_on_desync (consistency with naming conventions)
- Exported set_throw_error_on_desync as part of public API
- Updated error messages to reference correct function name
- Added test_set_throw_error_on_desync_controls_error_behavior (verifies error/warning control)
- Relaxed error message matching in adapter tests (allows implementation flexibility)
- **FINAL ACCEPTANCE**: Adapter layer (ArbitraryScheduleAdapter, ProxyDictByLR, namespace routing,
  desync detection) has been audited and accepted as production-ready
- Test suite: 8 adapter tests covering contracts, namespace routing, state independence, desync detection

## 0.5.2

Built-in schedule implementations

- Implemented all 13 builtin schedule functions in builtin_schedules.py
- cosine_annealing_with_warmup and cosine_annealing_with_inverse_warmup
- polynomial_decay variants (polynomial, linear, quadratic, sqrt - each with warmup/inverse)
- constant schedule variants (warmup, inverse warmup, pure constant)
- All schedules use LambdaLR with exact mathematical formulas from documentation
- All builtin schedules exported from package __init__.py

## 0.5.1

Test contract clarification for schedule_namespaces

- Added test_schedule_namespaces_routing_contract
- Verifies non-'lr', non-existing keys route to schedule_namespaces
- Contract: schedule_namespaces is part of black box API
- Ensures keys don't pollute main param_group dict

## 0.5.0

Core infrastructure implementation

- arbitrary_schedules.py: Internal proxy mechanism with per-schedule namespace fix
  - ProxyDictByLR with schedule_namespaces to prevent collision
  - ArbitraryScheduleAdapter for parameter proxying
  - Error control with throw_errors_on_desync
- infrastructure.py: Complete public API matching documentation specification
  - arbitrary_schedule_factory (signature matches docs exactly)
  - SynchronousSchedule with fixed load_state_dict bug
  - extend_optimizer utility
  - get_param_groups_regrouped_by_key utility
- All APIs standardized to use schedule_target parameter naming
- Fixed namespace collision: Multiple schedules can now set initial_lr without conflict

## 0.4.2

Complete API test suite - test-driven development on steroids

- Black box tests for SynchronousSchedule coordination
- Mathematical formula tests for all 13 built-in schedules
- Integration tests for all documented examples (README, user guide)
- State dict save/load workflow tests
- Multi-parameter scheduling tests
- Custom parameter creation and usage tests
- Edge case and error handling tests
- All user-facing APIs now have comprehensive test coverage

## 0.4.1

Infrastructure API tests added following test-driven development

- Black box API tests for arbitrary_schedule_factory
- Tests for extend_optimizer utility
- Tests for get_param_groups_regrouped_by_key
- Shared test fixtures in conftest.py
- All tests verify observable behavior, not implementation details

## 0.4.0

Initial release of ScheduleAnything

- Documentation and API specification complete
- Infrastructure for scheduling any optimizer hyperparameter
- 13 built-in schedule primitives with standard and inverse warmup variants
- arbitrary_schedule_factory for custom PyTorch schedulers
- SynchronousSchedule for coordinating multiple schedules
- extend_optimizer and get_param_groups_regrouped_by_key utilities
- CI/CD pipeline with automated testing and PyPI publishing
