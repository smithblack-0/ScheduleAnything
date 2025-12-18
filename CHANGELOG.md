# Changelog

## 0.7.1


## 0.7.0

- Ruff and black formatting fixes
- Fixed issues in arbitrary_schedules with messages being too long

## 0.6.3

### Changed
- Corrected major deficiencies in schedule contract tests.
- All built-in schedules now explicitly test contract invariants at:
  - step 0
  - warmup completion (`t = num_warmup_steps`)
  - immediate post-warmup (`t = num_warmup_steps + 1`)
  - final step (`t = num_training_steps`)
- Added interior-point checks for both warmup and annealing phases where applicable.
- Added explicit endpoint assertions (e.g. `λ(M) == anneal_to_value`).
- Added equivalence tests for alias schedules:
  - linear ⇔ polynomial (P=1)
  - quadratic ⇔ polynomial (P=2)
  - sqrt ⇔ polynomial (P=0.5)
  - including inverse-warmup variants.
- Expanded test docstrings to reflect full documented formulas.

### Removed
- Removed redundant boundary-only tests now fully covered by per-schedule contract tests:
  - `test_warmup_boundary_values`
  - `test_inverse_warmup_boundary_values`

### audit

- Human audited, a bit at a time. Module now accepted.

## 0.6.2

Fixed load_state_dict bug with proxy desync

**CRITICAL BUG FIX**: ProxyDictByLR now survives optimizer.load_state_dict()
- Changed ProxyDictByLR to accept get_dictionary callback instead of direct dict reference
- Made .dictionary on ProxyDictByLR a property that invokes callback (always returns current dict)
- ArbitraryScheduleAdapter now passes closures: `lambda i=i: self.optimizer.param_groups[i]`
- Removed desync detection error raising (load_state_dict causes expected "desyncs")
- Replaced with auto-resync: proxy automatically updates cache when backend changes
- Removed set_throw_error_on_desync from public API (no longer needed)
- Removed desync tests (test_desync_detection_raises_error, test_set_throw_error_on_desync_controls_error_behavior)
- **RESULT**: test_load_state_from_earlier_step now passes - checkpoint resume works correctly
- HUMAN_AUDIT: Sync is approved. 

## 0.6.1

Infrastructure, excluding syncronous schedule

- Added test verifying using factory on an invalid name will throw in an informative manner
- Accepted non-syncronous schedule infrastructure as final.

## 0.6.0

Beginning of final audit and quality control.
Human, LLM corrections.

Adapter and core infrastructure audit (FINAL)

- Renamed throw_errors_on_desync → set_throw_error_on_desync (consistency with naming conventions)
- Exported set_throw_error_on_desync as part of public API
- Updated error messages to reference correct function name
- Added test_set_throw_error_on_desync_controls_error_behavior (verifies error/warning control)
- Relaxed error message matching in adapter tests (allows implementation flexibility)
- Fixed inline import not isolated in monad. Was used to prevent circular reference.
- Test suite: 8 adapter tests covering contracts, namespace routing, state independence, desync detection
- **FINAL ACCEPTANCE**: Adapter layer (ArbitraryScheduleAdapter, ProxyDictByLR, namespace routing,
  desync detection) has been audited and accepted as production-ready

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
