# Changelog

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
