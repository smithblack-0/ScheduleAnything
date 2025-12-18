# Development Notes

## Development Workflow

This project follows document-driven development:
1. Documentation - Define the API and user experience first
2. DevOps - Set up CI/CD, packaging, and infrastructure
3. API Tests - Write tests based on documented API
4. Core Code - Implement the functionality
5. Internal Tests - Test implementation details

## Before Each Release

* Version bump in pyproject.toml
* CHANGELOG.md entry for the version
* All tests passing in CI
* Code formatted with black and ruff
* Documentation updated if API changed

## API Stability Policy

Once released, API changes will never invalidate existing code except if underlying PyTorch libraries change.

## CI/CD Pipeline

* **CI (ci.yml)**: Runs on every push/PR - tests across Python 3.9, 3.10, 3.11
* **Validate Release (validate-release.yml)**: Checks version format, CHANGELOG, and version increment
* **CD (cd.yml)**: Publishes to PyPI when PR with 'Release' label is merged

## Release Process

1. Update version in `pyproject.toml`
2. Add entry to `CHANGELOG.md`
3. Create PR with changes
4. Add "Release" label to PR
5. Merge after CI passes and validation succeeds
6. CD automatically publishes to PyPI

## Local Development

```bash
# Install in editable mode with dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Format code
python format_code.py

# Or manually
black src/ tests/
ruff check --fix src/ tests/
```
