# Contributing

## Scope
This repository contains educational machine learning exercises. Contributions should preserve existing learning content and focus on correctness, clarity, and maintainability.

## Local setup
1. Create a virtual environment.
2. Install development dependencies:
   ```bash
   make install-dev
   ```

## Development workflow
1. Create a focused branch for your change.
2. Keep changes minimal and avoid unrelated refactors.
3. Run quality checks before opening a pull request:
   ```bash
   make format
   make lint
   make typecheck
   make test-cov
   make security
   ```

## Pull requests
- Describe the problem and the implemented fix.
- Include test evidence.
- Keep pull requests reviewable by limiting scope.

## Coding standards
- Follow Ruff formatting and linting rules.
- Add or update tests when behavior changes.
- Avoid introducing network dependent tests.
