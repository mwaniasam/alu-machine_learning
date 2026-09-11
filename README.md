# ALU Machine Learning

## Overview
This repository contains educational machine learning exercises organized by topic. The content focuses on foundational mathematics, supervised learning, unsupervised learning, reinforcement learning, and supporting data workflows.

## Goals
- Build practical understanding of machine learning fundamentals.
- Provide concise reference implementations for learning.
- Maintain a reliable baseline for local development and CI checks.

## Repository structure
- `math/`: calculus, linear algebra, and advanced linear algebra exercises.
- `pipeline/`: API, database, and pandas workflow exercises.
- `supervised_learning/`: classification, RNNs, transformers, and related topics.
- `unsupervised_learning/`: clustering, dimensionality reduction, autoencoders, and HMM exercises.
- `reinforcement_learning/`: Q-learning and temporal difference methods.
- `tests/`: deterministic smoke tests for repository validation.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
```

## Quickstart
Run quality checks:
```bash
make format
make lint
make typecheck
make test-cov
make security
```

## Running notebooks and scripts safely
- Run scripts and notebooks in an isolated virtual environment.
- Avoid committing generated artifacts and credentials.
- Keep external downloads optional during local experimentation.

## Testing and quality
- Ruff is used for formatting and linting.
- Mypy is used for pragmatic type checking.
- Pytest runs deterministic smoke tests.
- Coverage is reported in CI and locally through `make test-cov`.

## CI and automation
- `ci.yml` runs linting, type checks, and tests on Python 3.10 and 3.11.
- `security.yml` runs Bandit and pip-audit.
- Dependabot updates Python and GitHub Actions dependencies.

## Contributing
See `/CONTRIBUTING.md` for contribution workflow and quality expectations.

## Security and support
- Security policy: `/SECURITY.md`
- Support guidelines: `/SUPPORT.md`

## License
This project is licensed under the MIT License. See `/LICENSE`.
