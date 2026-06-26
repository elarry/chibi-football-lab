# Chibi Football Lab

## Environment

Python 3.10.12 (pinned). Do not suggest upgrading — ml-agents breaks on 3.11+.

Dependencies are installed via `scripts/bootstrap.sh`, not `pyproject.toml`.
The venv is at `.venv/`; activate with `source .venv/bin/activate`.
`ml-agents` and `ml-agents-envs` are editable installs from `external/ml-agents/`.

## Structure

- `config/` — ML-Agents YAML training configs
- `models/` — trained model artifacts
- `notebooks/` — usage walkthrough notebooks
- `scripts/` — shell scripts for setup
