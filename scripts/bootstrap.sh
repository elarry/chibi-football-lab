#!/usr/bin/env bash

set -e

uv python install 3.10.12
uv python pin 3.10.12
uv venv --python 3.10.12 --seed
source .venv/bin/activate

git clone --depth 1 https://github.com/Unity-Technologies/ml-agents.git external/ml-agents
python -m pip install -e external/ml-agents/ml-agents-envs
python -m pip install -e external/ml-agents/ml-agents
