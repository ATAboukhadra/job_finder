#!/usr/bin/env bash
# AI Apply — profile generation + remote-focused scrape
# Usage: bash run.sh
set -euo pipefail

cd "$(dirname "$0")"

PY=.venv/bin/python
# Local LLM model for profile extraction (Ollama). qwen3.5:9b is a "thinking"
# model; llm.py sends think=false so the answer lands in the response field.
MODEL=qwen3.5:9b

# 1. Generate profile.yaml from life-story.md (remote-first by default:
#    search.remote=true and remote_preferred=true).
$PY main.py init-profile --model "$MODEL"

# 2. Scrape, focusing mainly on remote jobs.
#    remotive / himalayas / arbeitnow are remote-only or remote-first boards;
#    the general boards apply remote filters because search.remote=true.
$PY main.py scrape \
    --boards remotive himalayas arbeitnow indeed linkedin google themuse internet \
    --max 40

# 3. (optional) View top matches:
# $PY main.py top --limit 25
# 4. (optional) Launch the web dashboard:
# $PY main.py ui --port 5000
