#!/usr/bin/env bash
set -euo pipefail
python check_prompt_config.py
python -m unittest discover -s tests -p 'test_*.py' -v
