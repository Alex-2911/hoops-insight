#!/usr/bin/env bash
set -euo pipefail

python -m py_compile scripts/generate_dashboard_data.py scripts/config_loader.py scripts/strategy_logic.py
node scripts/validate_dashboard_state.mjs public/data
npm run lint
