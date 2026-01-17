# Usage:
#   make dashboard
#   make dashboard SOURCE_ROOT="/path/to/Basketball_prediction/2026"

SOURCE_ROOT ?= /Users/alexanderrazmyslov/1. Python/Basketball_prediction/2026

dashboard:
	python3 scripts/generate_dashboard_data.py --source-root "$(SOURCE_ROOT)"
	python3 -c 'import json; from pathlib import Path; base=Path("public/data"); lr=json.loads((base/"last_run.json").read_text(encoding="utf-8")); sm=json.loads((base/"summary.json").read_text(encoding="utf-8")); print("last_run.as_of_date:", lr.get("as_of_date")); print("last_run.last_run:", lr.get("last_run")); print("summary.as_of_date:", sm.get("as_of_date")); import sys; sys.exit(0 if lr.get("as_of_date")==sm.get("as_of_date") else 1)'
