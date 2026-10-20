# SAS historical profitability — legacy watch setup

Observed at: 2026-10-20T19:45:16Z UTC

## Result

Using the project watch/reference filter on settled historical rows with SAS as the home team:

- Sample: 17 games
- Record: 14–3
- Flat one-unit profit: +8.49 units
- ROI: +49.94%
- Average odds: 1.803
- Maximum drawdown: 1 unit

For all settled SAS home games in the same historical artifact:

- Sample: 49 games
- Record: 37–12
- Flat one-unit profit: +5.22 units
- ROI: +10.65%

## Filter definition

The reference filter was:

- home_win_rate >= 0.50
- home odds between 1.30 and 3.40
- model probability used >= 0.55
- positive model EV

This is a small, selected, legacy sample. It is not an out-of-sample estimate and does not validate the current-season roster.

## Current OKC–SAS price check

For the supplied SAS quote 1.769 and the legacy SAS probability 56.4297%:

- Implied gross return EV: approximately -0.18%
- Result under the project gate: `NO_BET`

At a different stale local reference price of 1.80, the arithmetic would be approximately +1.57%, but that price was not verified as current and is not a canonical signal.

## Caveats

The current roster, availability and minutes information are incomplete; the roster challenger remains blocked. No model or betting rule was changed.

## Local evidence references

- Historical artifact: `data/processed/combined_predictions_2026.parquet`
- Baseline configuration: `configs/baseline.json`
- Current baseline output: `outputs/script3_model_predictions/20261020T120021000000Z/baseline_2026_predictions.csv`
