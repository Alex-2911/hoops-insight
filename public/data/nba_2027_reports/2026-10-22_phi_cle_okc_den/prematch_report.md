# NBA 2027 Pre-Game Report: PHI–CLE and OKC–DEN

Report package created from the guarded 2027 pipeline on 2026-10-22.

## Safety status

`RESEARCH_ONLY / BASELINE_PROXY / ROSTER_CHALLENGER_BLOCKED / NO_BET`

The published pipeline artifacts contain no credentials or private configuration. No canonical pipeline bet was generated or executed.

## PHI–CLE

- Scheduled game date: `2026-10-22`
- DraftKings odds: Philadelphia `1.704225` (`-142`), Cleveland `2.20` (`+120`)
- Baseline model: `baseline_2026_lightgbm`
- Philadelphia baseline probability: `40.3355%`
- Diagnostic strategy probability used: `54.1857%`
- Diagnostic EV: `-7.6554` per 100 stake
- Canonical decision: `NO_BET`

## OKC–DEN

- Scheduled game date: `2026-10-22`
- DraftKings odds: Oklahoma City `1.294118` (`-340`), Denver `3.70` (`+270`)
- Baseline model: `baseline_2026_lightgbm`
- Oklahoma City baseline probability: `55.6590%`
- Diagnostic strategy probability used: `68.9092%`
- Diagnostic EV: `-10.8234` per 100 stake
- Canonical decision: `NO_BET`

## Availability

The official NBA injury report dated October 22, 2026 at 2:45 p.m. ET was retrieved before tipoff. Relevant listed statuses were:

- Philadelphia: Dominick Barlow `OUT`, Tacko Fall `OUT`, Ariel Hukporti `OUT`, Dean Wade `QUESTIONABLE`.
- Cleveland: Jarrett Allen `QUESTIONABLE`, Khalifa Diop `OUT`, Nae'Qwan Tomlin `OUT`.
- Oklahoma City: Otega Oweh `OUT`, Thomas Sorber `OUT`, Nikola Topic `OUT`.
- Denver: Spencer Jones `QUESTIONABLE`, Julian Strawther `OUT`.

The report does not provide an explicit status for LeBron James or Jaylen Brown. Their absence from the listed rows is not treated as confirmation that they are available. Expected starters and minutes projections remain unavailable; no roster adjustment was applied to the baseline.

Source: <https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-22_02_45PM.pdf>

Local evidence snapshot SHA-256: `713e42b9f6987eddcd1f38db630812ea690cbc314e42329b3f4872d16a6761be`

## Interpretation

Philadelphia has a meaningful market-implied chance, but the model remains a 2026-statistics proxy and is not roster-aware. Current odds and availability were collected as evidence only. The strategy gate remained disabled for canonical execution and produced zero stake.
