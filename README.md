# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/samoyz/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/samoyz/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each PS5 trophy in week one, as five quantiles per trophy, scored by pinball loss on the logit scale.

## Method

Per-trophy logit trend in log-time. For each trophy, fit a recency-weighted least-squares line of logit(rate) against ln(1 + hours since launch) over the case history (24-hour half-life weights, so recent reads dominate). Shrink the slope toward zero (persistence) so a noisy trophy does not chase its own noise; a tier mean is never used as a prior, since a tier mixes rising early trophies with falling late ones. Read the shrunk line at the target for the median, blended slightly toward the last rate. The spread combines the house random-walk term (0.5 logit units per square-root day), slope uncertainty times extrapolation distance, and fit residual, clamped to 0.4–1.8 logit units; trophies at 0% get a wider spread because they may emerge. Quantiles are the median shifted by the standard normal quantiles times the spread, mapped back to percent.

Why this beats persistence: Exophase rates move on two forces, progression (tracked players reach the trophy, rate up) and dilution (new players join having earned nothing, rate down). Early trophies rise, late trophies fall. A line in ln(1 + hours) captures the net drift and damps the early surge, while holding the last rate misses both moves.

## Run

Node 22, standard library only. Input: every `*.in` case in the input directory (`MAKEFUN_INPUT_DIR` or `/input`). Output: one `*.out` per case in the output directory (`MAKEFUN_OUTPUT_DIR` or `/output`): `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}`. Deterministic: no randomness, reads processed in name order, so two runs write identical bytes.
