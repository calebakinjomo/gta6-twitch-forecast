# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/calebakinjomo/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/calebakinjomo/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for hours 2–24 of its launch day (2026-11-19 UTC), as five quantiles per hour, from the hour-1 print plus the launch-day shapes of past big launches.

## Model

For each comparable launch, take its release-day hourly curve and form log-ratios `r_h = ln(viewers_h / viewers_1)`. A launch only counts when it is a real rolling-midnight console launch: tier "comparable" (no early-access contamination, no free-to-play, no slow-burn weak launches), no PC in its platforms, and an hour-1 print already at least 10% of its own release-day peak (so hour 1 is a post-unlock hour — simultaneous global unlocks fail this and would otherwise blow the ratios up). Then fit one pooled model over every (launch, hour) point: `r_h = s_h + (b0 + b1·u_h + b2·u_h²) · ln(hour-1 print) + c0 · (ln viewers-per-channel − mean)`, where `u_h = max(0, h−5)/19` ramps from the US-East unlock and viewers-per-channel is hour-1 viewers / hour-1 channels. The 23 hour intercepts carry the shared unlock-wave shape; the slope basis lets the scale effect grow through the day instead of estimating 23 separate noisy per-hour slopes; the viewers-per-channel term is a level shift (broad grassroots audiences run hotter all day than concentrated big-streamer ones). The median curve is fitted by direct pinball loss (iteratively reweighted least squares from the least-squares fit, deterministic): least squares is pulled by the extreme launches, while the pinball median tracks the typical launch — and the scored metric is the pinball loss itself. The spread is the pooled residual SD × 1.3, and the five quantiles are median + z_τ · spread.

Why the pooled fit: pooling the slope across hours and the spread across hours removes most of the estimation noise of fitting each hour separately, and the viewers-per-channel term captures a real level difference the scale regression misses. The model was selected by leave-one-launch-out cross-validation over the console comparables and the four practice cases — a variant had to win on both.

On the four practice cases (ground truth fetched from SullyGnome's hourly averages) this scores a mean pinball loss of **0.177** against the house baseline's **0.387** (v2 scored 0.179, v1 0.194), with a typical miss of ~32% and ~0.80 ninety-percent coverage. Per case: Wolverine 0.254, Red Dead Redemption 2 0.091, God of War Ragnarök 0.136, Spider-Man 2 0.228. Leave-one-launch-out over the 7 console comparables: 0.143 (v2: 0.155); over the 11 console launches (the 7 plus the 4 practice launches, as the final case has them): 0.150 (v2: 0.156).

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---
<sub>Entered on [make.fun](https://make.fun/?ref=hdavhdvz&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
