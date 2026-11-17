# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/calebakinjomo/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/calebakinjomo/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category for hours 2–24 of its launch day (2026-11-19 UTC), as five quantiles per hour, from the hour-1 print plus the launch-day shapes of past big launches.

## Model

For each comparable launch, take its release-day hourly curve and form log-ratios `r_h = ln(viewers_h / viewers_1)`. A launch only counts when it is a real rolling-midnight console launch: tier "comparable" (no early-access contamination, no free-to-play, no slow-burn weak launches), no PC in its platforms, and an hour-1 print already at least 10% of its own release-day peak (so hour 1 is a post-unlock hour — simultaneous global unlocks fail this and would otherwise blow the ratios up). Then, per hour, fit a weighted least-squares line `r_h ~ a_h + b_h · ln(hour-1 print)` — weights `1 / (1 + 2|ln h1_comp − ln p|)` so launches near the target's scale count most — with each hour's slope shrunk toward the pooled global slope. The median forecast is the fitted line at the target's log hour-1 print; the spread is the weighted residual standard deviation × 1.6, and the five quantiles are median + z_τ · spread.

Why the regression: across launches the evening rise is strongly scale-dependent — small hour-1 prints rise much more by US prime time than big ones do (fitted late-day slopes around −0.4). A flat median misses that; the regression captures it.

On the four practice cases (ground truth fetched from SullyGnome's hourly averages) this scores a mean pinball loss of **0.194** against the house baseline's **0.387**, with a typical miss of ~36% and ~0.87 ninety-percent coverage. Per case: Wolverine 0.369, Red Dead Redemption 2 0.135, God of War Ragnarök 0.104, Spider-Man 2 0.169.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---
<sub>Entered on [make.fun](https://make.fun/?ref=hdavhdvz&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
