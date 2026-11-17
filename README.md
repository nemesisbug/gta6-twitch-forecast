[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01&ref=w642jqgx)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nemesisbug/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nemesisbug/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01&ref=w642jqgx)

# GTA VI Twitch launch forecast

An offline Python 3.13 standard-library program forecasts hours 2–24 of launch-day Twitch concurrent viewers at five quantile levels. It reads each local `.in` JSON case and writes its `<id>.out` forecast.

The model extracts complete positive release-day curves from the case's comparables, excluding the target game's own history. For each hour it fits a ridge regression of the log viewer ratio to hour 1 on three features — log first-hour viewers, log pre-release median (time-weighted median over days −14..−2), and log channels/viewers — using a Gaussian kernel with a 1.5× weight for console launches. The kernel bandwidth and ridge penalty adapt to the launch's scale, because small launches are noisy and need pooling while big launches are information-rich and need locality: hour-1 < 50k uses bandwidth 1.6 and ridge 4.0, 50k–130k uses 0.8 and 0.5, and ≥ 130k uses 0.6 and 0.25. Predictions are clipped to the training ratios' interpolated 2nd–98th percentiles. The hour-1 anchor is shrunk 5% toward the pre-release-implied scale when the print is a single hourly read (reads < 30); a precise multi-read print is trusted as-is. Weighted residual standard deviations, bounded to 0.25–2.5 and scaled by 0.85, supply the quantile spread. With fewer than five usable launches, it uses persistence with spread `0.15 * sqrt(hour - 1)`. Instead of fixed normal scores, the five quantiles use the empirical quantiles of standardized inner leave-one-out residuals, pooled over all comparable launches and hours and shrunk 50% toward the normal scores; the upper tail is narrowed by 0.85. Quantiles use half-up rounding to two decimals.

Leave-one-launch-out over the 37 comparable launches: BIG launches (hour-1 ≥ 50k) 0.162 vs 0.181 for v2 (0.204 for v1); proxy practice loss ~0.150 vs ~0.168 for v2 (0.190 for v1); paired bootstrap 95% CI for the improvement over v2 [−0.034, +0.005], with 18 of 20 BIG launches improving. The top-20 practice bar is ~0.137, so v3 is better than v2 but still short of it.

Run locally:

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/tmp/forecast-output python3 main.py
```

Defaults are `/input` and `/output`. Input files are processed in sorted order. The program uses no network and learns only from each case's own comparables.
