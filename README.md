# GTA VI Twitch launch forecast

An offline Python 3.13 standard-library program forecasts hours 2–24 of launch-day Twitch concurrent viewers at five quantile levels. It reads each local `.in` JSON case and writes its `<id>.out` forecast.

The model extracts complete positive release-day curves from the case's comparables, excluding the target game's own history. For each hour it regresses the log viewer ratio to hour 1 on the log first-hour audience, using a Gaussian kernel with bandwidth 1.25. Predictions are clipped to the training ratios' interpolated 2nd–98th percentiles; weighted residual standard deviations, bounded to 0.25–2.5, supply the quantile spread. With fewer than five usable launches, it uses persistence with spread `0.15 * sqrt(hour - 1)`. Quantiles use fixed normal scores and half-up rounding to two decimals.

Run locally:

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/tmp/forecast-output python3 main.py
```

Defaults are `/input` and `/output`. Input files are processed in sorted order. The program uses no network and learns only from each case's own comparables.
