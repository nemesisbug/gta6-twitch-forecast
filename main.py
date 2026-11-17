"""Offline, first-hour-conditioned launch curves; Python standard library only."""
import json
import math
import os
from datetime import datetime
from pathlib import Path

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
BANDWIDTH = 1.25


def release_curves(case):
    """Keep complete positive release days, excluding the target game's history."""
    curves = []
    target = case['game']['name'].casefold()
    for launch in case['comparables']:
        if target in launch['name'].casefold():
            continue
        start = datetime.fromisoformat(launch['day_start'])
        hours = {}
        for row in launch['hourly']:
            offset = (datetime.fromisoformat(row['from']) - start).total_seconds()
            if 0 <= offset < 86400 and offset % 3600 == 0:
                hours[int(offset // 3600)] = row['viewers']
        if len(hours) == 24 and all(math.isfinite(v) and v > 0 for v in hours.values()):
            curves.append([hours[h] for h in range(24)])
    return curves


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def fit(curves, p1):
    """Return per-hour conditional log moves and residual population deviations."""
    if len(curves) < 5:
        return [(0.0, 0.15 * math.sqrt(h - 1)) for h in range(2, 25)]
    x0 = math.log(p1)
    xs = [math.log(c[0]) for c in curves]
    # A common rescaling preserves the fit and prevents kernel underflow.
    log_weights = [-0.5 * ((x - x0) / BANDWIDTH) ** 2 for x in xs]
    largest = max(log_weights)
    weights = [math.exp(w - largest) for w in log_weights]
    total = sum(weights)
    xbar = sum(w * x for w, x in zip(weights, xs)) / total
    variance = sum(w * (x - xbar) ** 2 for w, x in zip(weights, xs))
    result = []
    for h in range(1, 24):
        ratios = [math.log(c[h] / c[0]) for c in curves]
        rbar = sum(w * r for w, r in zip(weights, ratios)) / total
        slope = (sum(w * (x - xbar) * (r - rbar)
                     for w, x, r in zip(weights, xs, ratios)) / variance
                 if variance > 1e-12 else 0.0)
        prediction = rbar + slope * (x0 - xbar)
        prediction = max(percentile(ratios, 0.02), min(percentile(ratios, 0.98), prediction))
        residual = sum(w * (r - rbar - slope * (x - xbar)) ** 2
                       for w, x, r in zip(weights, xs, ratios)) / total
        result.append((prediction, max(0.25, min(2.5, math.sqrt(residual)))))
    return result


def quantiles(p1, mean, spread):
    values = []
    for z in Z:
        # Bound in log space before exponentiation, including extreme inputs.
        log_value = max(math.log(0.01), min(math.log(1e9),
                        math.log(p1) + mean + z * spread))
        value = math.exp(log_value)
        rounded = math.floor(value * 100 + 0.5) / 100
        values.append(int(rounded) if rounded.is_integer() else rounded)
    return values


def forecast(case):
    p1 = max(1, next(row['viewers'] for row in case['known'] if row['hour'] == 1))
    parameters = fit(release_curves(case), p1)
    return {'hours': [{'hour': h, 'q': quantiles(p1, *parameters[h - 2])}
                      for h in range(2, 25)]}


def main():
    source = Path(os.environ.get('MAKEFUN_INPUT_DIR', '/input'))
    destination = Path(os.environ.get('MAKEFUN_OUTPUT_DIR', '/output'))
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob('*.in')):
        case = json.loads(path.read_text(encoding='utf-8'))
        result = forecast(case)
        (destination / (str(case['id']) + '.out')).write_text(
            json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
