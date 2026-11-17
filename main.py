"""Offline, first-hour-conditioned launch curves; Python standard library only.

v3 changes vs v2 (validated by leave-one-launch-out over 37 comparable
launches; practice loss is server-only, so LOO is the decision metric):

- Scale-adaptive bandwidth + ridge (the big win). Small launches are noisy
  and need pooling; big launches are information-rich and need locality:
  p1 < 50k   -> bw 1.6, ridge 4.0
  50k-130k   -> bw 0.8, ridge 0.5
  p1 >= 130k -> bw 0.6, ridge 0.25
  LOO BIG (hour-1 >= 50k) 0.1805 -> 0.1624; proxy practice 0.168 -> 0.150.
- Pooled standardized-residual quantile shape (cross-conformal style).
  The lognormal z is replaced by empirical quantiles of standardized
  inner-LOO residuals, pooled over all train launches x 23 hours and
  shrunk 50% toward the normal z. This also corrects the median bias
  (residual median != 0). The shape is recomputed inside every outer
  LOO fold (nested jackknife) and, at runtime, inside every case from
  that case's own comparables -- never reused across cases.
- Upper tail narrowed: tail_hi 0.85 (lower tail unchanged, 1.0).

LOO (37 launches): ALL 0.382, BIG 0.162 (v2: ALL 0.402, BIG 0.181);
proxy practice ~0.150 (v2: ~0.168); top-20 practice bar is ~0.137, so
v3 is better than v2 but still short of the top-20 bar -- per the
standing top-score gate it is staged, NOT submitted.
Paired bootstrap on BIG vs v2: mean delta -0.018, 95% CI [-0.034, +0.005],
18/20 BIG launches improve.

Rejected in the v3 round (all tested on LOO BIG, worse or neutral):
mixture quantiles (alone and as 0.2 admixture), LAD/quantile-regression
center, k-NN center, median+IQR center, pre-release trend/spike features,
recency decay, quality down-weight, Hogwarts/Cyberpunk exclusion,
spread floor 0.15, bw 2.303, guarded stretch gamma, hour-smoothing of
center (James-Stein shrinkage), peak-hour feature, pooled shared-slope
model, vitality shrinkage (redundant with adaptivity), excluding
simultaneous-global or early-access launches from the pool, Student-t
quantiles, heteroskedastic spread growing with h.
"""
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
SPREAD_MULT = 0.85
TAIL_HI = 0.85
TAIL_LO = 1.0
LAMBDA_NOISY = 0.95
READS_PRECISE = 30
SIM_WEIGHT = 1.5
FEATURES = ('logp1', 'logP', 'logch')
PRE_RELEASE_DAYS = 14
PRE_RELEASE_END_DAYS = 2
MIN_PRE_RELEASE_POINTS = 24
SHAPE_SHRINK = 0.5


def params_for_p1(p1):
    """Scale-adaptive (bandwidth, ridge): small launches need pooling,
    big launches need locality."""
    if p1 < 50000:
        return 1.6, 4.0
    if p1 < 130000:
        return 0.8, 0.5
    return 0.6, 0.25


def pre_release_median(rows, day_start):
    """Time-weighted median of viewers over days -14..-2 before day_start.

    Each point represents the gap to the next point, capped at 12h; the last
    point counts 1h. Returns (median, point_count), median None if empty.
    """
    low = day_start - timedelta(days=PRE_RELEASE_DAYS)
    high = day_start - timedelta(days=PRE_RELEASE_END_DAYS)
    points = []
    for row in rows:
        if row.get('viewers') is None or row['viewers'] <= 0:
            continue
        when = datetime.fromisoformat(row['from'])
        if low <= when < high:
            points.append((when, row['viewers']))
    if not points:
        return None, 0
    points.sort()
    items = []
    for j, (when, value) in enumerate(points):
        if j + 1 < len(points):
            duration = min((points[j + 1][0] - when).total_seconds() / 3600.0, 12.0)
        else:
            duration = 1.0
        items.append((value, max(duration, 0.25)))
    items.sort()
    total = sum(d for _, d in items)
    accrued = 0.0
    for value, duration in items:
        accrued += duration
        if accrued >= total / 2:
            return value, len(points)
    return items[-1][0], len(points)


def release_curves(case):
    """Complete positive release days, excluding the target game's history.

    Returns (pool, pre_release_by_name): pool entries carry the 24 hourly
    viewers, channels at hour 1, platforms string and pre-release median.
    """
    target = case['game']['name'].casefold()
    pool = []
    pre_release_by_name = {}
    for launch in case['comparables']:
        start = datetime.fromisoformat(launch['day_start'])
        median, _ = pre_release_median(launch['hourly'], start)
        pre_release_by_name[launch['name'].casefold()] = median
        if target in launch['name'].casefold():
            continue
        hours = {}
        channels = {}
        for row in launch['hourly']:
            offset = (datetime.fromisoformat(row['from']) - start).total_seconds()
            if 0 <= offset < 86400 and offset % 3600 == 0:
                hours[int(offset // 3600)] = row['viewers']
                channels[int(offset // 3600)] = row.get('channels')
        if len(hours) == 24 and all(math.isfinite(v) and v > 0 for v in hours.values()):
            pool.append({
                'name': launch['name'],
                'platforms': launch.get('platforms') or '',
                'v': [hours[h] for h in range(24)],
                'ch1': channels.get(0),
                'P': median,
            })
    return pool, pre_release_by_name


def target_pre_release(case, pre_release_by_name):
    """Pre-release median for the target game.

    From the case's own history when it covers the -14..-2 day window;
    otherwise from the matching comparable curve (e.g. the target category's
    pre-release history curve, whose own history is longer).
    """
    start = datetime.fromisoformat(case['day_start'])
    median, count = pre_release_median(case.get('history_hourly') or [], start)
    if median is not None and count >= MIN_PRE_RELEASE_POINTS:
        return median
    target = case['game']['name'].casefold()
    for name, value in pre_release_by_name.items():
        if value is None:
            continue
        if target in name or name in target:
            return value
    return median


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def weighted_median(pairs):
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    accrued = 0.0
    for value, weight in ordered:
        accrued += weight
        if accrued >= total / 2:
            return value
    return ordered[-1][0]


def feat_vecs(curve, feats):
    values = []
    scale = max(curve['P'] if curve.get('P') else curve['v'][0], 10.0)
    if 'logp1' in feats:
        values.append(math.log(curve['v'][0]))
    if 'logP' in feats:
        values.append(math.log(scale))
    if 'logch' in feats:
        channels = curve['ch1'] if curve.get('ch1') else 1
        values.append(math.log(max(channels, 1) / max(curve['v'][0], 1)))
    return values


def ridge_fit(features, targets, weights, ridge):
    """Weighted ridge (intercept unpenalized) on pre-standardized features."""
    k = len(features[0])
    matrix = [[0.0] * (k + 1) for _ in range(k + 1)]
    rhs = [0.0] * (k + 1)
    for row, target, weight in zip(features, targets, weights):
        extended = [1.0] + list(row)
        for a in range(k + 1):
            rhs[a] += weight * extended[a] * target
            for b in range(k + 1):
                matrix[a][b] += weight * extended[a] * extended[b]
    for a in range(1, k + 1):
        matrix[a][a] += ridge
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(k + 1):
        pivot = max(range(col, k + 1), key=lambda r: abs(augmented[r][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        if abs(augmented[col][col]) < 1e-12:
            continue
        for r in range(k + 1):
            if r != col and augmented[r][col] != 0:
                factor = augmented[r][col] / augmented[col][col]
                for cc in range(col, k + 2):
                    augmented[r][cc] -= factor * augmented[col][cc]
    beta = [augmented[i][k + 1] / augmented[i][i] if abs(augmented[i][i]) > 1e-12 else 0.0
            for i in range(k + 1)]
    return beta[0], beta[1:]


def fit(pool, p1, p_target, ch1_target, reads):
    """Return (log anchor, per-hour (center, spread)) for hours 2..24."""
    if len(pool) < 5:
        return math.log(p1), [(0.0, 0.15 * math.sqrt(h - 1)) for h in range(2, 25)]
    bandwidth, ridge = params_for_p1(p1)
    logp1 = math.log(p1)
    xs = [math.log(c['v'][0]) for c in pool]
    log_weights = [-0.5 * ((x - logp1) / bandwidth) ** 2 for x in xs]
    largest = max(log_weights)
    weights = []
    for curve, log_weight in zip(pool, log_weights):
        weight = math.exp(log_weight - largest)
        if 'PC' not in curve['platforms']:
            weight *= SIM_WEIGHT
        weights.append(weight)
    total = sum(weights)
    weights = [w / total for w in weights]
    # Anchor: shrink a noisy single-read hour-1 print toward the
    # pre-release-implied scale; trust a precise multi-read print as-is.
    if reads < READS_PRECISE:
        target_scale = max(p_target if p_target else p1, 10.0)
        terms = []
        for curve, weight in zip(pool, weights):
            curve_scale = max(curve['P'] if curve.get('P') else curve['v'][0], 10.0)
            terms.append((math.log(curve['v'][0] / curve_scale), weight))
        log_b = math.log(target_scale) + weighted_median(terms)
        log_anchor = LAMBDA_NOISY * logp1 + (1 - LAMBDA_NOISY) * log_b
    else:
        log_anchor = logp1
    result = []
    for h in range(1, 24):
        ratios = [math.log(c['v'][h] / c['v'][0]) for c in pool]
        features = [feat_vecs(c, FEATURES) for c in pool]
        k = len(FEATURES)
        means = []
        scales = []
        for j in range(k):
            mean = sum(w * x[j] for w, x in zip(weights, features))
            means.append(mean)
            spread = math.sqrt(sum(w * (x[j] - mean) ** 2 for w, x in zip(weights, features)))
            scales.append(spread if spread > 1e-9 else 1.0)
        standardized = [[(x[j] - means[j]) / scales[j] for j in range(k)] for x in features]
        target_scale = max(p_target if p_target else p1, 10.0)
        target = {'v': [p1], 'P': target_scale, 'ch1': ch1_target}
        target_vec = [(feat_vecs(target, FEATURES)[j] - means[j]) / scales[j] for j in range(k)]
        intercept, coefs = ridge_fit(standardized, ratios, weights, ridge)
        center = intercept + sum(b * x for b, x in zip(coefs, target_vec))
        center = max(percentile(ratios, 0.02), min(percentile(ratios, 0.98), center))
        residual = sum(w * (r - (intercept + sum(b * x for b, x in zip(coefs, xi)))) ** 2
                       for xi, r, w in zip(standardized, ratios, weights))
        spread = max(0.25, min(2.5, math.sqrt(residual))) * SPREAD_MULT
        result.append((center, spread))
    return log_anchor, result


def residual_shape(pool):
    """Empirical quantiles of standardized inner-LOO residuals, pooled over
    all pool launches x 23 hours, shrunk 50% toward the normal z.

    Each pool launch is forecast from the pool without it (its own hour-1
    is a single SullyGnome read, so the noisy-read anchor applies); the
    standardized residual (log miss / spread) is collected per hour.
    Returns 5 shape values aligned with TAUS, or Z if the pool is tiny.
    """
    if len(pool) < 8:
        return list(Z)
    residuals = []
    for j, launch in enumerate(pool):
        inner = pool[:j] + pool[j + 1:]
        log_a, params = fit(inner, launch['v'][0], launch['P'], launch['ch1'], 1)
        for h_idx, h in enumerate(range(2, 25)):
            y = math.log(max(launch['v'][h - 1], 1))
            center, spread = params[h_idx]
            residuals.append((y - (log_a + center)) / spread)
    shape = []
    for tau, z in zip(TAUS, Z):
        shape.append(SHAPE_SHRINK * percentile(residuals, tau)
                     + (1 - SHAPE_SHRINK) * z)
    return shape


def quantiles(log_anchor, center, spread, shape):
    values = []
    med = shape[2]
    for qf in shape:
        gap = (qf - med) * spread
        if gap > 0:
            gap *= TAIL_HI
        elif gap < 0:
            gap *= TAIL_LO
        log_value = max(math.log(0.01), min(math.log(1e9),
                        log_anchor + center + med * spread + gap))
        value = math.exp(log_value)
        rounded = math.floor(value * 100 + 0.5) / 100
        values.append(int(rounded) if rounded.is_integer() else rounded)
    return values


def forecast(case):
    known = sorted(case['known'], key=lambda row: row['hour'])
    hour_one = next((row for row in known if row['hour'] == 1), known[0])
    p1 = max(1, hour_one['viewers'])
    reads = hour_one.get('reads') or 1
    ch1 = hour_one.get('channels')
    pool, pre_release_by_name = release_curves(case)
    p_target = target_pre_release(case, pre_release_by_name)
    log_anchor, parameters = fit(pool, p1, p_target, ch1, reads)
    shape = residual_shape(pool)
    return {'hours': [{'hour': h, 'q': quantiles(log_anchor, *parameters[h - 2], shape)}
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
