#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: Monte-Carlo top-5 market-cap forecaster.

Reads /input (or $MAKEFUN_INPUT_DIR): task.json + history.ndjson of 10-minute
marks ranking new pump.fun coins by market cap. Writes forecast.json (or
$MAKEFUN_OUTPUT_DIR/task.output) with per-place coin probabilities, an ordered
top 5, and per-place market-cap size-bin probabilities, for the task's answer
reads (which may be many hours after the history ends).

Model:
  * Per-coin log10-cap dynamics are fitted from the history itself: the drift
    and robust volatility of log-cap increments, bucketed by coin age and by
    whether its bonding curve has completed, plus a shrunk per-coin momentum.
  * Each candidate coin is walked forward in 0.5 h steps with a fat-tailed
    Gaussian random walk (drift from its age/complete bucket) to every answer
    read; rankings across simulations give place probabilities and the
    distribution of each place's cap (for the size bins).
  * "Phantom newcomers": coins born after the cut cannot be named, only
    reached through the probability a forecast leaves unnamed at a place.
    The simulation races phantom coins whose birth rate and peak-cap
    distribution are fitted from the young strong coins of the history, so
    the probability mass left unnamed at each place is calibrated, not guessed.
Deterministic: a fixed seed, so two runs on the same history are identical.
Standard library only, no network.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 987654321
SIMS = 1800
STEP_H = 0.5
MAX_CANDIDATES = 220
P_MIN = 0.004          # smallest named place probability worth emitting
NAMED_CAP = 0.985      # most of a place we ever name
TAIL_PROB = 0.08       # fat-tail mixture weight in the random walk
TAIL_MULT = 2.5
AGE_EDGES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_inputs():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and isinstance(m.get("rows"), list):
                marks.append(m)
    marks.sort(key=lambda m: m["at"])
    return task, marks


def logcap_sol(row):
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def age_bucket(age_h):
    for i, e in enumerate(AGE_EDGES):
        if age_h < e:
            return i
    return len(AGE_EDGES)


def build_stats(series, ages, mark_times):
    """Diffusion estimates from consecutive-mark log-cap increments, bucketed
    by (age bucket, complete), then age bucket, then global. Most listed coins
    are dormant between marks (increment exactly 0), so the estimator uses the
    time-weighted mean and the total variance of winsorized increments:
    drift = sum(dL)/sum(dt), sigma^2 = sum((dL - drift*dt)^2)/sum(dt).
    Returns drift_sigma(age_h, complete) -> (drift/hour, sigma/sqrt(hour))."""
    buckets = {}
    incs = {}
    for mint, obs in series.items():
        created = ages[mint]
        for (i0, l0, c0), (i1, l1, _c1) in zip(obs, obs[1:]):
            dt = (mark_times[i1] - mark_times[i0]) / 3600.0
            if dt <= 0:
                continue
            dl = max(-0.75, min(0.75, l1 - l0))
            age_h = (mark_times[i0] - created) / 3600.0
            b = age_bucket(age_h)
            for key in ((b, bool(c0)), (b, None), None):
                acc = buckets.setdefault(key, [0.0, 0.0, 0])
                acc[0] += dl
                acc[1] += dt
                acc[2] += 1
                incs.setdefault(key, []).append((dl, dt))

    def stat(key):
        acc = buckets.get(key)
        if not acc or acc[2] < 15 or acc[1] <= 0:
            return None
        drift = max(-1.5, min(1.5, acc[0] / acc[1]))
        var = 0.0
        for dl, dt in incs.get(key, []):
            var += (dl - drift * dt) ** 2
        sigma = math.sqrt(var / acc[1])
        return (drift, max(0.05, min(1.2, sigma)))

    cache = {}

    def drift_sigma(age_h, complete):
        b = age_bucket(age_h)
        for key in ((b, bool(complete)), (b, None), None):
            if key not in cache:
                cache[key] = stat(key)
            if cache[key] is not None:
                return cache[key]
        return (-0.05, 0.35)

    return drift_sigma


def main():
    task, marks = load_inputs()
    places = int(task.get("places", 5))
    edges = task["size"]["edges"]
    n_bins = len(edges) + 1
    reads = [parse_ts(r) for r in task["reads"]]
    out_name = task.get("output", "forecast.json")

    mark_times = [parse_ts(m["at"]).timestamp() for m in marks]
    cut = mark_times[-1]
    horizons = [max(0.25, (r.timestamp() - cut) / 3600.0) for r in reads]
    h_max = max(horizons)

    # Per-coin observation series.
    series = {}
    ages = {}
    last_seen = {}
    meta = {}
    for idx, m in enumerate(marks):
        for row in m["rows"]:
            mint = row["mint"]
            series.setdefault(mint, []).append(
                (idx, logcap_sol(row), bool(row.get("complete"))))
            ages[mint] = parse_ts(row["created_at"]).timestamp()
            last_seen[mint] = idx
            meta[mint] = row
    drift_sigma = build_stats(series, ages, mark_times)

    last_idx = len(marks) - 1
    place5_cap = None
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    if len(last_rows) >= places:
        place5_cap = logcap_sol(last_rows[places - 1])
    elif last_rows:
        place5_cap = logcap_sol(last_rows[-1])

    # Candidates: seen in the last mark, or strong in a recent mark; staleness
    # adjusted by bucket drift. Keep the best MAX_CANDIDATES.
    cand = {}
    for mint, obs in series.items():
        li = last_seen[mint]
        if li < last_idx - 2 and max(o[1] for o in obs) < (place5_cap or 0) + 0.3:
            continue
        l_last = obs[-1][1]
        stale_h = (cut - mark_times[li]) / 3600.0
        age_h = (cut - ages[mint]) / 3600.0
        d, _s = drift_sigma(age_h, obs[-1][2])
        adj = l_last + d * stale_h
        if place5_cap is not None and adj < place5_cap - 1.6 and li != last_idx:
            continue
        # momentum: shrunk slope over the coin's last up-to-3 observations
        mom = 0.0
        if len(obs) >= 3:
            (ia, la, _), (ib, lb, _) = obs[-3], obs[-1]
            dt = (mark_times[ib] - mark_times[ia]) / 3600.0
            if dt > 0:
                slope = max(-2.0, min(2.0, (lb - la) / dt))
                mom = 0.35 * slope
        cand[mint] = {
            "cap": adj, "age": age_h, "complete": obs[-1][2], "mom": mom,
        }
    top = sorted(cand.items(), key=lambda kv: -kv[1]["cap"])[:MAX_CANDIDATES]
    cand = dict(top)
    mints = list(cand.keys())

    # Phantom newcomers fitted from the history: each young coin's peak cap
    # measured as its excess over the place-5 cap of the same mark (so the
    # measure is scale-free as the whole chart grows over the window).
    hist_hours = max(0.5, (cut - parse_ts(task["t0"]).timestamp()) / 3600.0)
    p5_at = {}
    for idx, m in enumerate(marks):
        rows = sorted(m["rows"], key=lambda r: r["rank"])
        if len(rows) >= places:
            p5_at[idx] = logcap_sol(rows[places - 1])
    excess = []
    n_strong_coins = 0
    for mint, obs in series.items():
        pk = None
        for idx, l, _c in obs:
            if idx in p5_at and (mark_times[idx] - ages[mint]) / 3600.0 <= 2.5:
                e = l - p5_at[idx]
                pk = e if pk is None else max(pk, e)
        if pk is not None:
            excess.append(pk)
            if pk >= -0.7:
                n_strong_coins += 1
    strong = [e for e in excess if e >= -0.7]
    rate = (n_strong_coins / hist_hours) if strong else 0.0
    lam_total = min(16.0, rate * h_max)

    rng = random.Random(SEED)
    n_steps = int(math.ceil(h_max / STEP_H))
    read_steps = [min(n_steps, max(1, round(h / STEP_H))) for h in horizons]

    place_count = {m: [0] * places for m in mints}
    size_hist = [[0] * n_bins for _ in range(places)]
    total_outcomes = SIMS * len(reads)

    def bin_of(size):
        if size < edges[0]:
            return 0
        if size >= edges[-1]:
            return n_bins - 1
        lo, hi = 0, len(edges) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid] <= size:
                lo = mid + 1
            else:
                hi = mid
        return lo  # bin i holds edges[i-1] <= size < edges[i]

    for _sim in range(SIMS):
        # Walk every candidate to the furthest read, snapshotting at reads.
        snaps = {m: {} for m in mints}
        for m in mints:
            c = cand[m]
            x = c["cap"]
            age = c["age"]
            for s in range(1, n_steps + 1):
                d, sig = drift_sigma(age, c["complete"])
                step_drift = d * STEP_H
                if s * STEP_H <= 2.0:
                    step_drift += c["mom"] * STEP_H * math.exp(
                        -(s * STEP_H) / 1.5)
                z = rng.gauss(0.0, 1.0)
                if rng.random() < TAIL_PROB:
                    z *= TAIL_MULT
                x += step_drift + sig * math.sqrt(STEP_H) * z
                if x < -1.0:
                    x = -1.0
                age += STEP_H
                if s in read_steps:
                    snaps[m][s] = x
        # Phantom newcomers: births over the horizon, peak then decay.
        phantoms = []  # (birth_h, peak_logcap)
        if lam_total > 0:
            n_new = 0
            # Knuth Poisson
            limit = math.exp(-lam_total)
            p_prod = 1.0
            while True:
                p_prod *= rng.random()
                if p_prod <= limit:
                    break
                n_new += 1
                if n_new >= 16:
                    break
            for _ in range(n_new):
                birth = rng.uniform(0.05, max(0.1, h_max - 0.3))
                peak = (place5_cap or 0.0) + strong[rng.randrange(len(strong))]
                phantoms.append((birth, peak))
        for ri, s in enumerate(read_steps):
            h = s * STEP_H
            ranked = [(snaps[m][s], m) for m in mints]
            for birth, peak in phantoms:
                age = h - birth
                if age < 0.15:
                    continue
                d, sig = drift_sigma(age, True)
                cap = peak + d * max(0.0, age - 0.75) \
                    + sig * math.sqrt(max(0.0, age - 0.75)) * rng.gauss(0, 1)
                ranked.append((cap, None))
            ranked.sort(key=lambda t: (-t[0], t[1] or ""))
            for k in range(min(places, len(ranked))):
                cap, m = ranked[k]
                size_hist[k][bin_of(cap)] += 1
                if m is not None:
                    place_count[m][k] += 1

    # Place probabilities, averaged over simulations and reads.
    probabilities = []
    for k in range(places):
        rows = []
        for m in mints:
            p = place_count[m][k] / total_outcomes
            if p >= P_MIN:
                rows.append((m, p))
        rows.sort(key=lambda t: (-t[1], t[0]))
        ssum = sum(p for _m, p in rows)
        if ssum > NAMED_CAP:
            rows = [(m, p * NAMED_CAP / ssum) for m, p in rows]
        for m, p in rows:
            p6 = math.floor(p * 1_000_000) / 1_000_000
            if p6 > 0:
                probabilities.append({"mint": m, "place": k + 1, "p": p6})

    # top5: weighted place score.
    score = {}
    for m in mints:
        score[m] = sum((places - k) * place_count[m][k]
                       for k in range(places)) / total_outcomes
    top5 = [m for m, _s in sorted(
        score.items(), key=lambda kv: (-kv[1], kv[0]))[:places]]

    # Size bins: histogram with 10% neighbour smoothing, floored to 6 dp.
    sizes = []
    for k in range(places):
        hist = size_hist[k]
        tot = sum(hist) or 1
        raw = [c / tot for c in hist]
        smooth = []
        for i in range(n_bins):
            v = 0.8 * raw[i]
            if i > 0:
                v += 0.1 * raw[i - 1]
            if i < n_bins - 1:
                v += 0.1 * raw[i + 1]
            smooth.append(v)
        for b, v in enumerate(smooth):
            p6 = math.floor(v * 1_000_000) / 1_000_000
            if p6 >= 0.0005:
                sizes.append({"place": k + 1, "bin": b, "p": p6})

    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
