#!/usr/bin/env python3
"""pumpfun-top5-forecast-01: forecast the 5 largest new pump.fun coins by market cap.

Reads /input/task.json + /input/history.ndjson (10-minute marks of the coins
launched since the opening, ranked by market cap), writes /output/forecast.json.

Method (standard library only, deterministic):
  - For each coin, fit its log10 market-cap trajectory over the marks (recent
    weighted slope, shrunk toward the global prior by observation count).
  - Extrapolate to the mean answer-read horizon with a decaying drift
    (growth mean-reverts: total drift is capped), and a predictive sd that
    grows with horizon and with the coin's own residual volatility.
  - Monte Carlo (fixed-seed splitmix64 PRNG, so two runs on the same history
    write the same forecast): sample every candidate's cap at the horizon,
    rank, and count place frequencies (order probabilities) and the
    k-th-largest cap distribution (size-bin probabilities).
  - Reserve an explicit unnamed share per place for coins born after the
    cut (the practice shows a third of top-5 holders were newcomers), and
    name the sure losers at p=0 (up to the 100-row cap) so the unnamed share
    concentrates on the plausible-but-unnamed coins and newcomers, as the
    rules describe.
  - top5: the coins with the highest predicted mean log-cap, in order.

Outside the grader's container, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand
in for /input and /output:
  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tunables (fixed in advance, not tuned on the final; see notes) ---
# Env overrides exist only for local calibration runs; the grader sets none
# of them, so the submitted program always runs with these defaults.
def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


N_DRAWS = 50000          # Monte Carlo draws (deterministic seed)
PRNG_SEED = 0x9E3779B97F4A7C15
MAX_CANDIDATES = 220     # MC candidate coins (last mark + recent top coins)
RESERVE = _f("PUMPFUN_RESERVE", 0.35)        # unnamed share reserved per place
MIN_NAMED_P = 0.0005     # drop named probabilities below this (back to unnamed)
SLOPE_WINDOW_H = 3.0     # slope fit window (hours, capped by history span)
SLOPE_SHRINK_K = _f("PUMPFUN_SHRINK_K", 6.0)  # shrinkage strength
SLOPE_CAP = 1.5          # |slope| winsorized to this (log10 cap per hour)
DRIFT_TAU_H = _f("PUMPFUN_DRIFT_TAU", 2.5)    # drift decay tau (hours)
SD0 = _f("PUMPFUN_SD0", 0.35)                # baseline predictive sd (log10)
VOL_FLOOR = 0.20         # per-hour volatility floor (log10 / sqrt(hour))
VOL_CAP = 0.80           # per-hour volatility cap
SD_CAP = 1.20            # predictive sd cap (log10)
P0_MAX_PROB = 0.002      # p=0 rows only for coins with MC P(any top-5) below this
P0_MIN_RANK = 25         # p=0 rows only for coins ranked below this at the cut
MICRO = 1_000_000


# --- deterministic PRNG (splitmix64) + Box-Muller normals ---
class Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self):
        # 53-bit uniform in [0, 1)
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def normal_pair(self):
        u1 = self.uniform()
        while u1 <= 0.0:
            u1 = self.uniform()
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        t = 2.0 * math.pi * u2
        return r * math.cos(t), r * math.sin(t)


def parse_time(s):
    """ISO-8601 with Z -> epoch seconds (float)."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()


def logcap_sol(market_cap_lamports):
    """log10 of a market cap in SOL, as the scorer takes it."""
    try:
        lamports = int(market_cap_lamports)
    except (TypeError, ValueError):
        return None
    if lamports <= 0:
        return None
    return math.log10(lamports) - 9.0


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    output_name = task.get("output", "forecast.json")

    reads = [parse_time(r) for r in task["reads"]]
    history_until = parse_time(task["history_until"])
    h_mid = (sum(reads) / len(reads) - history_until) / 3600.0
    h_span = (max(reads) - min(reads)) / 3600.0
    h_mid = max(h_mid, 0.0)
    # sd horizon covers the later reads too
    h_sd = h_mid + 0.5 * h_span

    # --- parse marks ---
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is not True or not isinstance(m.get("rows"), list):
                continue
            rows = m["rows"]
            if not rows:
                continue
            at = parse_time(m["at"])
            marks.append((at, rows))
    if not marks:
        raise SystemExit("the history holds no ok mark with rows")
    marks.sort(key=lambda mr: mr[0])
    t_first = marks[0][0]
    t_last = marks[-1][0]
    span_h = max((t_last - t_first) / 3600.0, 1e-9)

    # --- per-coin series: mint -> list of (t_hours, logcap, rank) ---
    series = {}
    last_mark_mints = []
    last_mark_rank = {}
    for at, rows in marks:
        t_h = (at - t_first) / 3600.0
        seen = set()
        for row in rows:
            mint = row.get("mint")
            if not mint or mint in seen:
                continue
            seen.add(mint)
            y = logcap_sol(row.get("market_cap_lamports"))
            if y is None:
                continue
            rank = row.get("rank")
            series.setdefault(mint, []).append((t_h, y, rank))
    last_at, last_rows = marks[-1]
    for row in last_rows:
        mint = row.get("mint")
        if mint and mint not in last_mark_rank:
            last_mark_rank[mint] = row.get("rank", 10 ** 9)
            last_mark_mints.append(mint)
    if len(last_mark_mints) < places:
        # fall back: rank all coins by their last-seen cap
        by_last = sorted(series.items(), key=lambda kv: -kv[1][-1][1])
        last_mark_mints = [m for m, _ in by_last]
        last_mark_rank = {m: i + 1 for i, m in enumerate(last_mark_mints)}

    # --- global slope prior: median slope of coins with >=3 points ---
    def fit_slope(points, window_h):
        """Weighted least-squares slope over the recent window (per hour)."""
        if len(points) < 2:
            return None, 0
        t_end = points[-1][0]
        pts = [p for p in points if p[0] >= t_end - window_h]
        if len(pts) < 2:
            pts = points[-2:]
        n = len(pts)
        # weights ramp 0.5 -> 1.0 toward the recent end
        ws = [0.5 + 0.5 * (i + 1) / n for i in range(n)]
        sw = sum(ws)
        mt = sum(w * p[0] for w, p in zip(ws, pts)) / sw
        my = sum(w * p[1] for w, p in zip(ws, pts)) / sw
        num = sum(w * (p[0] - mt) * (p[1] - my) for w, p in zip(ws, pts))
        den = sum(w * (p[0] - mt) ** 2 for w, p in zip(ws, pts))
        if den <= 1e-12:
            return 0.0, n
        return num / den, n

    def resid_vol(points, slope):
        """Residual sd around the fitted line, per sqrt(hour) of span."""
        if len(points) < 3:
            return None
        t_end = points[-1][0]
        pts = [p for p in points if p[0] >= t_end - SLOPE_WINDOW_H]
        if len(pts) < 3:
            pts = points
        n = len(pts)
        mt = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        # intercept through the centroid with the fitted slope
        b = my - slope * mt
        ss = sum((p[1] - (slope * p[0] + b)) ** 2 for p in pts)
        span = max(pts[-1][0] - pts[0][0], 1e-9)
        var_per_h = ss / max(n - 2, 1) / span
        return math.sqrt(max(var_per_h, 0.0))

    window_h = min(SLOPE_WINDOW_H, span_h)
    slopes = {}
    for mint, pts in series.items():
        s, n = fit_slope(pts, window_h)
        slopes[mint] = (s, n)
    observed = sorted(
        s for s, n in slopes.values() if s is not None and n >= 3
    )
    if observed:
        prior_slope = observed[len(observed) // 2]
    else:
        prior_slope = -0.1
    prior_slope = max(-SLOPE_CAP, min(SLOPE_CAP, prior_slope))

    # --- candidate set: last mark + recent top-20 coins missing from it ---
    candidates = list(last_mark_mints)
    cand_set = set(candidates)
    for at, rows in reversed(marks[:-1]):
        for row in rows[:20]:
            mint = row.get("mint")
            if mint and mint not in cand_set:
                candidates.append(mint)
                cand_set.add(mint)
        if len(candidates) >= MAX_CANDIDATES:
            break
    candidates = candidates[:MAX_CANDIDATES]
    # deterministic order for sampling: by last-mark rank, then mint
    candidates.sort(key=lambda m: (last_mark_rank.get(m, 10 ** 9), m))

    # --- predictive mean/sd per candidate ---
    decay = DRIFT_TAU_H / (DRIFT_TAU_H + h_mid) if h_mid > 0 else 1.0
    means = []
    sds = []
    for mint in candidates:
        pts = series[mint]
        y_last = pts[-1][1]
        s, n = slopes[mint]
        if s is None:
            s_shrunk = prior_slope
        else:
            s_clipped = max(-SLOPE_CAP, min(SLOPE_CAP, s))
            w = n / (n + SLOPE_SHRINK_K)
            s_shrunk = w * s_clipped + (1.0 - w) * prior_slope
        drift = s_shrunk * h_mid * decay
        mean = y_last + drift
        vol = resid_vol(pts, s if s is not None else prior_slope)
        if vol is None:
            vol = VOL_FLOOR
        vol = max(VOL_FLOOR, min(VOL_CAP, vol))
        sd = math.sqrt(SD0 * SD0 + vol * vol * max(h_sd, 0.0))
        sd = min(sd, SD_CAP)
        means.append(mean)
        sds.append(sd)

    # --- Monte Carlo: place counts + place-cap histograms ---
    n_cand = len(candidates)
    place_counts = [[0] * (places + 1) for _ in range(n_cand)]
    size_counts = [[0] * (len(edges) + 1) for _ in range(places + 1)]
    rng = Rng(PRNG_SEED)
    n_bins = len(edges) + 1

    def bin_of(y):
        # bins: 0 = below edges[0]; i = [edges[i-1], edges[i]); last = >= edges[-1]
        lo, hi = 0, len(edges)
        while lo < hi:
            mid = (lo + hi) // 2
            if y < edges[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo  # 0..len(edges)

    for _ in range(N_DRAWS):
        vals = [0.0] * n_cand
        i = 0
        while i < n_cand:
            z1, z2 = rng.normal_pair()
            vals[i] = means[i] + sds[i] * z1
            i += 1
            if i < n_cand:
                vals[i] = means[i] + sds[i] * z2
                i += 1
        order = sorted(range(n_cand), key=lambda j: (-vals[j], candidates[j]))
        for k in range(places):
            j = order[k]
            place_counts[j][k + 1] += 1
            size_counts[k + 1][bin_of(vals[j])] += 1

    # --- order probabilities: scale by (1 - RESERVE), floor to 6 decimals ---
    probabilities = []
    named_per_place = {}
    for k in range(1, places + 1):
        rows = []
        for j, mint in enumerate(candidates):
            p = (place_counts[j][k] / N_DRAWS) * (1.0 - RESERVE)
            u = math.floor(p * MICRO)
            if u >= math.floor(MIN_NAMED_P * MICRO):
                rows.append((mint, u / MICRO))
        rows.sort(key=lambda r: (-r[1], r[0]))
        named_per_place[k] = {m for m, _ in rows}
        for mint, p in rows:
            probabilities.append({"mint": mint, "place": k, "p": p})

    # --- p=0 rows: sure losers, filling toward the 100-row cap per place ---
    any_top5 = [
        sum(place_counts[j][k] for k in range(1, places + 1)) / N_DRAWS
        for j in range(n_cand)
    ]
    mean_by_mint = {m: means[j] for j, m in enumerate(candidates)}
    p0_pool = [
        m
        for j, m in enumerate(candidates)
        if m in last_mark_rank
        and last_mark_rank[m] > P0_MIN_RANK
        and any_top5[j] < P0_MAX_PROB
    ]
    p0_pool.sort(key=lambda m: (mean_by_mint[m], m))
    for k in range(1, places + 1):
        used = len(named_per_place[k])
        budget = 100 - used
        if budget <= 0:
            continue
        added = 0
        for mint in p0_pool:
            if added >= budget:
                break
            if mint in named_per_place[k]:
                continue
            probabilities.append({"mint": mint, "place": k, "p": 0})
            added += 1

    # --- sizes: MC histograms, floored to 6 decimals, zeros dropped ---
    sizes = []
    for k in range(1, places + 1):
        for b in range(n_bins):
            u = math.floor((size_counts[k][b] / N_DRAWS) * MICRO)
            if u > 0:
                sizes.append({"place": k, "bin": b, "p": u / MICRO})

    # --- top5: highest predicted mean, in order ---
    top5 = [
        m
        for m, _ in sorted(
            zip(candidates, means), key=lambda cm: (-cm[1], cm[0])
        )[:places]
    ]

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))
    sizes.sort(key=lambda r: (r["place"], r["bin"]))
    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    payload = json.dumps(out) + "\n"
    if len(payload.encode("utf-8")) > 1_048_576:
        raise SystemExit("forecast exceeds the 1 MiB limit")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(payload)


if __name__ == "__main__":
    main()
