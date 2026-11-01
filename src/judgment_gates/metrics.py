import math

import numpy as np
from scipy.stats import kurtosis, norm, skew


def calibration(y, p, bins=10):
    """Compute Brier score, equal-width reliability bins and expected calibration error."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if (
        not len(y)
        or y.shape != p.shape
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("aligned binary labels and finite probabilities required")
    index = np.minimum((p * bins).astype(int), bins - 1)
    reliability, ece = [], 0.0
    for i in range(bins):
        mask = index == i
        n = int(mask.sum())
        confidence = float(p[mask].mean()) if n else None
        frequency = float(y[mask].mean()) if n else None
        if n:
            ece += n / len(y) * abs(confidence - frequency)
        reliability.append(
            {
                "lower": i / bins,
                "upper": (i + 1) / bins,
                "n": n,
                "probability": confidence,
                "frequency": frequency,
            }
        )
    return {
        "n": len(y),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece,
        "prevalence": float(y.mean()),
        "reliability": reliability,
    }


def block_ci(values, timestamps, block="hour", seed=42, repetitions=2000, alpha=0.05):
    """Bootstrap calendar-block sums and counts, withholding intervals below five blocks."""
    values, timestamps = np.asarray(values, float), np.asarray(timestamps, dtype=np.int64)
    duration = 3_600_000_000_000 * (24 if block == "day" else 1)
    groups = timestamps // duration
    keys = np.unique(groups)
    point = float(values.mean()) if len(values) else None
    result = {"mean": point, "ci": None, "blocks": len(keys), "block": block, "n": len(values)}
    if len(keys) < 5:
        return {**result, "reason": "fewer than five UTC blocks; precision is not established"}
    sums = np.array([values[groups == g].sum() for g in keys])
    counts = np.array([(groups == g).sum() for g in keys])
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(keys), size=(repetitions, len(keys)))
    draws = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    ci = np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist()
    return {
        **result,
        "ci": ci,
        "half_width": (ci[1] - ci[0]) / 2,
        "repetitions": repetitions,
        "alpha": alpha,
    }


def paired_state_ci(values, seed=42, repetitions=2000, alpha=0.05):
    """Bootstrap paired state differences; this is a small-cohort descriptive interval."""
    values = np.asarray(values, float)
    result = {
        "ci": None,
        "n": len(values),
        "seed": seed,
        "repetitions": repetitions,
        "alpha": alpha,
        "unit": "paired market state",
        "scope": "descriptive conditional-cohort interval; assumes exchangeable state pairs; "
        "does not establish time-series or population precision and has no multiplicity correction",
    }
    if len(values) < 2:
        return {**result, "reason": "fewer than two paired states"}
    indices = np.random.default_rng(seed).integers(len(values), size=(repetitions, len(values)))
    result["ci"] = np.quantile(values[indices].mean(axis=1), [alpha / 2, 1 - alpha / 2]).tolist()
    return result


def latency(samples):
    """Summarise empirical latency samples in milliseconds."""
    if not samples:
        return {"n": 0, "p50_ms": None, "p99_ms": None}
    return {
        "n": len(samples),
        "p50_ms": float(np.quantile(samples, 0.5)),
        "p99_ms": float(np.quantile(samples, 0.99)),
        "samples_ms": list(samples),
    }


def risk_intervals(returns, hours, capital, block, repetitions=2000):
    """Bootstrap calendar-block returns for descriptive risk intervals."""
    groups = hours // (24 if block == "day" else 1)
    keys = np.unique(groups)
    result = {
        "block": block,
        "blocks": len(keys),
        "repetitions": repetitions,
        "seed": 42,
        "confidence": 0.95,
        "pnl_usd": None,
        "sharpe": None,
        "sortino": None,
        "note": "Descriptive percentile intervals; stationary calendar blocks assumed. "
        "P&L normalised to the observed number of hours.",
    }
    if len(keys) < 5:
        return {**result, "reason": "fewer than five UTC blocks"}
    chunks = [returns[groups == g] for g in keys]
    rng = np.random.default_rng(42)
    draws = {key: [] for key in ("pnl_usd", "sharpe", "sortino")}
    for _ in range(repetitions):
        sample = np.concatenate([chunks[i] for i in rng.integers(len(keys), size=len(keys))])
        mean, std = float(sample.mean()), float(sample.std(ddof=1))
        downside = float(np.sqrt(np.mean(np.minimum(sample, 0) ** 2)))
        draws["pnl_usd"].append(mean * len(returns) * capital)
        draws["sharpe"].append(mean / std * math.sqrt(24 * 365) if std else np.nan)
        draws["sortino"].append(mean / downside * math.sqrt(24 * 365) if downside else np.nan)
    result["undefined_resamples"] = {}
    for name, values in draws.items():
        invalid = int((~np.isfinite(values)).sum())
        result["undefined_resamples"][name] = invalid
        if not invalid:
            result[name] = np.quantile(values, [0.025, 0.975]).tolist()
    return result


def performance(equity, capital, trials, variant_sharpes=()):
    """Compute observed P&L risk metrics and explicitly qualified DSR sensitivity."""
    if not equity:
        return {"sharpe": None, "sortino": None, "dsr": None, "max_drawdown_usd": 0.0}
    pnl = np.array([r["pnl"] for r in equity])
    groups = np.array([r["ns"] // 3_600_000_000_000 for r in equity])
    hours = np.unique(groups)
    hourly = np.array([pnl[np.flatnonzero(groups == g)[-1]] for g in hours])
    returns = np.diff(np.r_[0, hourly]) / capital
    drawdown = float(np.max(np.maximum.accumulate(np.r_[0, pnl])[1:] - pnl))
    result = {
        "hourly_returns": returns.tolist(),
        "hours": len(returns),
        "max_drawdown_usd": drawdown,
        "sharpe": None,
        "sortino": None,
        "dsr": None,
        "trials": trials,
        "dsr_reason": "all-trial dispersion unavailable; retained-variant sensitivity only",
        "dsr_sensitivity_retained_variants": None,
        "missing_hours": int(np.maximum(np.diff(hours) - 1, 0).sum()),
        "risk_intervals": None,
    }
    if result["missing_hours"]:
        return {**result, "reason": "missing hourly marks; no compressed-time annualisation"}
    if len(returns) < 20 or np.std(returns, ddof=1) == 0:
        return {**result, "reason": "need twenty variable hourly returns for ratio estimates"}
    sr = float(np.mean(returns) / np.std(returns, ddof=1))
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)))
    result.update(
        sharpe=sr * math.sqrt(24 * 365),
        sortino=(float(returns.mean() / downside * math.sqrt(24 * 365)) if downside else None),
    )
    result["risk_intervals"] = {
        block: risk_intervals(returns, hours, capital, block) for block in ("hour", "day")
    }
    # Bailey/López de Prado: expected maximum trial SR, then probabilistic SR.
    if trials > 1 and len(variant_sharpes) >= 2:
        std = float(np.std(variant_sharpes, ddof=1))
        gamma = 0.5772156649015329
        benchmark = std * (
            (1 - gamma) * norm.ppf(1 - 1 / trials) + gamma * norm.ppf(1 - 1 / (trials * math.e))
        )
        denom = (
            1 - float(skew(returns)) * sr + (float(kurtosis(returns, fisher=False)) - 1) * sr**2 / 4
        )
        if denom > 0:
            result["dsr_sensitivity_retained_variants"] = float(
                norm.cdf((sr - benchmark) * math.sqrt(len(returns) - 1) / math.sqrt(denom))
            )
            result["sensitivity_benchmark_hourly_sr"] = benchmark
    return result


def markouts(fills, states):
    """Measure signed post-fill moves at observable endpoints with censoring."""
    times = np.array([s["ns"] for s in states], dtype=np.int64)
    result = {}
    for seconds in (1, 10, 60):
        values, stamps, adverse_usd = [], [], 0.0
        for fill in fills:
            i = int(np.searchsorted(times, fill["ns"] + seconds * 1_000_000_000))
            if i == len(states) or times[i] > fill["ns"] + (seconds + 2) * 1e9:
                continue
            move = fill["side"] * (states[i]["mid"] - fill["mid"])
            values.append(1e4 * move / fill["mid"])
            stamps.append(fill["ns"])
            adverse_usd += fill["size"] * move
        result[str(seconds)] = {
            "hour": block_ci(values, stamps),
            "day": block_ci(values, stamps, "day"),
            "signed_future_move_usd": adverse_usd,
            "censored": len(fills) - len(values),
        }
    return result
