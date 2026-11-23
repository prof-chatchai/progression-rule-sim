"""Compare candidate decision-rule settings before registration (MCRU-Province protocol).

Simulation only. Same data-generating model as sim_algorithm1.py (teachers/classrooms nested in 9 schools,
binary delivery and participation units, lognormal preparation time, ICC 0.2).
Candidates differ in interval level (95/90/80%), GO thresholds for delivery/participation, and whether the
leave-one-school-out widening is applied. STOP thresholds (0.50, 0.50, 6 h) and the aggregation rule are unchanged.
Usage: python -I -X utf8 sim_candidates.py [--R 1000] [--B 400] [--seed 20261123] [--out <dir>]
"""
import argparse, csv, json, math, os, platform, sys, time
import numpy as np

SCHOOLS = 9
STOP_DEL = STOP_PAR = 0.50
GO_PREP, STOP_PREP = 3.0, 6.0
ZS = {0.95: 1.959964, 0.90: 1.644854, 0.80: 1.281552}
QS = {0.95: (2.5, 97.5), 0.90: (5.0, 95.0), 0.80: (10.0, 90.0)}
CANDS = {
    "C0_proposal":     dict(level=0.95, go_del=0.70, go_par=0.75, loo=True),
    "C1_go60_95":      dict(level=0.95, go_del=0.60, go_par=0.65, loo=True),
    "C2_ci90":         dict(level=0.90, go_del=0.70, go_par=0.75, loo=True),
    "C3_ci80":         dict(level=0.80, go_del=0.70, go_par=0.75, loo=True),
    "C4_go60_ci90":    dict(level=0.90, go_del=0.60, go_par=0.65, loo=True),
    "C5_go60_ci80":    dict(level=0.80, go_del=0.60, go_par=0.65, loo=True),
    "C6_ci80_noLOO":   dict(level=0.80, go_del=0.70, go_par=0.75, loo=False),
    "C7_go60_ci90_noLOO": dict(level=0.90, go_del=0.60, go_par=0.65, loo=False),
}
LABELS = ["GO", "ADAPT-LOW", "ADAPT-UNCERTAIN", "STOP"]


def wilson(x, n, z):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_q(x, rng, B):
    idx = rng.integers(0, len(x), (B, len(x)))
    m = np.median(x[idx], axis=1)
    qs = np.percentile(m, [2.5, 5.0, 10.0, 90.0, 95.0, 97.5])
    return {0.95: (qs[0], qs[5]), 0.90: (qs[1], qs[4]), 0.80: (qs[2], qs[3])}


def decide(L, U, go, stop, reverse=False):
    if not reverse:
        if L >= go: return "GO"
        if U < stop: return "STOP"
        if U < go: return "ADAPT-LOW"
        return "ADAPT-UNCERTAIN"
    if U <= go: return "GO"
    if L > stop: return "STOP"
    if L > go: return "ADAPT-LOW"
    return "ADAPT-UNCERTAIN"


def aggregate(ds):
    if "STOP" in ds: return "STOP"
    if all(d == "GO" for d in ds): return "GO"
    if "ADAPT-LOW" in ds: return "ADAPT-LOW"
    return "ADAPT-UNCERTAIN"


def draw_binary(n, pi, rho, schools, rng):
    if rho > 0:
        a, b = pi * (1 - rho) / rho, (1 - pi) * (1 - rho) / rho
        ps = rng.beta(a, b, SCHOOLS)
    else:
        ps = np.full(SCHOOLS, pi)
    return (rng.random(n) < ps[schools]).astype(int)


def draw_prep(n, median, rho, schools, rng, tot_sd=0.55):
    sb, sw = math.sqrt(rho) * tot_sd, math.sqrt(1 - rho) * tot_sd
    u = rng.normal(0, sb, SCHOOLS) if sb > 0 else np.zeros(SCHOOLS)
    return np.exp(math.log(median) + u[schools] + rng.normal(0, sw, n))


def prop_iv(x, schools):
    """intervals[level] = (full, loo_widened)"""
    n, k = len(x), int(x.sum())
    out = {}
    for lv, z in ZS.items():
        L1, U1 = wilson(k, n, z)
        Ls, Us = [L1], [U1]
        for s in range(SCHOOLS):
            m = schools != s
            l, u = wilson(int(x[m].sum()), int(m.sum()), z)
            Ls.append(l); Us.append(u)
        out[lv] = ((L1, U1), (min(Ls), max(Us)))
    return out


def prep_iv(x, schools, rng, B):
    full = boot_q(x, rng, B)
    loo = [boot_q(x[schools != s], rng, B) for s in range(SCHOOLS)]
    out = {}
    for lv in ZS:
        Ls = [full[lv][0]] + [q[lv][0] for q in loo]
        Us = [full[lv][1]] + [q[lv][1] for q in loo]
        out[lv] = (full[lv], (min(Ls), max(Us)))
    return out


def one_dataset(n, pdel, ppar, pprep, rho, rng, B):
    schools = np.repeat(np.arange(SCHOOLS), n // SCHOOLS)
    iv_d = prop_iv(draw_binary(n, pdel, rho, schools, rng), schools)
    iv_p = prop_iv(draw_binary(n, ppar, rho, schools, rng), schools)
    iv_h = prep_iv(draw_prep(n, pprep, rho, schools, rng), schools, rng, B)
    res = {}
    for nm, c in CANDS.items():
        k = 1 if c["loo"] else 0
        lv = c["level"]
        ds = [decide(*iv_d[lv][k], c["go_del"], STOP_DEL),
              decide(*iv_p[lv][k], c["go_par"], STOP_PAR),
              decide(*iv_h[lv][k], GO_PREP, STOP_PREP, reverse=True)]
        res[nm] = aggregate(ds)
    return res


def run_config(name, n, pdel, ppar, pprep, rho, R, B, rng):
    cnt = {c: {l: 0 for l in LABELS} for c in CANDS}
    for _ in range(R):
        for c, d in one_dataset(n, pdel, ppar, pprep, rho, rng, B).items(): cnt[c][d] += 1
    rows = []
    for c in CANDS:
        row = dict(scenario=name, n=n, p_delivery=pdel, p_participation=ppar, prep_median_h=pprep, icc=rho, candidate=c, R=R)
        for l in LABELS: row["P_" + l] = round(cnt[c][l] / R, 4)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--B", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows, t0 = [], time.time()
    for n in (18, 27, 36):
        for pdel in (0.50, 0.55, 0.60, 0.70, 0.80, 0.85, 0.95):
            rows += run_config("A_delivery_sweep", n, pdel, 0.92, 2.0, 0.2, a.R, a.B, rng)
        print("A n", n, round(time.time() - t0), "s", file=sys.stderr, flush=True)
    profiles = {"B_meets_go": (0.85, 0.85, 2.0), "B_prep_too_high": (0.85, 0.85, 4.5),
                "B_participation_low": (0.85, 0.55, 2.0), "B_poor": (0.40, 0.40, 7.0)}
    for nm, (pd_, pp_, ph_) in profiles.items():
        for n in (18, 27, 36):
            rows += run_config(nm, n, pd_, pp_, ph_, 0.2, a.R, a.B, rng)
        print(nm, round(time.time() - t0), "s", file=sys.stderr, flush=True)
    p = os.path.join(a.out, "results_candidates.csv")
    with open(p, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump(dict(seed=a.seed, R=a.R, B=a.B, numpy=np.__version__, python=platform.python_version(),
                   seconds=round(time.time() - t0), candidates=CANDS),
              open(os.path.join(a.out, "run_meta_candidates.json"), "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
