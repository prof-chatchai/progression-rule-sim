"""Final registered-rule simulation (MCRU-Province): rule C5 with 27 delivering teachers in 9 schools.

Rule C5: 80% intervals; leave-one-school-out widening; GO delivery >= 0.60, GO participation >= 0.65,
STOP when the upper bound < 0.50 (both proportions); extra preparation time (median hours): GO when the upper bound <= 3,
STOP when the lower bound > 6; overall GO needs every core indicator GO, STOP if any core indicator STOP.
Simulation only. Units: teacher (delivery) and classroom (participation) are binary; beta-binomial school effect;
lognormal preparation time with a school effect (total log-sd 0.55).
Phases: (1) decisions incl. attrition and fallback n = 18; (2) coverage of the 80% intervals.
Usage: python -I -X utf8 sim_final.py [--R 1000] [--B 300] [--seed 20261125] [--out <dir>]
"""
import argparse, csv, json, math, os, platform, sys, time
import numpy as np

S = 9
Z80 = 1.281552
GO_DEL, GO_PAR, STOP_PR = 0.60, 0.65, 0.50
GO_PREP, STOP_PREP = 3.0, 6.0
LAB = ["GO", "ADAPT-LOW", "ADAPT-UNCERTAIN", "STOP"]


def wilson(x, n, z=Z80):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot80(x, rng, B):
    idx = rng.integers(0, len(x), (B, len(x)))
    m = np.median(x[idx], axis=1)
    q = np.percentile(m, [10.0, 90.0])
    return float(q[0]), float(q[1])


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


def agg(ds):
    if "STOP" in ds: return "STOP"
    if all(d == "GO" for d in ds): return "GO"
    if "ADAPT-LOW" in ds: return "ADAPT-LOW"
    return "ADAPT-UNCERTAIN"


def draw_bin(n, pi, rho, schools, rng):
    ps = rng.beta(pi * (1 - rho) / rho, (1 - pi) * (1 - rho) / rho, S) if rho > 0 else np.full(S, pi)
    return (rng.random(n) < ps[schools]).astype(int)


def draw_prep(n, med, rho, schools, rng, sd=0.55):
    sb, sw = math.sqrt(rho) * sd, math.sqrt(1 - rho) * sd
    u = rng.normal(0, sb, S) if sb > 0 else np.zeros(S)
    return np.exp(math.log(med) + u[schools] + rng.normal(0, sw, n))


def widened_prop(x, schools):
    n, k = len(x), int(x.sum())
    L1, U1 = wilson(k, n)
    Ls, Us = [L1], [U1]
    for s in range(S):
        m = schools != s
        if m.sum() == n or m.sum() == 0: continue
        l, u = wilson(int(x[m].sum()), int(m.sum())); Ls.append(l); Us.append(u)
    return (L1, U1), (min(Ls), max(Us))


def widened_prep(x, schools, rng, B):
    L1, U1 = boot80(x, rng, B)
    Ls, Us = [L1], [U1]
    for s in range(S):
        m = schools != s
        if m.sum() == len(x) or m.sum() == 0: continue
        l, u = boot80(x[m], rng, B); Ls.append(l); Us.append(u)
    return (L1, U1), (min(Ls), max(Us))


def one(n, drop, pdel, ppar, pprep, rho, rng, B):
    schools = np.repeat(np.arange(S), n // S)
    xd = draw_bin(n, pdel, rho, schools, rng); xp = draw_bin(n, ppar, rho, schools, rng); xh = draw_prep(n, pprep, rho, schools, rng)
    if drop > 0:
        keep = rng.random(n) >= drop
        if keep.sum() < 6: keep[:6] = True
        xd, xp, xh, schools = xd[keep], xp[keep], xh[keep], schools[keep]
    d1 = decide(*widened_prop(xd, schools)[1], GO_DEL, STOP_PR)
    d2 = decide(*widened_prop(xp, schools)[1], GO_PAR, STOP_PR)
    d3 = decide(*widened_prep(xh, schools, rng, B)[1], GO_PREP, STOP_PREP, reverse=True)
    return agg([d1, d2, d3]), d3


def config(name, n, drop, pdel, ppar, pprep, rho, R, B, rng):
    c = {l: 0 for l in LAB}; cp = {l: 0 for l in LAB}
    for _ in range(R):
        a, p = one(n, drop, pdel, ppar, pprep, rho, rng, B); c[a] += 1; cp[p] += 1
    row = dict(scenario=name, n_planned=n, dropout=drop, p_delivery=pdel, p_participation=ppar, prep_median_h=pprep, icc=rho, R=R)
    for l in LAB: row["P_" + l] = round(c[l] / R, 4)
    for l in LAB: row["prepIndicator_" + l] = round(cp[l] / R, 4)
    return row


def coverage(R, Rb, B, rng):
    rows = []
    for rho in (0.0, 0.1, 0.2, 0.3):
        for n in (18, 27, 36):
            schools = np.repeat(np.arange(S), n // S)
            for pi in (0.6, 0.8, 0.9):
                c1 = c2 = 0
                for _ in range(R):
                    x = draw_bin(n, pi, rho, schools, rng)
                    (L1, U1), (L2, U2) = widened_prop(x, schools)
                    c1 += (L1 <= pi <= U1); c2 += (L2 <= pi <= U2)
                rows.append(dict(type="proportion", icc=rho, n=n, truth=pi, level=0.80, cover_plain=round(c1 / R, 4), cover_widened=round(c2 / R, 4), R=R))
    for rho in (0.0, 0.2):
        for n in (18, 27):
            schools = np.repeat(np.arange(S), n // S)
            for med in (2.0, 4.0):
                c1 = c2 = 0
                for _ in range(Rb):
                    x = draw_prep(n, med, rho, schools, rng)
                    (L1, U1), (L2, U2) = widened_prep(x, schools, rng, B)
                    c1 += (L1 <= med <= U1); c2 += (L2 <= med <= U2)
                rows.append(dict(type="median_prep_hours", icc=rho, n=n, truth=med, level=0.80, cover_plain=round(c1 / Rb, 4), cover_widened=round(c2 / Rb, 4), R=Rb))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); t0 = time.time(); rows = []
    for rho in (0.1, 0.2, 0.3):
        for pdel in (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95):
            rows.append(config("A_delivery_sweep", 27, 0.0, pdel, 0.92, 2.0, rho, a.R, a.B, rng))
        print("A rho", rho, round(time.time() - t0), "s", file=sys.stderr, flush=True)
    prof = {"B_meets_go": (0.85, 0.85, 2.0), "B_at_threshold": (0.60, 0.65, 3.0), "B_delivery_low": (0.55, 0.85, 2.0),
            "B_participation_low": (0.85, 0.55, 2.0), "B_prep_high": (0.85, 0.85, 4.5), "B_poor": (0.40, 0.40, 7.0), "B_very_good": (0.95, 0.92, 2.0)}
    for nm, (d_, p_, h_) in prof.items():
        rows.append(config(nm, 27, 0.0, d_, p_, h_, 0.2, a.R, a.B, rng))
    print("B done", round(time.time() - t0), "s", file=sys.stderr, flush=True)
    for h in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0):
        rows.append(config("C_prep_sweep", 27, 0.0, 0.85, 0.85, h, 0.2, a.R, a.B, rng))
    print("C done", round(time.time() - t0), "s", file=sys.stderr, flush=True)
    for nm in ("B_meets_go", "B_delivery_low", "B_poor"):
        d_, p_, h_ = prof[nm]
        for (n, dr) in ((27, 0.10), (27, 0.20), (18, 0.0)):
            rows.append(config("D_attrition_" + nm[2:], n, dr, d_, p_, h_, 0.2, a.R, a.B, rng))
    print("D done", round(time.time() - t0), "s", file=sys.stderr, flush=True)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "results_final_decisions.csv")
    with open(p, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    cv = coverage(4000, 1500, 300, rng)
    p2 = os.path.join(a.out, "results_final_coverage80.csv")
    with open(p2, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(cv[0].keys())); w.writeheader(); w.writerows(cv)
    json.dump(dict(seed=a.seed, R=a.R, B=a.B, numpy=np.__version__, python=platform.python_version(), seconds=round(time.time() - t0),
                   rule="C5: 80% intervals, LOO widening, GO .60/.65, STOP upper<.50, prep GO<=3 STOP>6", n_delivering_teachers=27),
              open(os.path.join(a.out, "run_meta_final.json"), "w"), indent=1)
    print("wrote", p, p2)


if __name__ == "__main__":
    main()
