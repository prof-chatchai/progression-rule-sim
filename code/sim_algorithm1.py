"""Operating characteristics of the pre-specified decision rule (Algorithm 1), MCRU-Province protocol.

Simulation only: no human data. Teachers are nested in 9 schools. Three core indicators:
  delivery       (proportion of teachers who delivered as planned; go >= 0.70, stop upper < 0.50)
  participation  (proportion of classrooms reaching planned attendance; go >= 0.75, stop upper < 0.50)
  prep time      (median extra preparation hours per unit, lower is better; go upper <= 3, stop lower > 6)
Three rules are compared:
  R0 point estimate only (L = U = estimate)
  R1 teacher-level interval (Wilson for proportions, percentile bootstrap of the median)
  R2 proposal rule: widest of R1 and the nine leave-one-school-out intervals
Usage: python -I -X utf8 sim_algorithm1.py [--R 1000] [--B 400] [--seed 20261122] [--out <dir>]
"""
import argparse, csv, json, math, os, platform, sys, time
import numpy as np

Z = 1.96
GO_DEL, STOP_DEL = 0.70, 0.50
GO_PAR, STOP_PAR = 0.75, 0.50
GO_PREP, STOP_PREP = 3.0, 6.0
SCHOOLS = 9


def wilson(x, n):
    p = x / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def boot_ci(x, rng, B):
    idx = rng.integers(0, len(x), (B, len(x)))
    m = np.median(x[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


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
    sb = math.sqrt(rho) * tot_sd
    sw = math.sqrt(1 - rho) * tot_sd
    u = rng.normal(0, sb, SCHOOLS) if sb > 0 else np.zeros(SCHOOLS)
    return np.exp(math.log(median) + u[schools] + rng.normal(0, sw, n))


def prop_intervals(x, schools):
    n = len(x); k = int(x.sum())
    est = k / n
    L1, U1 = wilson(k, n)
    Ls, Us = [L1], [U1]
    for s in range(SCHOOLS):
        m = schools != s
        l, u = wilson(int(x[m].sum()), int(m.sum()))
        Ls.append(l); Us.append(u)
    return (est, est), (L1, U1), (min(Ls), max(Us))


def prep_intervals(x, schools, rng, B):
    est = float(np.median(x))
    L1, U1 = boot_ci(x, rng, B)
    Ls, Us = [L1], [U1]
    for s in range(SCHOOLS):
        l, u = boot_ci(x[schools != s], rng, B)
        Ls.append(l); Us.append(u)
    return (est, est), (L1, U1), (min(Ls), max(Us))


def one_dataset(n, pdel, ppar, pprep, rho, rng, B):
    schools = np.repeat(np.arange(SCHOOLS), n // SCHOOLS)
    xd = draw_binary(n, pdel, rho, schools, rng)
    xp = draw_binary(n, ppar, rho, schools, rng)
    xh = draw_prep(n, pprep, rho, schools, rng)
    iv = [prop_intervals(xd, schools), prop_intervals(xp, schools), prep_intervals(xh, schools, rng, B)]
    out = []
    for r in range(3):
        ds = [decide(*iv[0][r], GO_DEL, STOP_DEL), decide(*iv[1][r], GO_PAR, STOP_PAR),
              decide(*iv[2][r], GO_PREP, STOP_PREP, reverse=True)]
        out.append(aggregate(ds))
    return out


LABELS = ["GO", "ADAPT-LOW", "ADAPT-UNCERTAIN", "STOP"]
RULES = ["R0_point", "R1_interval", "R2_proposal"]


def run_config(name, n, pdel, ppar, pprep, rho, R, B, rng):
    cnt = [{l: 0 for l in LABELS} for _ in RULES]
    r1go_r2not = 0
    for _ in range(R):
        res = one_dataset(n, pdel, ppar, pprep, rho, rng, B)
        for i, d in enumerate(res): cnt[i][d] += 1
        if res[1] == "GO" and res[2] != "GO": r1go_r2not += 1
    rows = []
    for i, rl in enumerate(RULES):
        row = dict(scenario=name, n=n, p_delivery=pdel, p_participation=ppar, prep_median_h=pprep, icc=rho, rule=rl, R=R)
        for l in LABELS: row["P_" + l] = round(cnt[i][l] / R, 4)
        row["P_R1GO_but_R2notGO"] = round(r1go_r2not / R, 4) if rl == "R2_proposal" else ""
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--B", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    t0 = time.time()
    # A. delivery sweep (other indicators set to good values, so delivery drives the decision)
    for rho in (0.0, 0.2):
        for n in (18, 27, 36):
            for pdel in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
                rows += run_config("A_delivery_sweep", n, pdel, 0.92, 2.0, rho, a.R, a.B, rng)
            print("A done rho", rho, "n", n, round(time.time() - t0), "s", file=sys.stderr, flush=True)
    # B. joint profiles at rho = 0.2
    profiles = {"B_good": (0.95, 0.92, 2.0), "B_at_threshold": (0.70, 0.75, 3.0),
                "B_poor": (0.40, 0.40, 7.0), "B_prep_problem": (0.90, 0.90, 4.5)}
    for nm, (pd_, pp_, ph_) in profiles.items():
        for n in (18, 27, 36):
            rows += run_config(nm, n, pd_, pp_, ph_, 0.2, a.R, a.B, rng)
        print(nm, "done", round(time.time() - t0), "s", file=sys.stderr, flush=True)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "results_algorithm1.csv")
    with open(p, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    meta = dict(seed=a.seed, R=a.R, B=a.B, numpy=np.__version__, python=platform.python_version(),
                seconds=round(time.time() - t0))
    json.dump(meta, open(os.path.join(a.out, "run_meta.json"), "w"), indent=1)
    print("wrote", p, meta)


if __name__ == "__main__":
    main()
