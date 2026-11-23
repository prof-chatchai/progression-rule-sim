"""Comparison of interval/decision methods for ONE proportion indicator (delivery) with teachers nested in 9 schools.
ADEMP summary
 Aims      : does the registered rule (M2: 80% Wilson widened by leave-one-school-out) behave better than standard alternatives?
 DGM       : school probabilities ~ Beta(mean pi, ICC rho); teacher outcomes Bernoulli; 9 schools; n = 18, 27, 36 (balanced) or 27 unbalanced;
             optional informative withdrawal (failing teachers withdraw more often).
 Estimand  : the proportion of teachers who deliver as planned (true value = pi); decision GO if lower bound >= 0.60, STOP if upper bound < 0.50,
             ADAPT-LOW if upper < 0.60, otherwise ADAPT-UNCERTAIN (same rule as Algorithm 1, delivery only).
 Methods   : M0 point estimate; M1 teacher-level Wilson; M2 registered rule (Wilson widened by leave-one-school-out); M3 design-effect Wilson with ANOVA-
             estimated ICC; M3o design-effect Wilson with the TRUE ICC (oracle); M4 cluster (school) bootstrap percentile; M5 school-level t-interval
             on school proportions (df 8); M6 Clopper-Pearson exact (ignores clustering); M7 Jeffreys Bayesian interval (ignores clustering). All at 80%.
 Perf.     : probabilities of each decision, coverage of the true pi, mean width, Monte Carlo SE; exact check at ICC = 0; seed stability.
Usage: python -I -X utf8 sim_compare.py [--R 10000] [--seeds 10] [--out <dir>]
"""
import argparse, csv, json, math, os, platform, sys, time
import numpy as np
from scipy import stats

S = 9
Z80 = float(stats.norm.ppf(0.90)); T80 = float(stats.t.ppf(0.90, S - 1))
GO, STOP = 0.60, 0.50
METHODS = ["M0_point", "M1_wilson", "M2_registered", "M3_deff_est", "M3o_deff_true", "M4_cluster_boot", "M5_school_t", "M6_clopper", "M7_jeffreys"]
LAB = ["GO", "ADAPT-LOW", "ADAPT-UNCERTAIN", "STOP"]


def wilson_pn(p, n, z=Z80):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(np.maximum(p * (1 - p) / n + z * z / (4 * n * n), 0)) / d
    return c - h, c + h


def cp(k, n, a=0.10):
    lo = np.where(k == 0, 0.0, stats.beta.ppf(a, np.maximum(k, 1), n - k + 1))
    hi = np.where(k == n, 1.0, stats.beta.ppf(1 - a, k + 1, np.maximum(n - k, 1)))
    return lo, hi


def jeff(k, n, a=0.10):
    return stats.beta.ppf(a, k + 0.5, n - k + 0.5), stats.beta.ppf(1 - a, k + 0.5, n - k + 0.5)


def icc_est(K, M):
    N = M.sum(1); p = K.sum(1) / N
    ok = M > 0; ps = np.where(ok, K / np.where(ok, M, 1), 0.0)
    ssb = (M * (ps - p[:, None]) ** 2).sum(1)
    ssw = (M * ps * (1 - ps)).sum(1)
    s = ok.sum(1)
    msb = ssb / np.maximum(s - 1, 1); msw = ssw / np.maximum(N - s, 1)
    m0 = (N - (M ** 2).sum(1) / N) / np.maximum(s - 1, 1)
    den = msb + (m0 - 1) * msw
    r = np.where(den > 0, (msb - msw) / np.where(den > 0, den, 1), 0.0)
    return np.clip(r, 0.0, 1.0)


def intervals(K, M, rho, rng, B=500):
    R = K.shape[0]; N = M.sum(1).astype(float); Kt = K.sum(1).astype(float); ph = Kt / N
    out = {}
    out["M0_point"] = (ph, ph)
    L1, U1 = wilson_pn(ph, N); out["M1_wilson"] = (L1, U1)
    Ls, Us = [L1], [U1]
    for s in range(S):
        n2 = N - M[:, s]; k2 = Kt - K[:, s]; ok = n2 > 0
        l, u = wilson_pn(np.where(ok, k2 / np.where(ok, n2, 1), 0.5), np.where(ok, n2, 1.0))
        Ls.append(np.where(ok, l, L1)); Us.append(np.where(ok, u, U1))
    out["M2_registered"] = (np.minimum.reduce(Ls), np.maximum.reduce(Us))
    mbar = N / S
    re = icc_est(K, M); neff = N / (1 + (mbar - 1) * re); out["M3_deff_est"] = wilson_pn(ph, neff)
    neff_o = N / (1 + (mbar - 1) * rho); out["M3o_deff_true"] = wilson_pn(ph, neff_o)
    Lb = np.empty(R); Ub = np.empty(R); ch = 500
    for a in range(0, R, ch):
        b = min(R, a + ch); r = b - a
        idx = rng.integers(0, S, (r, B, S)); rows = np.arange(r)[:, None, None]
        kb = K[a:b][rows, idx].sum(2); nb = M[a:b][rows, idx].sum(2)
        pb = kb / np.maximum(nb, 1)
        q = np.percentile(pb, [10, 90], axis=1); Lb[a:b], Ub[a:b] = q[0], q[1]
    out["M4_cluster_boot"] = (Lb, Ub)
    ok = M > 0; ps = np.where(ok, K / np.where(ok, M, 1), np.nan)
    mu = np.nanmean(ps, 1); sd = np.nanstd(ps, 1, ddof=1); k_ = ok.sum(1)
    hw = T80 * sd / np.sqrt(np.maximum(k_, 1)); out["M5_school_t"] = (np.clip(mu - hw, 0, 1), np.clip(mu + hw, 0, 1))
    kk = Kt.astype(int); nn = N.astype(int)
    out["M6_clopper"] = cp(kk, nn); out["M7_jeffreys"] = jeff(kk, nn)
    return out


def decide(L, U):
    return np.select([L >= GO, U < STOP, U < GO], [0, 3, 1], default=2)  # 0 GO,1 LOW,2 UNC,3 STOP


def gen(R, M0, pi, rho, rng, informative=False):
    M = np.tile(np.asarray(M0), (R, 1))
    P = rng.beta(pi * (1 - rho) / rho, (1 - pi) * (1 - rho) / rho, (R, S)) if rho > 0 else np.full((R, S), pi)
    K = rng.binomial(M, P)
    if informative:
        K2 = rng.binomial(K, 0.95); F2 = rng.binomial(M - K, 0.70); K, M = K2, K2 + F2
    return K, M


def config(tag, M0, pi, rho, R, rng, informative=False):
    K, M = gen(R, M0, pi, rho, rng, informative)
    iv = intervals(K, M, rho, rng)
    rows = []
    for m in METHODS:
        L, U = iv[m]; d = decide(L, U)
        row = dict(scenario=tag, n=int(sum(M0)), balanced=int(len(set(M0)) == 1), informative_dropout=int(informative), icc=rho, true_p=pi, method=m, R=R)
        for j, l in enumerate(LAB): row["P_" + l] = round(float((d == j).mean()), 4)
        row["coverage"] = round(float(((L <= pi) & (pi <= U)).mean()), 4); row["mean_width"] = round(float((U - L).mean()), 4)
        row["se_PGO"] = round(math.sqrt(max(row["P_GO"] * (1 - row["P_GO"]), 1e-12) / R), 4)
        rows.append(row)
    return rows


def unit_tests():
    l, u = wilson_pn(np.array([0.5]), np.array([10.0]), 1.959964); assert abs(l[0] - 0.2366) < 5e-4 and abs(u[0] - 0.7634) < 5e-4
    l, u = cp(np.array([0]), np.array([10]), 0.025); assert abs(u[0] - 0.3085) < 5e-4 and l[0] == 0
    rng = np.random.default_rng(1); K, M = gen(20000, [3] * S, 0.6, 0.2, rng); r = icc_est(K, M).mean()
    assert abs(r - 0.2) < 0.03, r  # ANOVA estimator recovers the simulated ICC (small-sample bias allowed)
    return round(float(r), 3)


def exact_validation(R, rng):
    rows = []
    for n in (18, 27, 36):
        k = np.arange(n + 1); L, U = wilson_pn(k / n, np.full(n + 1, float(n)))
        for pi in (0.55, 0.60, 0.70, 0.85, 0.95):
            pm = stats.binom.pmf(k, n, pi); d = decide(L, U)
            ex = {l: float(pm[d == j].sum()) for j, l in enumerate(LAB)}
            cov = float(pm[(L <= pi) & (pi <= U)].sum())
            K, M = gen(R, [n // S] * S, pi, 0.0, rng); Ls, Us = wilson_pn(K.sum(1) / M.sum(1), M.sum(1).astype(float)); ds = decide(Ls, Us)
            sim = {l: float((ds == j).mean()) for j, l in enumerate(LAB)}
            se = math.sqrt(max(ex["GO"] * (1 - ex["GO"]), 1e-12) / R)
            rows.append(dict(n=n, true_p=pi, exact_PGO=round(ex["GO"], 4), sim_PGO=round(sim["GO"], 4), exact_PSTOP=round(ex["STOP"], 4), sim_PSTOP=round(sim["STOP"], 4),
                             exact_coverage=round(cov, 4), diff_PGO=round(abs(ex["GO"] - sim["GO"]), 4), z_PGO=round(abs(ex["GO"] - sim["GO"]) / se, 2), R=R))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=10000); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args(); t0 = time.time()
    icc_hat = unit_tests(); print("unit tests passed; ANOVA ICC recovered", icc_hat, file=sys.stderr, flush=True)
    rng = np.random.default_rng(20261126); rows = []
    for rho in (0.0, 0.1, 0.2, 0.3, 0.5):
        for n in (18, 27, 36):
            for pi in (0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
                rows += config("A_grid", [n // S] * S, pi, rho, a.R, rng)
        print("grid icc", rho, round(time.time() - t0), "s", file=sys.stderr, flush=True)
    for pi in (0.55, 0.60, 0.85, 0.95):
        rows += config("B_unbalanced_1to6", [1, 1, 2, 2, 3, 3, 4, 5, 6], pi, 0.2, a.R, rng)
        rows += config("C_informative_withdrawal", [3] * S, pi, 0.2, a.R, rng, informative=True)
    print("B,C done", round(time.time() - t0), "s", file=sys.stderr, flush=True)
    os.makedirs(a.out, exist_ok=True)
    def w(name, rr):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rr[0].keys())); wr.writeheader(); wr.writerows(rr)
    w("results_compare.csv", rows)
    seedrows = []
    for sd in range(a.seeds):
        r2 = np.random.default_rng(1000 + sd)
        for pi in (0.55, 0.60, 0.85, 0.95):
            for r in config("D_seed", [3] * S, pi, 0.2, a.R, r2): r["seed"] = 1000 + sd; seedrows.append(r)
    w("results_compare_seeds.csv", seedrows)
    print("seeds done", round(time.time() - t0), "s", file=sys.stderr, flush=True)
    w("validation_exact.csv", exact_validation(a.R, np.random.default_rng(777)))
    json.dump(dict(R=a.R, seeds=a.seeds, B=500, numpy=np.__version__, scipy=__import__("scipy").__version__, python=platform.python_version(),
                   icc_recovery=icc_hat, seconds=round(time.time() - t0), methods=METHODS), open(os.path.join(a.out, "run_meta_compare.json"), "w"), indent=1)
    print("wrote results_compare.csv, results_compare_seeds.csv, validation_exact.csv")


if __name__ == "__main__":
    main()
