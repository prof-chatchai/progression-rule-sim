"""Withdrawal handling: exclude withdrawn teachers from the denominator vs count them as non-delivering (intention to treat).
Same data-generating model as sim_compare.py (imported). Delivery indicator only; methods M2 (registered rule) and M3 (design-effect Wilson, estimated ICC).
Scenarios: informative withdrawal (successes withdraw 5%, failures 30%); random withdrawal 20%.
Usage: python -I -X utf8 sim_itt.py [--R 10000]"""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_compare as sc

LAB = sc.LAB


def run(R, seed):
    rng = np.random.default_rng(seed); rows = []
    for tag, mode in (("informative_exclude", "inf_ex"), ("informative_ITT", "inf_itt"), ("random20_exclude", "r_ex"), ("random20_ITT", "r_itt")):
        for pi in (0.55, 0.60, 0.85, 0.95):
            M0 = np.tile(np.full(sc.S, 3), (R, 1))
            P = rng.beta(pi * 0.8 / 0.2, (1 - pi) * 0.8 / 0.2, (R, sc.S))
            K = rng.binomial(M0, P)
            if mode.startswith("inf"):
                K2 = rng.binomial(K, 0.95); F2 = rng.binomial(M0 - K, 0.70)
            else:
                K2 = rng.binomial(K, 0.80); F2 = rng.binomial(M0 - K, 0.80)
            if mode.endswith("ex"): Kf, Mf = K2, K2 + F2     # withdrawn teachers excluded from the denominator
            else: Kf, Mf = K2, M0                              # withdrawn teachers counted as not delivering
            iv = sc.intervals(Kf, Mf, 0.2, rng)
            for m in ("M2_registered", "M3_deff_est"):
                L, U = iv[m]; d = sc.decide(L, U)
                row = dict(scenario=tag, true_p=pi, method=m, R=R)
                for j, l in enumerate(LAB): row["P_" + l] = round(float((d == j).mean()), 4)
                rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--R", type=int, default=10000); a = ap.parse_args()
    rows = run(a.R, 20261127)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_itt.csv"), "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows: print(r["scenario"], r["true_p"], r["method"], "GO", r["P_GO"], "STOP", r["P_STOP"])
