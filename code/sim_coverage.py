"""Coverage of the teacher-level Wilson interval (R1) and the leave-one-school-out widened interval (R2)
for a proportion when teachers are nested in 9 schools (beta-binomial clustering). Simulation only.
Usage: python -I -X utf8 sim_coverage.py [--R 10000] [--seed 20261124] [--out <dir>]
"""
import argparse, csv, math, os
import numpy as np

Z = 1.959964
S = 9


def wilson(x, n):
    p = x / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for rho in (0.0, 0.1, 0.2, 0.3):
        for n in (18, 27, 36):
            t = n // S
            schools = np.repeat(np.arange(S), t)
            for pi in (0.6, 0.8, 0.9):
                c1 = c2 = 0
                w1 = w2 = 0.0
                for _ in range(a.R):
                    if rho > 0:
                        ps = rng.beta(pi * (1 - rho) / rho, (1 - pi) * (1 - rho) / rho, S)
                    else:
                        ps = np.full(S, pi)
                    x = (rng.random(n) < ps[schools]).astype(int)
                    L1, U1 = wilson(int(x.sum()), n)
                    Ls, Us = [L1], [U1]
                    for s in range(S):
                        m = schools != s
                        l, u = wilson(int(x[m].sum()), int(m.sum()))
                        Ls.append(l); Us.append(u)
                    L2, U2 = min(Ls), max(Us)
                    c1 += (L1 <= pi <= U1); c2 += (L2 <= pi <= U2)
                    w1 += U1 - L1; w2 += U2 - L2
                rows.append(dict(icc=rho, n=n, true_p=pi, cover_wilson=round(c1 / a.R, 4), cover_widened=round(c2 / a.R, 4),
                                 width_wilson=round(w1 / a.R, 3), width_widened=round(w2 / a.R, 3), R=a.R))
    p = os.path.join(a.out, "results_coverage.csv")
    with open(p, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", p)


if __name__ == "__main__":
    main()
