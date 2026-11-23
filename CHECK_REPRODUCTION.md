# Reproduction check (2026-11-24)
- `sim_final.py` re-run in a fresh copy of this folder (`python -I`, seed 20261125, R = 1000, B = 300, NumPy 2.4.6, Python 3.11.15): `results_final_decisions.csv` and `results_final_coverage80.csv` are byte-identical to the shipped files (see below if not).
- Not re-run: `sim_candidates.py`, `sim_compare.py`, `sim_itt.py`, `sim_coverage.py`, `sim_algorithm1.py` (their shipped outputs were produced by the original runs; seeds are in `results/run_meta_*.json`).
- The run by the same authors on one machine is not an independent reproduction.
