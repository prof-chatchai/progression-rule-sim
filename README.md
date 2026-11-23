# Simulation code: go, adapt or stop progression rule

Code and results for the simulations in the manuscript on pre-specified progression criteria for a small clustered feasibility study (27 teachers in nine schools). **Simulated data only; no human data.**

| Script (`code/`) | Purpose | Output (`results/`) |
|---|---|---|
| `sim_final.py` | registered rule C5: decision probabilities, withdrawal, coverage of 80% intervals | `results_final_decisions.csv`, `results_final_coverage80.csv`, `run_meta_final.json` |
| `sim_candidates.py` | eight rule variants compared before registration | `results_candidates.csv` |
| `sim_compare.py` | rule vs eight standard interval methods, exact validation, seed stability | `results_compare*.csv`, `validation_exact.csv` |
| `sim_itt.py` | withdrawal policy: exclusion vs intention to treat | `results_itt.csv` |
| `sim_coverage.py`, `sim_algorithm1.py` | earlier coverage and algorithm runs | `results_coverage.csv`, `results_algorithm1.csv` |

## Run
```
pip install -r requirements.txt     # Python 3.11, NumPy 2.4.6
cd code
python -I -X utf8 sim_final.py --out ../rerun      # about 5 minutes; default seed 20261125
```
Seeds, run sizes and versions are in `results/run_meta_*.json`. See `CHECK_REPRODUCTION.md` for what was and was not re-run.

## Caveats
Assumed data-generating model (beta-binomial school effects, lognormal preparation time, independent indicators). Probabilities describe the rules under these assumptions and are not forecasts for the field study. Not independently reproduced.
