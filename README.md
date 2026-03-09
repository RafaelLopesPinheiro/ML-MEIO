# Inventory CVaR Optimisation (OR-Focused)

Data-driven newsvendor inventory management under uncertainty, comparing
distribution-free, robust, and decision-focused optimisation methods.

## Methods (7, all RF-based for fair comparison)

| # | Method | Key Reference |
|---|--------|---------------|
| 0 | (s,S) Policy | Classical reorder-point benchmark |
| 1 | SAA | Ban & Rudin (2019) |
| 2 | Conformal + CVaR | Vovk et al. (2005); Rockafellar & Uryasev (2000) |
| 3 | Wasserstein DRO | Mohajerin Esfahani & Kuhn (2018) |
| 4 | EnbPI+CQR+CVaR | Xu & Xie (2021); Romano et al. (2019) — **proposed** |
| 5 | SPO (RF) | Elmachtoub & Grigas (2017) |
| 6 | Seer | Oracle upper bound |

## OR Contributions

- **MPC lookahead**: Multi-period CVaR LP with carryover coupling
- **Value decomposition**: Additive breakdown of cost gap into forecasting / UQ / risk components
- **Dual-bound analysis**: LP dual for optimality-gap measurement
- **Conformal chance constraint**: Finite-sample SL guarantee via CQR upper bound
- **Pareto frontier**: Cost vs service-level trade-off analysis

## Quick Start

```bash
pip install -r requirements.txt
python tests/test_smoke.py          # verify setup
python scripts/run_experiment.py    # main experiment (myopic)
python scripts/run_experiment.py --mpc-horizon 7  # with MPC lookahead
python scripts/run_sensitivity.py   # parameter sensitivity
```

## Project Structure

```
configs/          Configuration dataclasses
src/
  data/           Data loading, feature engineering, temporal splits
  models/         RF-based forecasting with uncertainty quantification
  optimization/   CVaR / DRO / MPC LP formulations (PuLP)
  evaluation/     Metrics, statistical tests, value decomposition
scripts/
  run_experiment.py    Main expanding-window experiment
  run_sensitivity.py   β × α × cost-ratio sweep
tests/
  test_smoke.py        Quick verification
```
