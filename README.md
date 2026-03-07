# Integrating Machine Learning Demand Forecasting with the Guaranteed Service Model for Multi-Echelon Safety Stock Optimization

**A Computational Study on Retail Data**

---

## Abstract

This project implements a complete framework for multi-echelon inventory optimization that integrates machine learning demand forecasting with the Guaranteed Service Model (GSM). Using Walmart retail sales data (45 stores, 10 departments, 143 weeks), we construct a three-echelon supply chain network and demonstrate that using cross-validated ML forecast residuals as the demand uncertainty input to the GSM reduces total safety stock holding costs by **32.9%** compared to classical statistical approaches, while maintaining identical service levels. The framework is validated through Monte Carlo simulation (520,000 demand realizations per scenario), verified by an independent PuLP/CBC MILP solver (0.000% optimality gap), benchmarked against single-echelon optimization (58.3% coordination value), and proven robust to ±50% parameter perturbations.

## Key Results

| Metric | Value |
|--------|-------|
| ML cost reduction vs. classical | **32.9%** (stable across all service levels) |
| Monte Carlo service level validation | ±0.01pp gap under normal demand |
| MEIO vs. SEIO coordination value | **58.3%** cost savings |
| DP vs. MILP optimality gap | **0.000%** (verified exact) |
| Robustness to parameter changes | 32.9% ± 0.1% across ±50% lead time variations |

## Research Contributions

1. **ML-GSM Integration Framework**: Systematic pipeline connecting ML demand forecasting (LightGBM, XGBoost) with the Graves & Willems (2000) GSM, using cross-validated forecast residuals as the demand uncertainty parameter — addressing a gap identified by Eruguz et al. (2016) and Gonçalves et al. (2020).

2. **Temporal Cross-Validation for Sigma Estimation**: Expanding-window temporal CV (5 folds, 35,326 out-of-fold predictions) produces honest, non-overfitting estimates of irreducible forecast uncertainty — a methodological contribution applicable beyond this specific problem.

3. **Comprehensive Computational Validation**: Seven experiments covering accuracy comparison, service level sensitivity, echelon decomposition, model comparison, Monte Carlo simulation, SEIO baseline, and parameter sensitivity — with independent MILP verification of the DP solver.

4. **Real-Data Application**: End-to-end demonstration on Walmart's retail dataset mapping flat store-department sales data into a hierarchical supply chain network suitable for MEIO.

## Project Structure

```
project/
├── config.py                          # Global parameters (lead times, costs, etc.)
├── run_pipeline.py                    # End-to-end pipeline runner
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── data/                              # Input data (place Walmart CSVs here)
│   ├── train.csv                      #   421,570 rows: Store, Dept, Date, Weekly_Sales
│   ├── stores.csv                     #   45 rows: Store, Type, Size
│   ├── features.csv                   #   8,190 rows: economic & promotional features
│   └── test.csv                       #   (optional, not used by pipeline)
│
├── src/                               # Source modules (executed sequentially)
│   ├── __init__.py                    #   Package init, sets up sys.path
│   ├── 01_data_preprocessing.py       #   Stage 1: Load, merge, feature engineering
│   ├── 02_eda.py                      #   Stage 2: Demand distribution analysis
│   ├── 03_demand_forecasting.py       #   Stage 3: ML forecasting + temporal CV sigma
│   ├── 04_network_construction.py     #   Stage 4: Multi-echelon spanning tree builder
│   ├── 05_gsm_optimization.py         #   Stage 5: GSM solver (DP on tree)
│   ├── 06_experiments.py              #   Stage 6: All 7 computational experiments
│   ├── 07_visualization.py            #   Stage 7: Publication-quality figures
│   ├── 08_monte_carlo_simulation.py   #   Monte Carlo service level validation
│   ├── 09_seio_baseline.py            #   Single-echelon independent optimization
│   └── 10_sensitivity_analysis.py     #   Parameter sensitivity + PuLP MILP verification
│
├── results/                           # Output CSVs (auto-generated)
└── figures/                           # Output PNGs (auto-generated)
```

## Methodology

### Stage 1 — Data Preprocessing (`01_data_preprocessing.py`)

Loads three Walmart CSVs and merges them on Store/Date keys. Engineers 12 temporal features including cyclical sine/cosine encodings for week-of-year and month. Imputes missing MarkDown values with zero, forward-fills CPI and Unemployment within store groups. Maps store types (A, B, C) to regions (R1, R2, R3) as a proxy for geographic clustering — each region corresponds to one Distribution Center in the multi-echelon network. Selects the top 10 departments by sales volume, yielding 63,961 records across 450 store-department pairs over 143 weeks.

### Stage 2 — Exploratory Data Analysis (`02_eda.py`)

Characterizes demand distributions at three aggregation levels (store-department, store, region). Computes coefficient of variation (CV), skewness, and mean-variance power-law relationships. Visualizes the risk-pooling effect across echelons — CV decreases at higher aggregation, motivating the multi-echelon approach. Generates 4 diagnostic figures.

### Stage 3 — Demand Forecasting (`03_demand_forecasting.py`)

Implements three forecasting models using 31 features (temporal, store attributes, economic indicators, promotions, 5 lag features, 6 rolling statistics, expanding mean):

- **Historical Baseline**: Per-group historical mean (classical approach)
- **LightGBM**: Gradient boosted trees (best model, RMSE 4,608 vs baseline 7,337)
- **XGBoost**: Alternative gradient boosting (RMSE 4,703)

**Part A** fits final models and evaluates on a held-out 12-week test set (for accuracy metrics).

**Part B** runs expanding-window temporal cross-validation on the training set (for honest sigma estimation). Five CV folds produce 35,326 out-of-fold residuals — the std of these residuals per store-department is the `sigma_cv_residual` input to the GSM. This avoids the overfitting bias of in-sample residuals and the instability of short test-window residuals. LightGBM achieves 16.8% average sigma reduction across 60% of series.

### Stage 4 — Network Construction (`04_network_construction.py`)

Builds a 4-level spanning tree verified as `Is tree: True`:

```
[SUPPLY] → [DC_R1] → [WH_R1_A] → [S1_D92, S1_D95, ...]
         → [DC_R2] → [WH_R2_B] → [S5_D38, S5_D72, ...]
         → [DC_R3] → [WH_R3_C] → [S33_D40, ...]
```

- **SUPPLY** (virtual root): L=0, h=0, max_S=2 (external lead time)
- **3 DCs** (echelon 1): L=2 weeks, h=0.005/unit/week, max_S=3
- **3 Warehouses** (echelon 2): L=1 week, h=0.010/unit/week, max_S=2
- **448 Store-Depts** (echelon 3): L=1 week, h=0.020/unit/week, max_S=0 (immediate customer service)

Internal nodes receive aggregated demand via bottom-up risk pooling: `σ_parent = √(Σ σ²_child)`, assuming demand independence across children.

### Stage 5 — GSM Optimization (`05_gsm_optimization.py`)

Solves the Guaranteed Service Model as formulated by Graves & Willems (2000):

```
min  Σ_j  h_j · k · σ_j · √(τ_j)

s.t.  τ_j = SI_j + L_j - S_j     ∀j     (net replenishment time)
      τ_j ≥ 0                      ∀j     (feasibility)
      0 ≤ S_j ≤ S_j^max           ∀j     (service time bounds, integer)
      SI_j = S_parent(j)           ∀j     (service coupling on tree)
      SI_root = external_LT               (boundary condition)
```

**Decision variables**: Outbound service time S_j at each node (integer).

**Solution method**: Dynamic programming on the spanning tree. Bottom-up pass computes cost-to-go functions; top-down pass extracts optimal service times. Exact for tree networks — verified at 0.000% gap against independent PuLP/CBC MILP solver.

**Optimal solution structure**: S_SUPPLY=0, S_DC=0, S_WH=0, S_store=0, giving τ_DC=2, τ_WH=1, τ_store=1. Store-department nodes carry 77% of total safety stock and 91% of holding cost, consistent with real supply chain economics.

### Stage 6 — Computational Experiments (`06_experiments.py`)

Seven experiments, each producing CSV results and publication-quality figures:

| # | Experiment | Key Finding |
|---|-----------|-------------|
| 1 | Classical vs. ML-Enhanced GSM | 34.0% SS reduction, 32.9% cost reduction |
| 2 | Service Level Sensitivity (90–99%) | 32.9% cost reduction uniform across all SLs |
| 3 | Echelon Contribution Analysis | Stores: 77% SS / 91% cost; DCs: 13% SS / 4% cost |
| 4 | Forecasting Model Comparison | LightGBM best by RMSE; XGBoost best by sigma reduction |
| 5 | Monte Carlo Validation (10K trials) | Normal: ±0.01pp; Log-Normal: -0.3 to -0.9pp |
| 6 | SEIO Baseline Comparison | 58.3% value of multi-echelon coordination |
| 7 | Parameter Sensitivity + MILP Verification | 32.9% ± 0.1% robust; 0.000% DP-MILP gap |

### Stage 7 — Visualization (`07_visualization.py`)

Generates 8 publication-quality PNG figures:

1. `fig_network_topology.png` — Multi-echelon network schematic
2. `fig_cost_comparison.png` — Classical vs. ML-Enhanced bar chart
3. `fig_sensitivity_analysis.png` — Cost-service tradeoff curves
4. `fig_echelon_breakdown.png` — Stacked echelon contribution
5. `fig_model_comparison.png` — Forecast RMSE vs. inventory cost scatter
6. `fig_monte_carlo_validation.png` — Simulated fill rates vs. targets
7. `fig_seio_vs_meio.png` — SEIO vs. MEIO cost comparison
8. `fig_parameter_sensitivity.png` — Tornado chart + reduction robustness

## Installation & Usage

### Prerequisites

- Python 3.10+
- Walmart Sales Forecast dataset from [Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)

### Setup

```bash
# Clone or download this project
cd project/

# Create virtual environment (recommended)
python -m venv .env
source .env/bin/activate        # Linux/Mac
# or: .env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Download the Walmart dataset from Kaggle and place the CSV files in the `data/` directory:

```
data/
├── train.csv
├── stores.csv
├── features.csv
└── test.csv        (optional)
```

### Running the Full Pipeline

```bash
python run_pipeline.py
```

This executes all 7 stages sequentially (approximately 3 minutes on a modern machine). Results are saved to `results/` and figures to `figures/`.

### Running Individual Stages

Each module can be executed independently after its dependencies have been generated:

```bash
# Stage 1: Preprocessing (must run first)
python src/01_data_preprocessing.py

# Stage 2: EDA (requires: results/processed_data.csv)
python src/02_eda.py

# Stage 3: Forecasting (requires: results/processed_data.csv)
python src/03_demand_forecasting.py
```

Stages 4–7 are typically run through `06_experiments.py` via the pipeline, as they require the forecast results dict passed in memory.

### Configuration

All parameters are centralized in `config.py` at the project root:

```python
# Key parameters you may want to adjust:
LEAD_TIMES = {"dc": 2, "warehouse": 1, "store": 1}       # weeks
HOLDING_COST_RATES = {"dc": 0.005, "warehouse": 0.010, "store": 0.020}
MAX_SERVICE_TIMES = {"dc": 3, "warehouse": 2, "store": 0}
EXTERNAL_LEAD_TIME = 2                                      # weeks
DEFAULT_SERVICE_LEVEL = 0.95
TOP_N_DEPARTMENTS = 10
CV_FOLDS = 5
CV_MIN_TRAIN_WEEKS = 40
```

## Output Files

### Results (`results/`)

- `processed_data.csv` — Merged and feature-engineered dataset
- `forecast_metrics.csv` — Model accuracy comparison table
- `residual_stats_*.csv` — Per-model sigma estimates for GSM
- `oof_residuals_*.csv` — Out-of-fold CV residuals (reproducibility)
- `gsm_results_*.csv` — GSM optimization results per experiment
- `experiment1_comparison.csv` through `experiment7*.csv` — Experiment outputs

### Figures (`figures/`)

Eight publication-quality PNG figures at 300 DPI, suitable for direct inclusion in journal manuscripts.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | Numerical computation |
| pandas | ≥2.0.0 | Data manipulation |
| scikit-learn | ≥1.3.0 | Train-test splitting, metrics |
| lightgbm | ≥4.0.0 | Gradient boosted tree forecasting |
| xgboost | ≥2.0.0 | Alternative gradient boosting |
| scipy | ≥1.11.0 | Statistical functions |
| matplotlib | ≥3.7.0 | Figure generation |
| seaborn | ≥0.12.0 | Statistical visualization |
| networkx | ≥3.1 | Graph/tree operations |
| pulp | ≥2.7.0 | MILP solver for GSM verification |
| tabulate | ≥0.9.0 | Table formatting |

## References

- Graves, S.C. & Willems, S.P. (2000). "Optimizing Strategic Safety Stock Placement in Supply Chains." *Manufacturing & Service Operations Management*, 2(1), 68–83.
- Eruguz, A.S., Sahin, E., Jemai, Z., & Dallery, Y. (2016). "A comprehensive survey of guaranteed-service models for multi-echelon inventory optimization." *International Journal of Production Economics*, 172, 110–125.
- Gonçalves, J.N.C., Sameiro Carvalho, M., & Cortez, P. (2020). "Operations research models and methods for safety stock determination: A review." *Operations Research Perspectives*, 7, 100164.
- Simpson, K.F. (1958). "In-Process Inventories." *Operations Research*, 6(6), 863–873.
- Magnanti, T.L., Shen, Z.J.M., Shu, J., Simchi-Levi, D., & Teo, C.P. (2006). "Inventory placement in acyclic supply chain networks." *Operations Research Letters*, 34(2), 228–238.
- Achkar, V.G. et al. (2024). "Extensions to the guaranteed service model for industrial applications of multi-echelon inventory optimization." *European Journal of Operational Research*, 313(1), 192–206.
- de Kok, T. et al. (2018). "A typology and literature review on stochastic multi-echelon inventory models." *European Journal of Operational Research*, 269(3), 955–983.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed., OTexts.

## License

Academic and research use. Dataset subject to [Kaggle terms of use](https://www.kaggle.com/terms).