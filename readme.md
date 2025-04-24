
# MLflow End-to-End Lifecycle Demo

This repo demonstrates **complete machine-learning lifecycle management with MLflow** on two independent problems:

1. **Breast‑Cancer Malignancy Detection**  (`scripts/cancer-detection/`)
2. **Baghdad Real‑Estate Rent Prediction**  (`scripts/house-price-prediction/` + `data/mansour-realestate.csv`)

Both pipelines cover the five core objectives required by the term‑project brief:

| # | Objective | Where implemented |
|---|-----------|-------------------|
| 1 | **Experiment Tracking** | All runs recorded in MLflow (`mlruns/`); separate experiment per task |
| 2 | **Model Training & Tuning** | Baseline + Hyperopt tuning scripts for each domain |
| 3 | **Model Deployment** | `mlflow models serve` exposes REST endpoints |
| 4 | **Performance Monitoring** | Live‑prediction scripts log accuracy / RMSE back into MLflow |
| 5 | **Model Registry** | Best models registered & promoted (Staging → Production) |

---

## 1  Project structure
```text
├── data/
│   └── mansour-realestate.csv
├── mlruns/                 # auto‑generated MLflow tracking data
├── scripts/
│   ├── cancer-detection/
│   │   ├── baseline_run.py
│   │   ├── tune_rf.py
│   │   ├── monitor_model.py
│   │   └── register_model.py
│   └── house-price-prediction/
│       ├── realestate_baseline.py
│       ├── tune_realestate_rf.py
│       ├── monitor_realestate.py
│       └── register_realestate.py
└── README.md
```

---

## 2  Quick start
```bash
python -m venv mlenv
# Windows: .\mlenv\Scripts\activate
source mlenv/bin/activate
pip install -r requirements.txt
mlflow ui --port 5000        # open http://127.0.0.1:5000
```

---

## 3  Breast‑Cancer pipeline
```bash
# 3.1 baseline
python scripts/cancer-detection/baseline_run.py

# 3.2 tuning (Hyperopt)
python scripts/cancer-detection/tune_rf.py

# 3.3 serve best run
mlflow models serve -m runs:/<best_run_id>/model -p 1234 --env-manager=local

# 3.4 monitoring
python scripts/cancer-detection/monitor_model.py

# 3.5 register & promote
python scripts/cancer-detection/register_model.py
```

---

## 4  Real‑Estate pipeline
*Dataset: `data/mansour-realestate.csv`, target `final_rent_price_usd`*

```bash
# 4.1 baseline
python scripts/house-price-prediction/realestate_baseline.py

# 4.2 tuning
python scripts/house-price-prediction/tune_realestate_rf.py

# 4.3 serve best run
mlflow models serve -m mlruns/<exp_id>/<best_run_id>/artifacts/model -p 1235 --env-manager=local

# 4.4 monitoring
python scripts/house-price-prediction/monitor_realestate.py

# 4.5 register & promote
python scripts/house-price-prediction/register_realestate.py
```

Input for the serving endpoint must be JSON in **`dataframe_records`** format (column names included).

---

## 5  Handy MLflow snippets
```bash
# list experiments
mlflow experiments list

# start UI
mlflow ui --port 5000

# promote via Python API
python - <<'PY'
from mlflow.tracking import MlflowClient
MlflowClient().transition_model_version_stage(
    name="realestate_rf_model", version=1, stage="Production")
PY
```

---

## 6  Troubleshooting
* **Missing columns error** → ensure you send `dataframe_records` with all feature names.  
* **pyenv not found (Windows)** → always add `--env-manager=local` when serving.  
* **Port already in use** → choose a different port (`1234`, `1235`, …).

---

Enjoy exploring the full MLflow lifecycle! 🚀
