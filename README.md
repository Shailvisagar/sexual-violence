# Sexual Violence Against Women (SVAW) — ML Classification (NCRB Table 3A.3)

> **Purpose:** From NCRB Table 3A.3 (Victims of rape by age group, by State/UT; single year), build a *transparent* ML classifier that flags **high-burden** State/UTs (top 50% by cases reported) **based on the age-distribution of victims**.  
> **Scope Note:** This repo **does not** attempt individual-level risk prediction. It uses *aggregate* public statistics only.

## Data
- Source file: `data/raw/NCRB_Table_3A.3.csv` (your uploaded CSV).  
- Columns include: `Cases Reported`, victim age buckets for child and adult groups, and `Total Victims` by State/UT.

## Target & Features
- **Target**: `label_high_cases` = 1 if `Cases Reported` >= dataset median (high-burden), else 0.
- **Features**: Proportions of victims in each age bucket (shares relative to `Total Victims`). This reduces target leakage from absolute totals.

## Quickstart

```bash
# 1) Create & activate env (Python >=3.10)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the full pipeline
bash scripts/run_all.sh

# 4) See outputs
ls models/
ls reports/
```

## Ethics & Responsible Use
- Aggregate-only, no personal data, no policing or profiling.  
- Results are descriptive and **not** causal; they should guide **resource allocation** and **support services**, not punitive actions.  
- Check for biases (e.g., reporting differences, population size). Consider normalizing by population if available.

## Repo Structure
```
svaw-ml/
  data/
    raw/           # original CSV (NCRB_Table_3A.3.csv)
    processed/     # features + labels
  models/          # trained models and artifacts
  reports/         # figures and evaluation
  src/             # modular python source
  scripts/         # run scripts
  README.md
  requirements.txt
  .gitignore
  LICENSE
```

## Results (from this run)
- Logistic Regression: accuracy = 0.664 ± 0.119, AUC = 0.763 ± 0.095
- Random Forest: accuracy = 0.718 ± 0.096, AUC = 0.821 ± 0.126

See `reports/top10_cases.png` for a quick view of top-10 states/UTs by cases.

## Next Steps
- Add population normalization (cases per 100k) if you have population by State/UT.  
- Try gradient boosting (e.g., XGBoost/LightGBM) and calibration plots.  
- Expand to multi-year NCRB tables to move from classification to time-series forecasting.
