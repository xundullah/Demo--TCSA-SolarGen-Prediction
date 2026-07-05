# MLS-DL Multilayer Stacked Deep Learning with Temporal Context Self Attention 

**PV Power-Generation Prediction from the solar plants** — a research pipeline and interactive
simulator for forecasting solar-plant power output from historical inverter
and weather-station data.

The project builds and evaluates eight deep-learning forecasting models
(LSTM / GRU / Transformer, each in a **baseline**, **MHSA** (multi-head
self-attention), and **TCSA** (proposed) variant) that take 168 h (7-day)
input windows and predict the next 24 h of PV generation, across three solar
plants at Site-02 (Gyeongju-si, Gyeongsangbuk-do, South Korea).

Hight light this https://xundullah.github.io/Demo--TCSA-SolarGen-Prediction/ this is for demo and simulation 

## Repository layout

```
DataBase/            Notebooks that pull raw data from the cloud DB and turn it
                      into clean, per-site CSV/gzip datasets
  1. BulkDataCollection.ipynb   Connect to the plant operator's MariaDB and export raw tables
  2. BlukDataProcess.ipynb      Bulk-process all plants, select relevant columns/sites
  3. Site01DataProcess.ipynb    Site #01 (Ansan-si) — 2 solar plants
  4. Site02DataProcess.ipynb    Site #02 (Gyeongju-si) — 9 solar plants

Code/                Main research pipeline notebooks (Site-02 study)
  Site02_DataPreparation.ipynb  Load, clean, and window the Site-02 dataset
  Site02_ModelDevelopment.ipynb Train all 8 models x 3 plants (TensorFlow/Keras)
  Site02_ResultAnalysis.ipynb   Evaluate models and generate publication figures

Library/             Reusable Python modules shared by the notebooks
  dataProcessing.py    PV/weather NaN-filling (climatology + diurnal bootstrap), plotting
  dataAnalysis.py      IEEE-style diurnal-pattern figures, spike removal
  modelDevelopment.py  Windowing, train/val split, model evaluation, loss-curve plots
  modelEvaluation.py   Full evaluation-figure suite (learning curves, forecast comparison,
                       daily profiles, TCSA-improvement summary) + metrics CSV export

Export/              Generated artifacts (figures, training history, processed data)
Sim/                 Exported prediction/metric CSVs consumed by the web simulator
Backups/              Snapshots of earlier code/figure/model/web versions
index.html           SPFS — Solar-Plant Forecasting Simulator (Site-02), a
                     browser-based dashboard that replays predictions vs. actuals
```

## Data pipeline

1. **Collection** (`DataBase/1. BulkDataCollection.ipynb`) — pulls PV inverter
   and weather-station readings for all monitored solar plants from the
   operator's MariaDB via SQLAlchemy.
2. **Bulk processing** (`DataBase/2. BlukDataProcess.ipynb`) — filters to the
   plants of interest and saves per-plant CSVs.
3. **Per-site processing** (`DataBase/3.` / `4.`) — builds the final hourly
   dataset for Site #01 (Ansan-si, 2 plants) and Site #02 (Gyeongju-si, 9
   plants → 3 plants used in this study).
4. **Data preparation** (`Code/Site02_DataPreparation.ipynb`) — cleans
   outliers/gaps using `Library/dataProcessing.py` (harmonic-climatology fill
   for weather columns, diurnal-bootstrap fill for PV columns so nighttime
   zeros and the sunrise/sunset ramp are preserved) and builds the analysis
   figures via `Library/dataAnalysis.py`.

## Model development

`Code/Site02_ModelDevelopment.ipynb` trains, per solar plant, eight
forecasting models sharing a 168 h → 24 h windowing scheme
(`Library/modelDevelopment.make_windows`):

| Family      | Baseline | Typical (MHSA) | Proposed (TCSA) |
|-------------|----------|----------------|------------------|
| LSTM        | ✓        | ✓              | ✓                |
| GRU         | ✓        | ✓              | ✓                |
| Transformer | —        | ✓              | ✓                |

Training history and validation loss curves are saved through
`modelDevelopment.save_history` / `plot_training_validation_loss`.

## Result analysis

`Code/Site02_ResultAnalysis.ipynb` uses `Library/modelEvaluation.py` to
produce the IEEE-styled figures used in the study, all colorblind-safe
(Okabe-Ito palette) and exported as vector PDF + 600 dpi PNG:

- **Learning-curve grid** — 3 plants x 8 models training/validation MSE.
- **Prediction-vs-actual comparison** — 24 h forecasts with zoomed daylight
  insets and a per-panel RMSE/MAE/skill-score table; exports all metrics to CSV.
- **Daily-profile comparison** — 0–24 h mean/min/max power bands plus
  per-hour R2/rRMSE.
- **Metric summary** — grouped bars showing the TCSA models' improvement over
  their family baseline/reference.

Exported metrics and predictions land in `Export/Data/Site-02/` and
`Sim/Site-02/` (e.g. `AllPlants-Metrics.csv`, `AllPlants-TCSA-Improvement.csv`,
`SolarPlant{1,2,3}_Predictions.csv`).

## Web simulator (`index.html`)

A self-contained, single-page dashboard ("SPFS — Solar-Plant Forecasting
Simulator") that loads the CSVs in `Sim/Site-02/` and replays each plant's
predicted vs. actual generation over time, with per-plant selection and
playback speed/timeline controls. Open `index.html` directly in a browser —
no build step or server required.

## Requirements

The notebooks are built around a **TensorFlow/Keras** deep-learning stack:

- Python 3.9.13 for `Code/*.ipynb` (TensorFlow-GPU < 2.10, tensorflow-addons 0.19.0)
- Python 3.11 for `DataBase/*.ipynb`
- pandas, numpy, matplotlib, scikit-learn, SQLAlchemy

## Backups

`Backups/` retains dated snapshots of earlier code, figures, models, and the
web dashboard, kept for traceability as the study evolved; the current,
canonical versions live under `Code/`, `Library/`, `Export/`, and `index.html`.
