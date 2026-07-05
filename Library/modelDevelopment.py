"""
modelDevelopment.py

Utility functions for the IGPS project (model development).

Usage:
    from Library.modelDevelopment import make_windows, split_train_val

    X, y = make_windows(df, input_len=168, output_len=24, target_col="P_kW", stride=1)
    X_train, y_train, X_val, y_val = split_train_val(X, y, val_frac=0.20, gap=168+24)
    evaluation_results, predicted_values = evaluate_model(model_name, model_prediction, target_scaler, X_val, y_val)
    plot_training_validation_loss(history, model_Name, patience, site_info='Site-02', save_fig=True, column='single')
"""








"""
Turn an hourly time-series DataFrame into supervised forecasting samples.

Each sample uses `input_len` hours of ALL feature columns to predict the
next `output_len` hours of `target_col`.

    X[k] = features[i : i+input_len]                 -> shape (input_len, n_features)
    y[k] = target[i+input_len : i+input_len+output_len] -> shape (output_len,)

Parameters
----------
df         : DataFrame with a DatetimeIndex, sorted ascending, no gaps.
input_len  : look-back window length in hours (default 168 = 7 days).
output_len : forecast horizon in hours (default 24 = 1 day).
target_col : column to predict (default "P_kW").
stride     : hours to slide between consecutive samples.
                1  -> a new sample every hour (max data, overlapping)
                24 -> one sample per day (no overlap)

Returns
-------
X : float32 array, shape (n_samples, input_len, n_features)
y : float32 array, shape (n_samples, output_len)
"""

import numpy as np
import pandas as pd

def make_windows(df, input_len=168, output_len=24, target_col="P_kW", stride=1):
    
    values = df.values.astype(np.float32)                 # (N, n_features)
    target = values[:, df.columns.get_loc(target_col)]    # (N,)
    N, n_feat = values.shape
    total = input_len + output_len

    starts = range(0, N - total + 1, stride)
    n = len(starts)
    X = np.empty((n, input_len, n_feat), dtype=np.float32)
    y = np.empty((n, output_len),        dtype=np.float32)
    for k, i in enumerate(starts):
        X[k] = values[i : i + input_len]
        y[k] = target[i + input_len : i + input_len + output_len]
    return X, y




"""
Split features and targets into training and validation sets.

Parameters
----------
X        : array of shape (n_samples, input_len, n_features)
y        : array of shape (n_samples, output_len)
val_frac : fraction of samples to use for validation (default 0.20)
gap      : hours to leave between training and validation sets to prevent data leakage (default 168+24 = 7 days + 1 day)

Returns
-------
X_train : array of shape (n_train_samples, input_len, n_features)
y_train : array of shape (n_train_samples, output_len)
X_val   : array of shape (n_val_samples, input_len, n_features)
y_val   : array of shape (n_val_samples, output_len)
"""
def split_train_val(X, y, val_frac=0.20, gap=168+24):
    n_val = int(len(X) * val_frac)
    cut = len(X) - n_val

    X_train, y_train = X[: cut - gap], y[: cut - gap]
    X_val,   y_val   = X[cut:],        y[cut:]
    return X_train, y_train, X_val, y_val












"""
Evaluates a model and returns performance metrics (in original kW units)
plus predicted values in original scale.

Version: 3.0
Date   : 2026.06.16

Args:
    model_name (str)   : Name or identifier of the model.
    model_prediction   : Trained model for prediction.
    target_scaler      : MinMaxScaler fit on the full (9-feature) frame.
    input              : X Input data, shape (n, 168, 7).
    target             : y Target, shape (n, 24), scaled.
    p_idx (int)        : Column index of P_kW in the scaler's fitted frame
                            (0 = P_PV1_kW).

Returns:
    evaluation_results (pd.DataFrame) : Evaluation metrics.
    predicted_values (pd.DataFrame)   : Predictions in original kW scale.
"""
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from math import sqrt
def evaluate_model(model_name, model_prediction, target_scaler, input, target, p_idx=0):
    
    # Predict (scaled), shape (n, 24)
    y_pred = model_prediction.predict(input)

    # Pull P_kW's own min/max out of the multi-feature scaler and invert manually
    pmin = target_scaler.data_min_[p_idx]
    pmax = target_scaler.data_max_[p_idx]
    y_pred_orig = y_pred * (pmax - pmin) + pmin
    y_test_orig = np.asarray(target) * (pmax - pmin) + pmin

    # Predictions back in original scale (each column = a sample, as before)
    predicted_values = pd.DataFrame(y_pred_orig).transpose()

    # Metrics in real kW — note correct (y_true, y_pred) argument order
    yt = y_test_orig.ravel()
    yp = y_pred_orig.ravel()
    mae  = mean_absolute_error(yt, yp)
    mse  = mean_squared_error(yt, yp)
    rmse = sqrt(mse)
    r2   = r2_score(yt, yp)

    evaluation_data = {
        'Model' : [model_name],
        'MAE'   : [mae],
        'MSE'   : [mse],
        'RMSE'  : [rmse],
        'R2'    : [r2]
    }
    evaluation_results = pd.DataFrame(evaluation_data)

    return evaluation_results, predicted_values







"""
Save and load Keras History objects to disk, keyed by model name, in both .pkl and .json formats.
"""

import os
import json
import pickle
import matplotlib.pyplot as plt
from matplotlib import font_manager


def save_history(history, model_Name, site_info='Site-02'):
    """
    Save a Keras History object's data to disk, keyed by model_Name.
    Writes both a .pkl (full fidelity) and a .json (human-readable).
    Accepts either a History object or a plain history dict.
    """
    hist_dict = history.history if hasattr(history, 'history') else history

    history_directory = f"../Export/History/{site_info}/"
    os.makedirs(history_directory, exist_ok=True)
    base = os.path.join(history_directory, f"history-{model_Name}")

    # Pickle: preserves exact Python objects
    with open(f"{base}.pkl", 'wb') as f:
        pickle.dump(hist_dict, f)

    # JSON: portable, readable; cast values to plain floats
    json_safe = {k: [float(v) for v in vals] for k, vals in hist_dict.items()}
    with open(f"{base}.json", 'w') as f:
        json.dump(json_safe, f, indent=2)

    print(f"History saved to:\n  {base}.pkl\n  {base}.json")
    return base


def load_history(model_Name, site_info='Site-02', fmt='pkl'):
    """
    Reload a saved history dict by model_Name.
    fmt: 'pkl' (default) or 'json'.
    Returns the history dict, which can be passed straight into
    plot_training_validation_loss as a stand-in for history.history.
    """
    base = f"../Export/History/{site_info}/history-{model_Name}"
    if fmt == 'pkl':
        with open(f"{base}.pkl", 'rb') as f:
            return pickle.load(f)
    elif fmt == 'json':
        with open(f"{base}.json", 'r') as f:
            return json.load(f)
    else:
        raise ValueError("fmt must be 'pkl' or 'json'")





"""
Plot training/validation loss over the full epoch range in IEEE figure style.

Version:
    3.0 (2026.06.16) - IEEE publication styling

Arguments:
history: keras.callbacks.History
    Returned by fit(); .history holds per-epoch metrics.
model_Name: str
    Model name (used in title and filename).
patience: int
    Used to locate the epoch where the best model was saved.
site_info: str
    Subfolder name for saving.
save_fig: bool
    Whether to save the figure.
save_hist: bool
    Whether to save the history dict (calls save_history).
column: str
    'single' (3.5 in wide) or 'double' (7.16 in wide) IEEE column.

Returns:
None
"""

import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

def plot_training_validation_loss(history, model_Name, patience, site_info='Site-02',
                                  save_fig=True, save_hist=True, column='single'):
    

    # Accept either a History object or a plain dict of metrics.
    hist = history.history if hasattr(history, 'history') else history

    # ---- Save the history first (so it's preserved even if plotting fails) ----
    if save_hist:
        save_history(hist, model_Name, site_info=site_info)
    
    # ---- IEEE rcParams ----
    # Prefer Times New Roman; fall back to a serif if unavailable.
    serif_candidates = ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif']
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in serif_candidates if f in available), 'serif')

    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [chosen],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.linewidth': 0.6,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'lines.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.constrained_layout.use': True,
    })

    # ---- Figure dimensions (inches) ----
    width = 3.5 if column == 'single' else 7.16
    fig_size = (width, width * 0.72)
    fig_dpi = 150
    save_fig_dpi = 600  # print-quality DPI for saved figure

    # ---- Epoch bookkeeping ----
    total_epochs = len(hist['loss'])
    considered_epochs = total_epochs - patience if patience is not None else total_epochs
    best_epoch = considered_epochs - 1

    plot_range = list(range(total_epochs))

    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)

    # Distinct line styles so the curves separate in grayscale too.
    ax.plot(plot_range, hist['loss'],
            label='Training MSE', color='black', linestyle='-')
    ax.plot(plot_range, hist['val_loss'],
            label='Validation MSE', color='dimgray', linestyle='--')

    # Best model marker
    ax.plot(best_epoch, hist['loss'][best_epoch],
            label='Best Model', color='black', linestyle='',
            marker='o', markersize=4, markerfacecolor='white',
            markeredgewidth=0.8, zorder=5)

    # Stop-saving reference line
    if patience is not None and patience <= total_epochs:
        ax.axvline(x=best_epoch, linestyle=':', color='black',
                   linewidth=0.8, label='Stop Saving Models')

    # ---- Labels and title ----
    ax.set_xlabel('Number of Training Epochs')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(f'Learning Curve of {model_Name}')

    # ---- Ticks ----
    ax.tick_params(axis='both', which='major', length=3.5, width=0.6)
    ax.tick_params(axis='both', which='minor', length=2.0, width=0.5)
    ax.minorticks_on()

    # ---- Light grid (IEEE figures are often clean; keep it subtle) ----
    ax.grid(True, which='major', linestyle=':', linewidth=0.4,
            color='gray', alpha=0.4)

    # ---- Legend: framed, thin border (IEEE convention) ----
    leg = ax.legend(loc='upper right', frameon=True, fancybox=False,
                    edgecolor='black', framealpha=1.0)
    leg.get_frame().set_linewidth(0.6)

    # ---- Save (vector + raster) ----
    if save_fig:
        figure_directory = f"../Export/Figure/{site_info}/"
        os.makedirs(figure_directory, exist_ok=True)
        base = os.path.join(figure_directory, f"LearningCurve-{model_Name}")
        fig.savefig(f"{base}.pdf", bbox_inches='tight')          # vector for LaTeX
        fig.savefig(f"{base}.png", dpi=save_fig_dpi, bbox_inches='tight')

    plt.show()