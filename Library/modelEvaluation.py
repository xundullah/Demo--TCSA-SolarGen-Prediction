"""modelEvaluation.py - IEEE-style evaluation figures for PV forecasting.

OBJECTIVE
    One library that produces every evaluation figure of the study
    (eight forecasting models x three solar plants) with a consistent,
    colorblind-safe (Okabe-Ito) IEEE look, and exports every number shown
    in a figure to CSV so tables can be built without re-running models.

MODEL TAXONOMY
    Families : LSTM, GRU, Transformer.
    Roles    : baseline (basic), MHSA (typical), TCSA (proposed).
    The Transformer family has no basic baseline, so its reference for
    skill/improvement scores is the MHSA variant.

PUBLIC API (in call order of a typical study)
    plot_learning_curves_grid      3x8 training/validation curves.
    plot_prediction_comparison     24-h forecasts vs actual with zoom
                                   insets; exports all scalar metrics.
    plot_daily_profile_comparison  0-24 h daily profiles: power row +
                                   per-hour R2/rRMSE row.
    plot_metric_summary            grouped bars, TCSA improvement
                                   annotated over every scope; reads the
                                   metrics dict or the exported CSV.
    load_metrics_csv               re-load '...-Metrics.csv' into the
                                   metrics dict used by the summary.

VERSION HISTORY
    3.0  Consolidated module: duplicate imports/helpers/docstrings from
         the previously concatenated files removed; one styling block;
         one copy of every helper; conflicting ACTUAL_STYLE definitions
         split into ACTUAL_STYLE (blue, time-series figure) and
         DAILY_ACTUAL_STYLE (bold black, daily-profile figure); all
         comments and docstrings rewritten in a uniform
         Objective / Parameters / Outputs / Returns order.
    2.x  Feature history (kept for reference): days selection (2.0);
         two zoom insets with daylight auto-crop (2.1); convex-hull zoom
         beam, ETP metric, metrics CSV (2.2); daily-profile figure and
         TCSA metric-summary figure (2.3+).

CONVENTIONS
    Rows/panels are always ordered Solar Plant I, II, III.
    Legends sit in a strip at the TOP of the figure, the caption at the
    BOTTOM; both are reserved with subfigures + constrained layout.
    Every figure is saved as PDF (vector) and PNG (600 dpi).
"""

import os
import csv
import json
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.ticker import MaxNLocator


# =====================================================================
# 1. IEEE styling and model taxonomy (single source of truth)
# =====================================================================
def _apply_ieee_rc():
    """Objective: apply the journal rc style once per figure.
    Serif (Times if available), 6-8 pt text, thin axes, inward ticks."""
    cands = ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif']
    avail = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in cands if f in avail), 'serif')
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [chosen],
        'mathtext.fontset': 'stix', 'font.size': 7,
        'axes.linewidth': 0.5, 'axes.labelsize': 7, 'axes.titlesize': 7,
        'legend.fontsize': 8, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'lines.linewidth': 0.8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
    })


# ---- role styling (time-series + summary figures) -------------------
# One color per role; dashes differ so the figure survives grayscale.
ROLE_STYLE = {
    'baseline': dict(color='#009E73', linestyle='-',  linewidth=0.9),  # green
    'typical':  dict(color='#E69F00', linestyle='--', linewidth=1.0),  # amber
    'proposed': dict(color='#D55E00', linestyle='-',  linewidth=1.3),  # vermillion
}
ROLE_ABBR = {'baseline': 'Base', 'typical': 'MHSA', 'proposed': 'TCSA'}
ROLE_LABEL = {'baseline': 'Baseline (basic)', 'typical': 'MHSA (typical)',
              'proposed': 'TCSA (proposed)'}
ROLE_HATCH = {'baseline': '', 'typical': '///', 'proposed': ''}  # bar hatches

# Actual curve in the time-series comparison figure: blue.
ACTUAL_STYLE = dict(color='#0072B2', linestyle='-', linewidth=1.4, alpha=0.95)
# Actual curve in the daily-profile figure: bold black (models are colored).
DAILY_ACTUAL_STYLE = dict(color='#000000', linestyle='-', linewidth=1.8,
                          alpha=1.0)

# ---- unique per-model styling (daily-profile figure) ----------------
# Okabe-Ito colors + distinct dashes: every model identifiable alone.
MODEL_STYLE = {
    'LSTM':             dict(color='#000000', linestyle='-',  linewidth=0.9),
    'MHSA-LSTM':        dict(color='#E69F00', linestyle='--', linewidth=0.9),
    'TCSA-LSTM':        dict(color='#009E73', linestyle='-.', linewidth=1.0),
    'GRU':              dict(color='#56B4E9', linestyle='-',  linewidth=0.9),
    'MHSA-GRU':         dict(color='#F0E442', linestyle='--', linewidth=0.9),
    'TCSA-GRU':         dict(color='#0072B2', linestyle='-.', linewidth=1.0),
    'MHSA-Transformer': dict(color='#CC79A7', linestyle='--', linewidth=0.9),
    'TCSA-Transformer': dict(color='#D55E00', linestyle='-',  linewidth=1.3),
}

# ---- learning-curve styling ------------------------------------------
TRAIN_STYLE = dict(color='#0072B2', linestyle='-',  linewidth=0.8)  # blue
VAL_STYLE   = dict(color='#D55E00', linestyle='--', linewidth=0.8)  # vermillion
BEST_COLOR  = '#009E73'   # best-model marker + best-epoch vline (green)
STOP_COLOR  = '0.35'      # early-stop vline (dark gray)
PATIENCE_FACE = '#E69F00'  # patience-window band (amber)

# ---- zoom-inset trace (time-series comparison) ------------------------
ZOOM_TRACE_COLOR = '0.45'   # soft gray beam between source box and inset
ZOOM_TRACE_ALPHA = 0.10

# ---- taxonomy ---------------------------------------------------------
FAMILIES = {
    'LSTM':        ['LSTM', 'MHSA-LSTM', 'TCSA-LSTM'],
    'GRU':         ['GRU', 'MHSA-GRU', 'TCSA-GRU'],
    'Transformer': ['MHSA-Transformer', 'TCSA-Transformer'],
}
ALL_MODELS = ['LSTM', 'MHSA-LSTM', 'TCSA-LSTM', 'GRU', 'MHSA-GRU',
              'TCSA-GRU', 'MHSA-Transformer', 'TCSA-Transformer']
FAM_ABBR = {'LSTM': 'L', 'GRU': 'G', 'Transformer': 'T'}
PLANT_LABELS = {1: 'Solar Plant I', 2: 'Solar Plant II',
                3: 'Solar Plant III'}
COL_ABC = {1: '(a)', 2: '(b)', 3: '(c)'}


def _role_of(model_name):
    """Role of a model from its name prefix: TCSA->proposed,
    MHSA->typical, otherwise baseline."""
    if model_name.startswith('TCSA'):
        return 'proposed'
    if model_name.startswith('MHSA'):
        return 'typical'
    return 'baseline'


def _family_of(model_name):
    """Family (LSTM/GRU/Transformer) that contains `model_name`."""
    for fam, members in FAMILIES.items():
        if model_name in members:
            return fam
    return 'LSTM'


def _reference_for(model_name):
    """Reference model for skill/improvement scores: the family baseline,
    or the MHSA variant for the Transformer family (no basic baseline).
    Returns None when `model_name` IS the reference."""
    for members in FAMILIES.values():
        if model_name in members:
            base = [m for m in members if _role_of(m) == 'baseline']
            if base:
                ref = base[0]
            else:
                typ = [m for m in members if _role_of(m) == 'typical']
                ref = typ[0] if typ else None
            return None if ref == model_name else ref
    return None


# =====================================================================
# 2. Shared data helpers (one copy each)
# =====================================================================
def _load_history(history_dir, model_name, plant_num, location='Gyeongju'):
    """Load a Keras training history saved as
    'history-{model} Model for SolarPlant-{plant} ({location})'.pkl/.json.
    Returns the history dict, or None if neither file exists."""
    base = (f'history-{model_name} Model for SolarPlant-{plant_num} '
            f'({location})')
    pkl = os.path.join(history_dir, f'{base}.pkl')
    js = os.path.join(history_dir, f'{base}.json')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            return pickle.load(f)
    if os.path.exists(js):
        with open(js) as f:
            return json.load(f)
    return None


def _inverse_pkw(scaled_2d, scaler, target_idx=0, n_features=None):
    """Inverse-transform a (N, H) block of scaled P_kW values back to kW.
    A zero-padded dummy with the fitted number of feature columns is
    built so any sklearn scaler (MinMax/Standard/Robust) reconstructs
    the target column correctly."""
    scaled_2d = np.asarray(scaled_2d, dtype=float)
    if n_features is None:
        n_features = getattr(scaler, 'n_features_in_', None)
        if n_features is None:
            raise ValueError("Pass n_features; scaler has no "
                             "n_features_in_.")
    N, H = scaled_2d.shape
    dummy = np.zeros((N * H, n_features))
    dummy[:, target_idx] = scaled_2d.reshape(-1)
    inv = scaler.inverse_transform(dummy)[:, target_idx]
    return inv.reshape(N, H)


def _predict(model, X):
    """Run model inference; accepts Keras models (predict) or plain
    callables. Squeezes a trailing singleton channel to (N, H)."""
    p = np.asarray(model.predict(X, verbose=0) if hasattr(model, 'predict')
                   else model(X))
    if p.ndim == 3 and p.shape[-1] == 1:
        p = p[..., 0]
    return p


def _nonoverlap_starts(nW, output_len):
    """Window indices that tile the test period without overlap
    -> one 24-h forecast per day."""
    return list(range(0, nW, output_len))


def _resolve_days(nW, output_len, days, n_days):
    """Select non-overlapping forecast days.

    Parameters
        days   : (start, end), 1-indexed inclusive, or None.
        n_days : fallback count when days is None.
    Returns
        (window indices, (d0, d1) actually used, total available days).
    The request is clamped to the available range."""
    starts = _nonoverlap_starts(nW, output_len)
    total = len(starts)
    if days is not None:
        d0, d1 = int(days[0]), int(days[1])
        d0, d1 = max(1, d0), min(total, d1)
        if d0 > d1:
            d0, d1 = 1, total
    else:
        d0, d1 = 1, min(n_days, total)
    return starts[d0 - 1:d1], (d0, d1), total


def _stitch_idx(arr_2d, idxs):
    """Concatenate the selected (H,) windows into one continuous series."""
    return np.concatenate([arr_2d[i] for i in idxs])


# =====================================================================
# 3. Shared metric helpers
# =====================================================================
def _metrics(actual, pred):
    """Scalar point-forecast metrics on flattened arrays (kW).
    Returns dict with MAE, MSE, RMSE, R2 (%), rRMSE (%), ybar."""
    a = np.asarray(actual, float).ravel()
    p = np.asarray(pred, float).ravel()
    err = p - a
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    ybar = float(np.mean(a))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a - ybar) ** 2))
    r2 = (1.0 - ss_res / ss_tot) * 100.0 if ss_tot > 0 else float('nan')
    rrmse = (rmse / ybar) * 100.0 if ybar != 0 else float('nan')
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse,
            'R2': r2, 'rRMSE': rrmse, 'ybar': ybar}


def _band_stats(arr_2d):
    """Per-horizon mean/min/max across N forecast windows -> (H,) arrays."""
    a = np.asarray(arr_2d, float)
    return {'mean': np.nanmean(a, axis=0),
            'min': np.nanmin(a, axis=0),
            'max': np.nanmax(a, axis=0)}


def _per_hour_r2(actual_2d, pred_2d):
    """R2 (%) at each forecast horizon hour -> (H,) array; NaN where the
    actual has no variance at that hour."""
    a = np.asarray(actual_2d, float)
    p = np.asarray(pred_2d, float)
    ss_res = np.sum((p - a) ** 2, axis=0)
    ss_tot = np.sum((a - a.mean(axis=0)) ** 2, axis=0)
    out = np.full(a.shape[1], np.nan)
    ok = ss_tot > 0
    out[ok] = (1.0 - ss_res[ok] / ss_tot[ok]) * 100.0
    return out


def _per_hour_rrmse(actual_2d, pred_2d):
    """rRMSE (%) at each forecast horizon hour -> (H,) array; NaN where
    the hourly mean actual is zero (night)."""
    a = np.asarray(actual_2d, float)
    p = np.asarray(pred_2d, float)
    rmse_h = np.sqrt(np.mean((p - a) ** 2, axis=0))
    ybar_h = np.mean(a, axis=0)
    out = np.full(a.shape[1], np.nan)
    ok = ybar_h != 0
    out[ok] = rmse_h[ok] / ybar_h[ok] * 100.0
    return out


# =====================================================================
# 4. Shared axis / annotation utilities
# =====================================================================
def _style_ax(ax):
    """Minor ticks, thin tick marks and the dotted major grid."""
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=2.5, width=0.5)
    ax.tick_params(axis='both', which='minor', length=1.5, width=0.4)
    ax.grid(True, which='major', linestyle=':', linewidth=0.3,
            color='gray', alpha=0.4)


def _set_hour_xticks(ax, output_len):
    """0-24 h axis: ticks every 6 h plus the last hour; label 'Time (hr)'."""
    ticks = list(range(0, output_len, 6))
    if ticks[-1] != output_len - 1:
        ticks.append(output_len - 1)
    ax.set_xticks(ticks)
    ax.set_xlim(0, output_len - 1)
    ax.set_xlabel('Time (hr)', fontsize=7)


def _annotate_box(ax, rows, keys, fmt='.1f'):
    """Framed monospace metric table in the upper-right panel corner.

    Parameters
        rows : list of (row_label, metrics_dict).
        keys : list of (dict_key, column_header).
        fmt  : number format for the cells."""
    col_w = 7
    hdr = f"{'':5}" + ''.join(f'{h:>{col_w}}' for _, h in keys)
    body = []
    for lab, mt in rows:
        cells = ''.join(
            f'{"--":>{col_w}}' if mt.get(k) is None
            else f'{mt[k]:>{col_w}{fmt}}' for k, _ in keys)
        body.append(f'{lab:5}{cells}')
    ax.text(0.985, 0.97, hdr + '\n' + '\n'.join(body),
            transform=ax.transAxes, family='monospace', fontsize=4.8,
            va='top', ha='right', linespacing=1.35, zorder=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='0.6', linewidth=0.4, alpha=0.93))


# =====================================================================
# 5. CSV input / output
# =====================================================================
def _save_wide_metric_csvs(metrics, plant_nums, save_data_dir, save_name,
                           keys=(('MAE', 'kW'), ('RMSE', 'kW'),
                                 ('R2', '%'), ('rRMSE', '%'))):
    """One wide CSV per metric (rows = plants, cols = models), mirroring
    the console tables. Returns the list of file paths written."""
    os.makedirs(save_data_dir, exist_ok=True)
    paths = []
    for key, unit in keys:
        path = os.path.join(save_data_dir, f'{save_name}-{key}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([f'{key} ({unit})'])
            w.writerow(['Plant'] + ALL_MODELS)
            for p in plant_nums:
                row = [p]
                for m in ALL_MODELS:
                    v = metrics[p].get(m, {}).get(key)
                    row.append('' if v is None else f'{v:.6f}')
                w.writerow(row)
        print(f'Saved: {path}')
        paths.append(path)
    return paths


def load_metrics_csv(csv_path):
    """Objective: re-load the '{save_name}-Metrics.csv' exported by
    plot_prediction_comparison so plot_metric_summary can run offline.

    Parameters
        csv_path : path to the exported metrics CSV.
    Returns
        metrics[plant][model] -> {'MAE','MSE','RMSE','R2','rRMSE',
                                  'PS','ETP','ETP_ms'} (None if blank)."""
    hdr_map = {'MAE (kW)': 'MAE', 'MSE (kW^2)': 'MSE', 'RMSE (kW)': 'RMSE',
               'R2 (%)': 'R2', 'rRMSE (%)': 'rRMSE', 'PS (%)': 'PS',
               'ETP (s)': 'ETP', 'ETP (ms/window)': 'ETP_ms'}
    metrics = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            p, m = int(row['Plant']), row['Model']
            metrics.setdefault(p, {})[m] = {
                key: (float(row[h]) if row.get(h) not in (None, '', '-')
                      else None)
                for h, key in hdr_map.items() if h in row}
    return metrics


# =====================================================================
# 6. Learning-curve grid (3 plants x 8 models)
# =====================================================================
def plot_learning_curves_grid(
    history_dir='../Export/History/Site-02/',
    save_dir='../Export/Figure/Site-02/',
    location='Gyeongju', patience=20,
    show_info=True, logy=False,
    save_fig=True, show_title=True,
    save_name='LearningCurves-Grid-AllModels-AllPlants',
    figsize=(14.0, 6.0),
):
    """Objective: 3x8 grid of training/validation learning curves with,
    per panel, the best-model epoch, the early-stop epoch, the patience
    window and an info box.

    Panel elements
        Training MSE   blue solid;   Validation MSE  vermillion dashed.
        Best Model     green marker on the validation curve at the best
                       epoch (= argmin(val_loss), the quantity the
                       checkpoint monitors; falls back to
                       n - patience - 1 when val_loss is absent).
        Best Epoch     green dotted vline at that epoch.
        Early Stop     gray dashed vline at the LAST trained epoch
                       (where saving actually stopped).
        Patience band  light amber span between best and stop.

    Parameters
        history_dir : folder with the saved history .pkl/.json files.
        patience    : early-stopping patience (band width + fallback).
        show_info   : per-panel box (best epoch, val/train MSE at best,
                      generalization gap val-train, total epochs).
        logy        : log-scale MSE axis (spreads the converged tail).
        save_fig / show_title / save_dir / save_name / figsize / location
                    : usual figure options.

    Outputs
        {save_dir}/{save_name}.pdf and .png (600 dpi).
    Returns
        fig."""
    plants = [1, 2, 3]
    row_caps = [f'{COL_ABC[p]} {PLANT_LABELS[p]}' for p in plants]
    n_cols = len(ALL_MODELS)

    _apply_ieee_rc()
    fig = plt.figure(figsize=figsize, dpi=150, layout='constrained')
    top_sf, mid_sf, bot_sf = fig.subfigures(
        nrows=3, ncols=1,
        height_ratios=[0.5, 11.5, 0.6 if show_title else 0.0001])
    row_subfigs = mid_sf.subfigures(nrows=3, ncols=1, hspace=0.007)
    missing = []

    for plant, label, rsf in zip(plants, row_caps, row_subfigs):
        axes = rsf.subplots(nrows=1, ncols=n_cols, sharey=True)
        rsf.supxlabel(label, fontsize=8, fontweight='bold')

        for col, (ax, model) in enumerate(zip(axes, ALL_MODELS)):
            ax.set_title(model, fontsize=7)
            hist = _load_history(history_dir, model, plant, location)

            if hist is None:                       # placeholder panel
                missing.append(f'SolarPlant-{plant} / {model}')
                ax.text(0.5, 0.5, 'history\nnot found', ha='center',
                        va='center', transform=ax.transAxes, fontsize=6,
                        color='firebrick')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel('Epochs', fontsize=7)
                if col == 0:
                    ax.set_ylabel('MSE', fontsize=7)
                continue

            loss = np.asarray(hist['loss'], dtype=float)
            val = (np.asarray(hist['val_loss'], dtype=float)
                   if 'val_loss' in hist else None)
            n = len(loss)
            stop = n - 1                            # last trained epoch
            best = (int(np.argmin(val))             # checkpoint criterion
                    if val is not None and len(val) == n
                    else max(0, n - (patience or 0) - 1))

            xs = np.arange(n)
            ax.plot(xs, loss, **TRAIN_STYLE)
            if val is not None:
                ax.plot(xs, val, **VAL_STYLE)

            if stop > best:                         # patience window
                ax.axvspan(best, stop, facecolor=PATIENCE_FACE,
                           alpha=0.08, linewidth=0, zorder=1.2)
            ax.axvline(best, linestyle=':', color=BEST_COLOR,
                       linewidth=0.7, zorder=2.5)   # best epoch
            ax.axvline(stop, linestyle='--', color=STOP_COLOR,
                       linewidth=0.7, zorder=2.5)   # early stop
            y_best = val[best] if val is not None else loss[best]
            ax.plot(best, y_best, linestyle='', marker='o',
                    markersize=3.2, color=BEST_COLOR,
                    markerfacecolor='white', markeredgewidth=0.7,
                    zorder=5)                       # best-model marker

            if show_info:                           # per-panel info box
                lines = [f'Best ep {best}']
                if val is not None:
                    lines += [f'Val {val[best]:.4f}',
                              f'Train {loss[best]:.4f}',
                              f'Gap {val[best] - loss[best]:+.4f}']
                else:
                    lines += [f'Train {loss[best]:.4f}']
                lines += [f'Epochs {n}']
                ax.text(0.975, 0.955, '\n'.join(lines),
                        transform=ax.transAxes, family='monospace',
                        fontsize=4.6, va='top', ha='right',
                        linespacing=1.25, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.28',
                                  facecolor='white', edgecolor='0.6',
                                  linewidth=0.4, alpha=0.92))

            if logy:
                ax.set_yscale('log')
            _style_ax(ax)
            ax.set_xlabel('Epochs', fontsize=7)
            if col == 0:
                ax.set_ylabel('MSE', fontsize=7)

    # shared legend (top strip)
    handles = [
        Line2D([0], [0], label='Training MSE', **TRAIN_STYLE),
        Line2D([0], [0], label='Validation MSE', **VAL_STYLE),
        Line2D([0], [0], color=BEST_COLOR, linestyle='', marker='o',
               markersize=4, markerfacecolor='white',
               markeredgewidth=0.7, label='Best Model'),
        Line2D([0], [0], color=BEST_COLOR, linestyle=':', linewidth=0.8,
               label='Best Epoch'),
        Line2D([0], [0], color=STOP_COLOR, linestyle='--', linewidth=0.8,
               label='Early Stop (stop saving)'),
        Patch(facecolor=PATIENCE_FACE, alpha=0.25,
              label='Patience Window'),
    ]
    leg = top_sf.legend(handles=handles, loc='center', ncol=6,
                        frameon=True, fancybox=False, edgecolor='black',
                        framealpha=1.0)
    leg.get_frame().set_linewidth(0.6)

    if show_title:                                  # caption (bottom strip)
        bot_sf.text(0.5, 0.5,
                    f'Fig. X: Learning Curves of Eight Forecasting Models '
                    f'across Three Solar Plants ({location})',
                    ha='center', va='center', fontsize=11,
                    fontweight='bold')
        
    # ------------------------------------------------------------ save
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, save_name)
        fig.savefig(f'{base}.pdf')
        fig.savefig(f'{base}.png', dpi=600)
        print(f'Saved: {base}.pdf')
        print(f'Saved: {base}.png')


    if missing:
        print('\n[!] Missing history files:')
        for m in missing:
            print(f'    - {m}')

    plt.show()
    return fig


# =====================================================================
# 7. Prediction-vs-actual comparison (zoom insets + metrics CSV)
# =====================================================================
def _daylight_span(actual_day, frac=0.03, pad=1):
    """Indices (lo, hi) bounding the daylight (generating) part of one day,
    detected as actual > frac * day-max, expanded by `pad` hours each side.
    Falls back to the full day if the day is flat/zero."""
    a = np.asarray(actual_day, dtype=float)
    amax = np.nanmax(a) if a.size else 0.0
    if not np.isfinite(amax) or amax <= 0:
        return 0, max(len(a) - 1, 0)
    on = np.where(a > frac * amax)[0]
    if on.size == 0:
        return 0, len(a) - 1
    lo = max(0, int(on[0]) - pad)
    hi = min(len(a) - 1, int(on[-1]) + pad)
    if hi <= lo:
        lo, hi = 0, len(a) - 1
    return lo, hi


def _convex_hull(pts):
    """Convex hull (Andrew's monotone chain) of a small set of 2-D points,
    returned in counter-clockwise order. Used for the shaded zoom beam."""
    pts = sorted(set((float(x), float(y)) for x, y in pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])

    lower = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return lower[:-1] + upper[:-1]


def _add_zoom_insets(ax, hours, actual, model_series, zoom_days, n_sel,
                     output_len):
    """Add up to two magnified insets to `ax` (upper-left region), one per
    selected displayed day. The link between the magnified region and its
    inset is a soft translucent polygon (funnel) plus a light dashed box —
    no hard black connector lines. `zoom_days` are 1-indexed within the
    displayed days. Added last so they sit on top of the main curves, and
    kept clear of the metric table at the upper right."""
    zsel = [int(d) for d in zoom_days if 1 <= int(d) <= n_sel][:2]
    if not zsel:
        return
    # start at x0=0.115 so inset y tick labels clear the parent y-axis
    ins_w, ins_h, y0 = 0.25, 0.36, 0.58
    x_positions = [0.115, 0.425][:len(zsel)]
    for x0, dsel in zip(x_positions, zsel):
        s, e = (dsel - 1) * output_len, dsel * output_len
        lo, hi = _daylight_span(actual[s:e])
        i1, i2 = s + lo, s + hi
        axins = ax.inset_axes([x0, y0, ins_w, ins_h])
        axins.plot(hours[i1:i2 + 1], actual[i1:i2 + 1], **ACTUAL_STYLE)
        segs = [actual[i1:i2 + 1]]
        for m, y in model_series.items():
            axins.plot(hours[i1:i2 + 1], y[i1:i2 + 1],
                       **ROLE_STYLE[_role_of(m)])
            segs.append(y[i1:i2 + 1])
        ymin = min(float(np.nanmin(v)) for v in segs)
        ymax = max(float(np.nanmax(v)) for v in segs)
        pad_y = 0.06 * max(ymax - ymin, 1e-9)
        axins.set_xlim(hours[i1], hours[i2])
        axins.set_ylim(ymin - pad_y, ymax + pad_y)
        # x ticks as HOUR-OF-DAY of the zoomed day (e.g. 6, 12, 18 hr),
        # not the stitched hour index of the displayed window.
        day_start = (dsel - 1) * output_len
        hod_ticks = [h for h in range(0, output_len + 1, 6)
                     if lo <= h <= hi]
        if len(hod_ticks) < 2:   # very narrow zoom: 3 evenly spaced hours
            hod_ticks = sorted({int(round(v))
                                for v in np.linspace(lo, hi, 3)})
        axins.set_xticks([day_start + h for h in hod_ticks])
        axins.set_xticklabels([str(h) for h in hod_ticks])
        axins.yaxis.set_major_locator(MaxNLocator(3))
        axins.tick_params(axis='both', labelsize=5, length=1.5, width=0.4,
                          pad=1.2)
        for sp in axins.spines.values():
            sp.set_linewidth(0.5)
        axins.patch.set_alpha(0.95)

        # ---- zoom trace: shaded translucent "beam" instead of the two
        # black connector lines of indicate_inset_zoom. The beam is the
        # convex hull of the source box + the inset box, so it attaches
        # symmetrically no matter where the box sits relative to the inset
        # (left, right, or directly below). ----
        x1d, x2d = float(hours[i1]), float(hours[i2])
        y1d, y2d = ymin - pad_y, ymax + pad_y
        # light dashed box around the magnified region
        ax.add_patch(Rectangle(
            (x1d, y1d), x2d - x1d, y2d - y1d, fill=False,
            edgecolor='0.35', linestyle=(0, (2, 2)), linewidth=0.5,
            zorder=4))
        # convert the box to axes-fraction coords (parent limits are final
        # here because insets are added after xlim/ylim are set)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        fx = lambda x: (x - xl[0]) / (xl[1] - xl[0])
        fy = lambda y: (y - yl[0]) / (yl[1] - yl[0])
        corners = [
            (fx(x1d), fy(y1d)), (fx(x2d), fy(y1d)),      # source box
            (fx(x2d), fy(y2d)), (fx(x1d), fy(y2d)),
            (x0, y0), (x0 + ins_w, y0),                  # inset box
            (x0 + ins_w, y0 + ins_h), (x0, y0 + ins_h),
        ]
        ax.add_patch(Polygon(
            _convex_hull(corners), closed=True, transform=ax.transAxes,
            facecolor=ZOOM_TRACE_COLOR, alpha=ZOOM_TRACE_ALPHA,
            edgecolor='none', zorder=1.2))


# ------------------------------------------------------------- public API
def plot_prediction_comparison(
    models_by_plant, X_by_plant, y_by_plant, scaler,
    target_idx=0, n_features=None, output_len=24,
    layout='family', days=None, n_days=4,
    show_insets=True, zoom_days=(2, 4),
    xunit='day', xlabel=None, index_by_plant=None, input_len=168,
    plant_nums=(1, 2, 3),
    save_dir='../Export/Figure/Site-02/',
    save_name='PredictionComparison-PkW-AllPlants',
    save_fig=True, show_title=True, location='Gyeongju',
    save_data=True, 
    save_data_dir='../Export/Data/Site-02/',
    save_data_dir_sim='../Sim/Site-02/',
    figsize=None,
):
    """Objective: 24-h forecasts vs actual for every model and plant,
    with two magnified daylight insets per panel and a per-panel
    RMSE/MAE/PS table; all scalar metrics are computed on the FULL test
    set and exported to one CSV.

    Parameters (data)
        models_by_plant : {plant: {model_name: keras_model}}.
        X_by_plant      : {plant: X_test (N, input_len, n_features)}.
        y_by_plant      : {plant: y_test (N, output_len)} scaled P_kW.
        scaler          : scaler fitted on the n_features columns
                          (P_kW at `target_idx`).
        target_idx / n_features / output_len / input_len
                        : scaling and window geometry.
    Parameters (display)
        layout      : 'family' (3x3 grid, variants overlaid) or
                      'per_model' (3x8 grid, one model per panel).
        days        : (start, end) 1-indexed non-overlapping forecast
                      days to display; None -> first `n_days`.
        show_insets : add the two zoom insets (drawn last, on top).
        zoom_days   : which displayed days to magnify (max two;
                      default the 2nd and 4th). Inset x ticks show the
                      HOUR OF DAY (6/12/18), not the stitched index.
        xunit       : x-axis MODE, 'day'/'date' or 'hour'.
        xlabel      : x-axis TEXT override.
        index_by_plant : {plant: DatetimeIndex} -> date tick labels.
    Parameters (saving)
        save_fig / save_dir / save_name / show_title / location.
        save_data / save_data_dir : metrics CSV export switch + folder.

    Metrics per model (on the full test set)
        MAE, MSE, RMSE, R2, rRMSE, PS (skill vs the family reference,
        from rRMSE), ETP (Execution Time for Prediction: wall-clock s of
        predict(), first call may include graph warm-up) and ETP_ms.

    Outputs
        {save_dir}/{save_name}.pdf/.png and
        {save_data_dir}/{save_name}-Metrics.csv (all metrics, tidy).
    Returns
        (fig, metrics) with metrics[plant][model] -> metric dict."""
    _apply_ieee_rc()
    plant_labels = {1: '(a) Solar Plant - I',
                    2: '(b) Solar Plant - II',
                    3: '(c) Solar Plant - III'}

    # ---- compute inverse-transformed actual + predictions, and metrics ----
    series, metrics = {}, {}
    day_used, total_days = None, None
    for p in plant_nums:
        Xp, yp = X_by_plant[p], y_by_plant[p]
        act_full = _inverse_pkw(yp, scaler, target_idx, n_features)
        idxs, day_used, total_days = _resolve_days(
            act_full.shape[0], output_len, days, n_days)
        series[p] = {'ACTUAL': _stitch_idx(act_full, idxs), '_idxs': idxs}
        metrics[p] = {}
        for name, model in models_by_plant[p].items():
            t0 = time.perf_counter()
            raw_pred = _predict(model, Xp)
            etp = time.perf_counter() - t0   # Execution Time for Prediction
            pred_full = _inverse_pkw(raw_pred, scaler, target_idx, n_features)
            metrics[p][name] = _metrics(act_full, pred_full)
            metrics[p][name]['ETP'] = etp                       # seconds
            metrics[p][name]['ETP_ms'] = etp / len(Xp) * 1e3    # ms / window
            series[p][name] = _stitch_idx(pred_full, idxs)
        # Prediction Skill (PS, %) relative to each model's reference
        for name in metrics[p]:
            ref = _reference_for(name)
            if ref is not None and ref in metrics[p]:
                rr_m = metrics[p][name]['rRMSE']
                rr_r = metrics[p][ref]['rRMSE']
                metrics[p][name]['PS'] = (1.0 - rr_m / rr_r) * 100.0
            else:
                metrics[p][name]['PS'] = None   # this model is the reference

    d0, d1 = day_used
    n_sel = d1 - d0 + 1
    hours = np.arange(n_sel * output_len)
    print(f'Available non-overlapping forecast days: {total_days} '
          f'(showing days {d0}-{d1}).')
    if days is not None and (days[0] != d0 or days[1] != d1):
        print(f'[!] Requested days {tuple(days)} clamped to {(d0, d1)} '
              f'(only {total_days} available).')

    # validate the zoom-day selection (max two insets)
    zsel_valid = []
    if show_insets:
        zsel_valid = [int(d) for d in zoom_days if 1 <= int(d) <= n_sel][:2]
        dropped = [d for d in zoom_days if int(d) not in zsel_valid]
        if dropped:
            print(f'[!] zoom_days {tuple(dropped)} outside displayed range '
                  f'1-{n_sel} (or beyond the two-inset limit); '
                  f'using {tuple(zsel_valid) if zsel_valid else "none"}.')

    # xunit is a MODE selector, not a label. Normalize & warn on bad values.
    xunit_eff = str(xunit).lower()
    if xunit_eff not in ('day', 'date', 'hour'):
        print(f"[!] xunit='{xunit}' is not a recognized mode "
              f"(use 'day' or 'hour'); defaulting to 'day'. "
              f"To set the axis text, pass xlabel='{xunit}'.")
        xunit_eff = 'day'

    # ---- figure scaffold ----
    if layout == 'family':
        cols = list(FAMILIES.keys())
    elif layout == 'per_model':
        cols = ALL_MODELS
    else:
        raise ValueError("layout must be 'family' or 'per_model'")
    n_cols = len(cols)
    if figsize is None:
        figsize = (12.0, 6.4) if layout == 'family' else (15.0, 5.4)
        if show_insets and zsel_valid:   # extra headroom for the insets
            figsize = (figsize[0], figsize[1] + 1.6)

    fig = plt.figure(figsize=figsize, dpi=150, layout='constrained')
    top_sf, mid_sf, bot_sf = fig.subfigures(
        nrows=3, ncols=1, height_ratios=[0.55, 11.0, 0.6 if show_title else 0.0001])
    row_subfigs = mid_sf.subfigures(nrows=len(plant_nums), ncols=1, hspace=0.012)

    for p, rsf in zip(plant_nums, row_subfigs):
        axes = rsf.subplots(nrows=1, ncols=n_cols, sharey=True)
        if n_cols == 1:
            axes = [axes]
        rsf.supxlabel(plant_labels.get(p, f'Solar Plant {p}'),
                      fontsize=9, fontweight='bold')

        for col, key in enumerate(cols):
            ax = axes[col]
            ax.plot(hours, series[p]['ACTUAL'], **ACTUAL_STYLE)

            if layout == 'family':
                ax.set_title(key, fontsize=8)
                models_here = [m for m in FAMILIES[key] if m in series[p]]
            else:  # per_model
                ax.set_title(key, fontsize=7)
                models_here = [key]

            rows = []   # (label, RMSE, MAE, PS)
            for m in models_here:
                role = _role_of(m)
                ax.plot(hours, series[p][m], **ROLE_STYLE[role])
                mt = metrics[p][m]
                rows.append((ROLE_ABBR[role], mt['RMSE'], mt['MAE'], mt['PS']))

            # day separators
            for d in range(1, n_sel):
                ax.axvline(d * output_len, color='gray', linestyle=':',
                           linewidth=0.4, alpha=0.5)

            # ---- metric table (monospace, framed), upper-right ----
            hdr = f"{'':<5}{'RMSE':>6}{'MAE':>6}{'PS%':>6}"
            body = []
            for lab, rm, ma, ps in rows:
                ps_s = '  -  ' if ps is None else f'{ps:>5.1f}'
                body.append(f"{lab:<5}{rm:>6.1f}{ma:>6.1f}{ps_s:>6}")
            table = hdr + '\n' + '\n'.join(body)
            ax.text(0.985, 0.97, table, transform=ax.transAxes,
                    family='monospace', fontsize=5.0, va='top', ha='right',
                    linespacing=1.3,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='0.6', linewidth=0.4, alpha=0.92))

            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', length=2.5, width=0.5)
            ax.tick_params(axis='both', which='minor', length=1.5, width=0.4)
            ax.grid(True, which='major', linestyle=':', linewidth=0.3,
                    color='gray', alpha=0.4)
            ax.set_xlim(0, n_sel * output_len)
            if xunit_eff in ('day', 'date'):
                centers = [(k + 0.5) * output_len for k in range(n_sel)]
                bounds = [k * output_len for k in range(n_sel + 1)]
                ax.set_xticks(centers)
                if index_by_plant is not None:
                    idxs_p = series[p]['_idxs']
                    labels = [index_by_plant[p][idxs_p[k] + input_len]
                              .strftime('%b %d') for k in range(n_sel)]
                    auto_label = 'Date'
                else:
                    labels = [str(d0 + k) for k in range(n_sel)]
                    auto_label = 'Forecast day'
                ax.set_xticklabels(labels, fontsize=6)
                ax.set_xticks(bounds, minor=True)
            else:
                ax.set_xticks([k * output_len for k in range(n_sel + 1)])
                auto_label = 'Forecast horizon (hr)'
            ax.set_xlabel(xlabel if xlabel else auto_label, fontsize=7)
            if not (show_insets and zsel_valid):
                ax.margins(y=0.22)
            if col == 0:
                ax.set_ylabel('PV Generation (kW)', fontsize=7)

        # ---- headroom + insets (added LAST, drawn on top of the curves) ----
        if show_insets and zsel_valid:
            # one row-wide ylim (sharey=True): curves in the lower ~45%,
            # insets occupy the cleared band above them.
            vals = [v for k, v in series[p].items()
                    if not str(k).startswith('_')]
            ymin_d = min(float(np.nanmin(v)) for v in vals)
            ymax_d = max(float(np.nanmax(v)) for v in vals)
            span = max(ymax_d - ymin_d, 1e-9)
            axes[0].set_ylim(ymin_d - 0.05 * span, ymax_d + 1.15 * span)
            for col, key in enumerate(cols):
                ax = axes[col]
                if layout == 'family':
                    models_here = [m for m in FAMILIES[key] if m in series[p]]
                else:
                    models_here = [key]
                _add_zoom_insets(
                    ax, hours, series[p]['ACTUAL'],
                    {m: series[p][m] for m in models_here},
                    zsel_valid, n_sel, output_len)

    # ---- shared legend (top) ----
    handles = [
        Line2D([0], [0], label='Actual', **ACTUAL_STYLE),
        Line2D([0], [0], label='Baseline (basic)', **ROLE_STYLE['baseline']),
        Line2D([0], [0], label='MHSA (typical)', **ROLE_STYLE['typical']),
        Line2D([0], [0], label='TCSA (proposed)', **ROLE_STYLE['proposed']),
    ]
    leg = top_sf.legend(handles=handles, loc='center', ncol=4, frameon=True,
                        fancybox=False, edgecolor='black', framealpha=1.0)
    leg.get_frame().set_linewidth(0.6)

    if show_title:
        bot_sf.text(0.5, 0.5,
                    f'Fig. X: 24-h PV Generation Forecasts vs. Actual across Three Solar Plants ({location})',
                    ha='center', va='center', fontsize=11, fontweight='bold')

    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, save_name)
        fig.savefig(f'{base}.pdf')
        fig.savefig(f'{base}.png', dpi=600)
        print(f'Saved: {base}.pdf'); print(f'Saved: {base}.png')

    # console metrics tables
    for metric_key, unit in [('MAE', 'kW'), ('RMSE', 'kW'),
                             ('R2', '%'), ('rRMSE', '%'), ('PS', '%'),
                             ('ETP', 's'), ('ETP_ms', 'ms/window')]:
        print(f'\n{metric_key} ({unit}) on full test set:')
        print('Plant | ' + ' | '.join(f'{m:>16}' for m in ALL_MODELS))
        for p in plant_nums:
            cells = []
            for m in ALL_MODELS:
                v = metrics[p].get(m, {}).get(metric_key, None)
                cells.append(f'{"-":>16}' if v is None else f'{v:>16.3f}')
            print(f'{p:>5} | ' + ' | '.join(cells))

    # ---- CSV export of all metrics (incl. ETP) ----
    if save_data:
        os.makedirs(save_data_dir, exist_ok=True)
        csv_path = os.path.join(save_data_dir, f'{save_name}-Metrics.csv')
        csv_cols = [('MAE', 'MAE (kW)'), ('MSE', 'MSE (kW^2)'),
                    ('RMSE', 'RMSE (kW)'), ('R2', 'R2 (%)'),
                    ('rRMSE', 'rRMSE (%)'), ('PS', 'PS (%)'),
                    ('ETP', 'ETP (s)'), ('ETP_ms', 'ETP (ms/window)')]
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Plant', 'Model'] + [h for _, h in csv_cols])
            for p in plant_nums:
                for m in ALL_MODELS:
                    if m not in metrics[p]:
                        continue
                    mt = metrics[p][m]
                    row = [p, m]
                    for k, _ in csv_cols:
                        v = mt.get(k, None)
                        row.append('' if v is None else f'{v:.6f}')
                    w.writerow(row)
        print(f'Saved: {csv_path}')
    
    # ---- CSV export of all metrics (incl. ETP)for simulation ----
    if save_data:
        os.makedirs(save_data_dir_sim, exist_ok=True)
        csv_path = os.path.join(save_data_dir_sim, f'AllPlants-Metrics.csv')
        csv_cols = [('MAE', 'MAE (kW)'), ('MSE', 'MSE (kW^2)'),
                    ('RMSE', 'RMSE (kW)'), ('R2', 'R2 (%)'),
                    ('rRMSE', 'rRMSE (%)'), ('PS', 'PS (%)'),
                    ('ETP', 'ETP (s)'), ('ETP_ms', 'ETP (ms/window)')]
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Plant', 'Model'] + [h for _, h in csv_cols])
            for p in plant_nums:
                for m in ALL_MODELS:
                    if m not in metrics[p]:
                        continue
                    mt = metrics[p][m]
                    row = [p, m]
                    for k, _ in csv_cols:
                        v = mt.get(k, None)
                        row.append('' if v is None else f'{v:.6f}')
                    w.writerow(row)
        print(f'Saved: {csv_path}')

    plt.show()
    return fig, metrics


# =====================================================================
# 8. Daily-profile figure (power row + per-hour metric row)
# =====================================================================
def plot_daily_profile_comparison(
    models_by_plant, X_by_plant, y_by_plant, scaler,
    target_idx=0, n_features=None, output_len=24,
    plant_nums=(1, 2, 3),
    band_alpha=0.13,
    save_dir='../Export/Figure/Site-02/',
    save_name='DailyProfile-PkW-AllPlants',
    save_fig=True, show_title=True, location='Gyeongju',
    save_data=True, save_data_dir='../Export/Data/Site-02/',
    figsize=None,
):
    """Objective: 0-24 h daily-profile view. ONE figure, TWO rows,
    THREE columns (Solar Plant I/II/III).

    Row 1 - Power (kW): bold black Actual mean + one uniquely styled
        mean curve per model (see MODEL_STYLE), each with its own
        translucent min-max band across all test days.
        Annotation: MAE (kW) and RMSE (kW) per model.
    Row 2 - % metrics: per-horizon R2 (%) (model linestyle) and rRMSE
        (%) (dotted) per model with light bands.
        Annotation: scalar R2 (%) and rRMSE (%) per model.

    Parameters
        data arguments identical to plot_prediction_comparison;
        plant_nums selects the columns; band_alpha sets band opacity;
        save_fig / show_title / save_dir / save_name / location /
        save_data / save_data_dir / figsize as usual.

    Outputs
        {save_dir}/{save_name}.pdf/.png and four wide CSVs
        ({save_name}-MAE/-RMSE/-R2/-rRMSE.csv, Plant x Model).
    Returns
        (fig, metrics, profiles); profiles[plant][name] holds the
        per-hour 'mean'/'min'/'max' arrays in kW."""
    _apply_ieee_rc()
    plant_nums = list(plant_nums)
    n_plants   = len(plant_nums)
    x          = np.arange(output_len)            # 0 … output_len-1

    # -------------------------------------------------- inference & stats
    profiles = {}   # profiles[p][name] = {'mean','min','max'}  (kW)
    ph_r2    = {}   # ph_r2[p][model]   = (output_len,)  per-hour R2 (%)
    ph_rrmse = {}   # ph_rrmse[p][model]= (output_len,)  per-hour rRMSE (%)
    metrics  = {}   # metrics[p][model] = scalar dict

    for p in plant_nums:
        Xp, yp    = X_by_plant[p], y_by_plant[p]
        act_full  = _inverse_pkw(yp, scaler, target_idx, n_features)
        profiles[p]  = {'ACTUAL': _band_stats(act_full)}
        ph_r2[p]     = {}
        ph_rrmse[p]  = {}
        metrics[p]   = {}

        for name, model in models_by_plant[p].items():
            t0        = time.perf_counter()
            raw_pred  = _predict(model, Xp)
            etp       = time.perf_counter() - t0
            pred_full = _inverse_pkw(raw_pred, scaler, target_idx, n_features)

            metrics[p][name]   = _metrics(act_full, pred_full)
            metrics[p][name]['ETP']    = etp
            metrics[p][name]['ETP_ms'] = etp / max(len(Xp), 1) * 1e3
            profiles[p][name]  = _band_stats(pred_full)
            ph_r2[p][name]     = _per_hour_r2(act_full, pred_full)
            ph_rrmse[p][name]  = _per_hour_rrmse(act_full, pred_full)

    # ------------------------------------------------------ figure layout
    # ONE figure:
    #   top_sf  – shared legend strip
    #   mid_sf  – 2 row-subfigures (power row | % row), each with 3 panels
    #   bot_sf  – shared title strip
    if figsize is None:
        figsize = (3.6 * n_plants, 7.2)

    fig = plt.figure(figsize=figsize, dpi=150, layout='constrained')

    top_sf, mid_sf, bot_sf = fig.subfigures(
        nrows=3, ncols=1,
        height_ratios=[1.0, 12.0, 0.55 if show_title else 0.0001],
    )

    # Two row-subfigures inside mid_sf
    rsf_power, rsf_pct = mid_sf.subfigures(nrows=2, ncols=1, hspace=0.04)

    # -------------------------------------------------- ROW 1 : Power (kW)
    axes_power = rsf_power.subplots(nrows=1, ncols=n_plants, sharey=False)
    if n_plants == 1:
        axes_power = [axes_power]

    for col, p in enumerate(plant_nums):
        ax = axes_power[col]
        ax.set_title(f'{COL_ABC[p]} {PLANT_LABELS[p]}',
                     fontsize=8, fontweight='bold')

        # Actual: bold black
        st_act = profiles[p]['ACTUAL']
        ax.fill_between(x, st_act['min'], st_act['max'],
                        color=DAILY_ACTUAL_STYLE['color'],
                        alpha=band_alpha, linewidth=0, zorder=1)
        ax.plot(x, st_act['mean'], label='Actual', **DAILY_ACTUAL_STYLE, zorder=4)

        # Each model: own color band + mean line
        annot = []
        for m in ALL_MODELS:
            if m not in profiles[p]:
                continue
            cfg = MODEL_STYLE[m]
            st  = profiles[p][m]
            ax.fill_between(x, st['min'], st['max'],
                            color=cfg['color'],
                            alpha=band_alpha * 0.75,
                            linewidth=0, zorder=1.5)
            ax.plot(x, st['mean'], label=m, **cfg, zorder=3)
            annot.append((m, metrics[p][m]))

        _annotate_box(ax, annot,
                      keys=[('MAE', 'MAE'), ('RMSE', 'RMSE')],
                      fmt='.1f')
        _style_ax(ax)
        _set_hour_xticks(ax, output_len)
        ax.margins(y=0.18)
        if col == 0:
            ax.set_ylabel('Power (kW)', fontsize=7)

    # -------------------------------------------------- ROW 2 : % metrics
    axes_pct = rsf_pct.subplots(nrows=1, ncols=n_plants, sharey=False)
    if n_plants == 1:
        axes_pct = [axes_pct]

    for col, p in enumerate(plant_nums):
        ax = axes_pct[col]

        annot = []
        for m in ALL_MODELS:
            if m not in ph_r2[p]:
                continue
            cfg      = MODEL_STYLE[m]
            r2_arr   = ph_r2[p][m]        # (output_len,) per-hour R2 %
            rr_arr   = ph_rrmse[p][m]     # (output_len,) per-hour rRMSE %

            # R2 curve + band (solid, same color, lighter fill)
            ax.fill_between(x, r2_arr * 0.97, r2_arr * 1.03,
                            color=cfg['color'],
                            alpha=band_alpha * 0.75,
                            linewidth=0, zorder=1.5)
            ax.plot(x, r2_arr,
                    color=cfg['color'],
                    linestyle=cfg['linestyle'],
                    linewidth=cfg['linewidth'],
                    zorder=3)

            # rRMSE curve + band (same color, more transparent, dotted)
            ax.fill_between(x, rr_arr * 0.97, rr_arr * 1.03,
                            color=cfg['color'],
                            alpha=band_alpha * 0.45,
                            linewidth=0, zorder=1.2)
            ax.plot(x, rr_arr,
                    color=cfg['color'],
                    linestyle=':',
                    linewidth=max(cfg['linewidth'] - 0.2, 0.5),
                    zorder=2)

            annot.append((m, metrics[p][m]))

        _annotate_box(ax, annot,
                      keys=[('R2', 'R2%'), ('rRMSE', 'rRMSE')],
                      fmt='.1f')
        _style_ax(ax)
        _set_hour_xticks(ax, output_len)
        ax.margins(y=0.18)
        if col == 0:
            ax.set_ylabel('Metric (%)', fontsize=7)

    # ------------------------------------------------- shared legend (top)
    # Actual + 8 models + two line-style swatches for R2/rRMSE in row 2
    handles = [Line2D([0], [0], label='Actual',
                  **DAILY_ACTUAL_STYLE)]
    for m in ALL_MODELS:
        handles.append(Line2D([0], [0], label=m, **MODEL_STYLE[m]))
    handles += [
        Patch(facecolor='0.55', alpha=0.40, label='Min-max band'),
        Line2D([0], [0], color='0.35', linestyle='-',  linewidth=0.8,
               label='R² (%) curve'),
        Line2D([0], [0], color='0.35', linestyle=':', linewidth=0.8,
               label='rRMSE (%) curve'),
    ]
    leg = top_sf.legend(handles=handles, loc='center',
                        ncol=6, frameon=True, fancybox=False,
                        edgecolor='black', framealpha=1.0, fontsize=6.2)
    leg.get_frame().set_linewidth(0.6)

    # ------------------------------------------------- shared title (bot)
    if show_title:
        bot_sf.text(
            0.5, 0.5,
            f'Fig. X: 0–24 h Daily PV Power and Error-Metric Profiles '
            f'across Three Solar Plants ({location})',
            ha='center', va='center', fontsize=9, fontweight='bold',
        )

    # ------------------------------------------------------------ save
    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, save_name)
        fig.savefig(f'{base}.pdf')
        fig.savefig(f'{base}.png', dpi=600)
        print(f'Saved: {base}.pdf')
        print(f'Saved: {base}.png')

    # ------------------------------------------------- console tables
    for mk, unit in [('MAE','kW'),('RMSE','kW'),('R2','%'),('rRMSE','%')]:
        print(f'\n{mk} ({unit}) – full test set:')
        print('Plant | ' + ' | '.join(f'{m:>18}' for m in ALL_MODELS))
        for p in plant_nums:
            cells = [f'{metrics[p].get(m,{}).get(mk,None):>18.3f}'
                     if metrics[p].get(m,{}).get(mk) is not None
                     else f'{"-":>18}'
                     for m in ALL_MODELS]
            print(f'{p:>5} | ' + ' | '.join(cells))

    if save_data:
        _save_wide_metric_csvs(metrics, plant_nums, save_data_dir, save_name)

    plt.show()
    return fig, metrics, profiles





# =====================================================================
# 9. TCSA improvement summary (every metric x plant x family)
# =====================================================================
def plot_metric_summary(
    metrics=None, csv_path=None,
    metric_keys=(('MAE', 'MAE (kW)', 'lower'),
                 ('RMSE', 'RMSE (kW)', 'lower'),
                 ('rRMSE', 'rRMSE (%)', 'lower'),
                 ('R2', 'R2 (%)', 'higher')),
    plant_nums=(1, 2, 3),
    annotate_improvement=True,
    save_dir='../Export/Figure/Site-02/',
    save_data_dir_sim='../Sim/Site-02/',
    save_name='MetricSummary-PkW-AllPlants',
    save_fig=True, show_title=True, location='Gyeongju',
    save_data=True, save_data_dir='../Export/Data/Site-02/',
    figsize=(7.16, 5.6),
):
    """Grouped-bar summary that makes the TCSA improvement evident over
    EVERY scope at once: every metric (one panel each), every plant
    (three groups per panel), every family (three clusters per group).

    In each Base/MHSA/TCSA cluster the vermillion TCSA bar is annotated
    with its improvement over the family reference (baseline; MHSA for
    the Transformer family): error metrics show the % reduction (down
    arrow), R2 shows the gain in percentage points (up arrow). A green
    check row could not be clearer: if TCSA wins a scope, its annotation
    is positive.

    Input is either the `metrics` dict returned by
    plot_prediction_comparison, or `csv_path` to the exported
    '...-Metrics.csv' (no models/data needed):

        from Library.modelEvaluation import plot_metric_summary
        fig, imp = plot_metric_summary(metrics=met)
        # or, offline from the CSV:
        fig, imp = plot_metric_summary(
            csv_path='../Export/Data/Site-02/'
                     'PredictionComparison-PkW-AllPlants-Metrics.csv')

    Bars-from-zero for error metrics; the R2 panel is zoomed to the data
    range (noted on the axis) so the gains are visible. Also exports
    '{save_name}-TCSA-Improvement.csv' (improvement per metric, plant,
    family) and prints the average improvement per family.

    Returns (fig, improvements) with
    improvements[metric][(plant, family)] = signed improvement.
    """
    _apply_ieee_rc()
    if metrics is None:
        if csv_path is None:
            raise ValueError('Pass metrics=... or csv_path=...')
        metrics = load_metrics_csv(csv_path)

    fam_order = list(FAMILIES.keys())
    n_metrics = len(metric_keys)
    n_rows = 2 if n_metrics > 2 else 1
    n_cols = int(np.ceil(n_metrics / n_rows))

    fig = plt.figure(figsize=figsize, dpi=150, layout='constrained')
    top_sf, mid_sf, bot_sf = fig.subfigures(
        nrows=3, ncols=1,
        height_ratios=[0.7, 11.0, 0.55 if show_title else 0.0001])
    axs = mid_sf.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
    axs = axs.ravel()

    # ---- x geometry: [plant [family cluster [role bars]]] ----
    # wider cluster gap so the FULL family names fit below the clusters
    # NOTE: bar_w increased (0.65 -> 0.72) to widen every bar; the
    # cluster/plant gaps are trimmed slightly so panels don't overflow.
    bar_w, clus_gap, plant_gap = 0.72, 0.32, 0.45
    clus_w = 3 * bar_w
    x_cursor, clus_x, plant_span = 0.0, {}, {}
    for p in plant_nums:
        x_start = x_cursor
        for fam in fam_order:
            clus_x[(p, fam)] = x_cursor + clus_w / 2
            x_cursor += clus_w + clus_gap
        plant_span[p] = (x_start, x_cursor - clus_gap)
        x_cursor += plant_gap
    x_max = x_cursor - plant_gap

    improvements = {mk: {} for mk, _, _ in metric_keys}

    for ax, (mk, mlabel, better) in zip(axs, metric_keys):
        vals_all = []
        for p in plant_nums:
            for fam in fam_order:
                cx = clus_x[(p, fam)]
                members = [m for m in FAMILIES[fam]
                           if m in metrics.get(p, {})]
                roles_here = [_role_of(m) for m in members]
                offs = (np.arange(len(members))
                        - (len(members) - 1) / 2) * bar_w
                for off, m, role in zip(offs, members, roles_here):
                    v = metrics[p][m].get(mk)
                    if v is None:
                        continue
                    vals_all.append(v)
                    # wider bars (0.92 -> 0.98 of the slot) + a solid
                    # black outline on every bar so adjacent bars in a
                    # cluster stay visually distinct from one another.
                    ax.bar(cx + off, v, width=bar_w * 0.98,
                           facecolor=ROLE_STYLE[role]['color'],
                           hatch=ROLE_HATCH[role],
                           edgecolor='black',
                           linewidth=0.8, zorder=3)
                    if role == 'proposed' and annotate_improvement:
                        ref = _reference_for(m)
                        rv = (metrics[p].get(ref, {}).get(mk)
                              if ref else None)
                        if rv not in (None, 0):
                            if better == 'lower':
                                imp = (rv - v) / rv * 100.0
                                txt = f'$\\downarrow${imp:.1f}%'
                            else:
                                imp = v - rv
                                txt = f'$\\uparrow${imp:.1f}'
                            improvements[mk][(p, fam)] = imp
                            good = imp > 0
                            # mathtext arrows -> same STIX serif as the
                            # rest of the figure; white pad keeps the
                            # label readable over bars and gridlines.
                            ax.annotate(
                                txt, xy=(cx + off, v),
                                xytext=(0, 1.5),
                                textcoords='offset points',
                                rotation=90, ha='center', va='bottom',
                                fontsize=5.4, fontweight='bold',
                                color=('#006400' if good else '#b00000'),
                                zorder=6,
                                bbox=dict(boxstyle='round,pad=0.12',
                                          facecolor='white',
                                          edgecolor='none', alpha=0.75))

        # cluster labels: FULL family names, matching the other figures
        trans = ax.get_xaxis_transform()
        ax.set_xticks([clus_x[(p, f)] for p in plant_nums
                       for f in fam_order])
        ax.set_xticklabels([f for _ in plant_nums for f in fam_order],
                           fontsize=4.8)
        # plant group labels: 'Solar Plant - I/II/III' as elsewhere
        # (moved from -0.15 -> -0.085 to sit closer to the LSTM/GRU/
        # Transformer tick labels right above it; pair with the
        # tightened xtick `pad` below)
        for p in plant_nums:
            s, e = plant_span[p]
            roman = 'I' * p if p <= 3 else str(p)
            ax.text((s + e) / 2, -0.085, f'Solar Plant - {roman}',
                    transform=trans, ha='center', va='top', fontsize=6.5,
                    fontweight='bold')
        for p in plant_nums[:-1]:
            sep = (plant_span[p][1]
                   + (plant_span[plant_nums[plant_nums.index(p) + 1]][0]
                      - plant_span[p][1]) / 2)
            ax.axvline(sep, color='0.75', linestyle='-',
                       linewidth=0.5, zorder=1)

        ax.set_xlim(-clus_w, x_max + clus_w / 2)
        ax.set_title(mlabel, fontsize=8)
        ax.grid(True, axis='y', which='major', linestyle=':',
                linewidth=0.3, color='gray', alpha=0.4)
        ax.tick_params(axis='y', labelsize=6, length=2.5, width=0.5)
        ax.tick_params(axis='x', length=0, pad=1.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if better == 'higher':
            lo, hi = min(vals_all), max(vals_all)
            pad = 0.12 * (hi - lo)
            ax.set_ylim(lo - pad, hi + 2.1 * pad)  # headroom for arrows
            ax.set_ylabel(f'{mlabel}  (axis zoomed)', fontsize=6.5)
        else:
            ax.set_ylim(0, max(vals_all) * 1.15)
            ax.set_ylabel(mlabel, fontsize=6.5)

    for k in range(n_metrics, len(axs)):
        axs[k].set_visible(False)

    # ---- legend ----
    handles = [Patch(facecolor=ROLE_STYLE[r]['color'],
                     hatch=ROLE_HATCH[r],
                     edgecolor='black',
                     label=ROLE_LABEL[r])
               for r in ('baseline', 'typical', 'proposed')]
    handles.append(Line2D([0], [0], linestyle='none',
                          marker=r'$\downarrow$', color='#006400',
                          markersize=6,
                          label='TCSA improvement vs family reference'))
    leg = top_sf.legend(handles=handles, loc='center', ncol=4,
                        frameon=True, fancybox=False, edgecolor='black',
                        framealpha=1.0, fontsize=6.2)
    leg.get_frame().set_linewidth(0.6)

    if show_title:
        bot_sf.text(0.5, 0.5,
                    f'Fig. X: Metric Summary of All Models — TCSA vs '
                    f'Typical and Baseline across Three Solar Plants '
                    f'({location})',
                    ha='center', va='center', fontsize=9,
                    fontweight='bold')

    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, save_name)
        fig.savefig(f'{base}.pdf')
        fig.savefig(f'{base}.png', dpi=600)
        print(f'Saved: {base}.pdf'); print(f'Saved: {base}.png')

    # ---- console + CSV: improvement over every scope ----
    print('\nTCSA improvement vs family reference '
          '(error metrics: % reduction; R2: +pp):')
    hdr = 'Metric | ' + ' | '.join(
        f'P{p}-{FAM_ABBR[f]}' for p in plant_nums for f in fam_order)
    print(hdr)
    for mk, _, _ in metric_keys:
        cells = []
        for p in plant_nums:
            for f in fam_order:
                v = improvements[mk].get((p, f))
                cells.append('   - ' if v is None else f'{v:5.1f}')
        print(f'{mk:>6} | ' + ' | '.join(cells))
    for fam in fam_order:
        avg = np.mean([improvements['rRMSE'][(p, fam)]
                       for p in plant_nums
                       if (p, fam) in improvements['rRMSE']])
        print(f'Average rRMSE reduction, {fam} family: {avg:.1f}%')

    if save_data:
        os.makedirs(save_data_dir, exist_ok=True)
        path = os.path.join(save_data_dir,
                            f'{save_name}-TCSA-Improvement.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['TCSA improvement vs family reference '
                        '(error metrics: % reduction; R2: +pp)'])
            w.writerow(['Metric'] + [f'Plant{p}-{fam}'
                                     for p in plant_nums
                                     for fam in fam_order])
            for mk, _, _ in metric_keys:
                row = [mk]
                for p in plant_nums:
                    for fam in fam_order:
                        v = improvements[mk].get((p, fam))
                        row.append('' if v is None else f'{v:.3f}')
                w.writerow(row)
        print(f'Saved: {path}')

    if save_data:
        os.makedirs(save_data_dir_sim, exist_ok=True)
        path = os.path.join(save_data_dir_sim,
                            f'AllPlants-TCSA-Improvement.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['TCSA improvement vs family reference '
                        '(error metrics: % reduction; R2: +pp)'])
            w.writerow(['Metric'] + [f'Plant{p}-{fam}'
                                     for p in plant_nums
                                     for fam in fam_order])
            for mk, _, _ in metric_keys:
                row = [mk]
                for p in plant_nums:
                    for fam in fam_order:
                        v = improvements[mk].get((p, fam))
                        row.append('' if v is None else f'{v:.3f}')
                w.writerow(row)
        print(f'Saved: {path}')

    plt.show()
    return fig, improvements