"""
dataProcessing.py

Utility functions for the IGPS project (data processing & visualization).

Usage:
    from Library.dataProcessing import plot_pv_generation
    plot_pv_generation(df_PV, PlantName='Plant-A', AreaName='Ansan Area')
    plot_weather_data(df_ENV, x_col='timestamp', save_fig=True)
    filled_ENV, mask_synth = fix_env_nans(df_ENV, rain_is_cumulative=True, add_noise=True)
    filled_PV, mask_synth = fix_pv_nans(df_PV, pv_cols=['PV1_P_pv', 'PV2_P_pv'], seed=42)
    plot_selected_columns(df, Selected_Columns=['T_env', 'RH_env'], x='timestamp')
"""
import pandas as pd
import matplotlib.pyplot as plt


def plot_pv_generation(df_PV, PlantName='Plant-A', AreaName='Ansan Area', ax=None):
    """
    Plot hourly PV power generation with a 7-day (weekly) moving average.

    Parameters
    ----------
    df_PV : pandas.DataFrame
        Must contain a 'timestamp' column and a 'P_pv' column (power in kW).
    PlantName : str, optional
        Name of the plant, used in the title. Default 'Plant-A'.
    AreaName : str, optional
        Name of the area/site, used in the title. Default 'Ansan Area'.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the plot was drawn on (useful for further customization).
    """
    # Work on a copy so the caller's DataFrame is not mutated
    df_PV = df_PV.copy()

    # Ensure datetime and use timestamp as index
    df_PV['timestamp'] = pd.to_datetime(df_PV['timestamp'])
    ts = df_PV.set_index('timestamp')['P_pv']

    # Hourly mean (grey) and weekly moving average (orange)
    hourly = ts.resample('h').mean()
    weekly_ma = hourly.rolling(window=24 * 7, min_periods=1).mean()  # 7-day moving avg over hourly data

    # Create axes only if one wasn't provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3), dpi=100)

    # Hourly generation — thin grey
    ax.plot(hourly.index, hourly.values,
            color='grey', linewidth=0.5, alpha=0.8,
            label='Hourly Power Generation')

    # Weekly moving average — orange/tan, thicker
    ax.plot(weekly_ma.index, weekly_ma.values,
            color='#E0A96D', linewidth=1.5,
            label='Weekly Moving Average')

    # Styling
    ax.set_title(f'{PlantName} : Solar Power Generation at {AreaName}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Timestamp (hr)', fontsize=11, fontstyle='italic', fontweight='bold')
    ax.set_ylabel('Power (kW)', fontsize=11, fontstyle='italic', fontweight='bold')
    ax.set_ylim(0, 100.5)
    ax.grid(True, linestyle=':', color='salmon', alpha=0.4)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    return ax



import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_weather_data(df, x_col='timestamp', save_fig=False):
    # ---- IEEE-style rcParams ----
    mpl.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'DejaVu Serif'],
        'font.size':        8,
        'axes.linewidth':   0.6,
        'lines.linewidth':  0.6,
        'xtick.direction':  'in',
        'ytick.direction':  'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'axes.grid':        True,
        'grid.linewidth':   0.3,
        'grid.alpha':       0.4,
    })

    # ---- Prepare & clean ----
    df[x_col] = pd.to_datetime(df[x_col])
    env = df.set_index(x_col).sort_index()

    # Drop physically impossible / empty records
    env = env[env['P_atm'] > 800]          # removes the 0-hPa startup garbage
    env.loc[env['V_wind'] > 60, 'V_wind'] = pd.NA   # 60 m/s ~ hurricane; spikes above = sensor errors

    cols   = ['T_env', 'RH_env', 'V_wind', 'P_rain', 'D_env', 'P_atm']
    labels = ['$T$ (°C)', 'RH (%)', '$V_{wind}$ (m/s)',
            '$P_{rain}$ (mm)', '$D_{env}$', '$P_{atm}$ (hPa)']

    env_h = env[cols].resample('h').mean()

    # ---- Figure: IEEE double-column width (7.16 in) ----
    fig, axes = plt.subplots(len(cols), 1, figsize=(7.16, 6.5), sharex=True)

    for ax, col, lab in zip(axes, cols, labels):
        ax.plot(env_h.index, env_h[col], color='black')
        ax.set_ylabel(lab, fontsize=8)
        ax.margins(x=0.01)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel('Time', fontsize=8)
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.15, left=0.10, right=0.98, top=0.98, bottom=0.07)

    # Save vector + high-res raster for submission 
    if save_fig:
        fig.savefig('../Export/Figure/ENV_timeseries.pdf', bbox_inches='tight')
        fig.savefig('../Export/Figure/ENV_timeseries.png', dpi=600, bbox_inches='tight')
    # Show the plot
    plt.show()





"""
Fill NaN in the environmental columns while keeping the same seasonal shape
AND the same hour-to-hour texture as the period that was actually measured.

The sensors only ran for a window (~2023 -> 2025); before and after that the
columns are entirely missing. ffill leaves a flat line for years, linear
interpolate draws a diagonal across years, and a plain harmonic fit leaves a
too-smooth sine ribbon. So each column is rebuilt as:
    smooth seasonal+daily curve  +  real weather scatter resampled from the
    measured period.

Per column:
    1. Drop physically-impossible readings (e.g. V_wind > 60 m/s) -> NaN.
    2. Fill short gaps (<= max_interp_gap hours) by time interpolation.
    3. Fill long gaps with a harmonic seasonal/daily backbone, then add back
       real residual blocks bootstrapped from the measured window so the
       filled stretch has the same spread and choppiness as the real data.

NOTE: the pre-2023 and post-2025 values are synthesized for recovery. They
match the statistics of the measured period to look like the real weather of
those years. fix_env_nans returns a boolean mask flagging every fabricated cell.
"""

import numpy as np
import pandas as pd

ENV_COLS = ['T_env_C', 'H_env_%', 'V_wind_m/s', 'P_rain_mm/h',
            'D_env_g/m^3', 'P_atm_hPa']

# Plausible physical ranges. Values outside -> NaN, then re-imputed.
BOUNDS = {
    'T_env_C':     (-45, 55),
    'H_env_%':     (0, 100),
    'V_wind_m/s':  (0, 60),      # kills the 220 / 270 m/s sensor spikes
    'D_env_g/m^3': (-30, 40),
    'P_atm_hPa':   (870, 1085),
}
# Columns clamped after the fill so resampled scatter can't push them
# past a hard physical limit (humidity 0-100, wind/rain >= 0).
HARD_FLOOR = {'H_env_%': (0, 100), 'V_wind_m/s': (0, None), 'P_rain_mm/h': (0, None)}


def load(df_or_path):
    """Parse DT, sort, and reindex onto a gap-free hourly grid."""
    df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path.copy()
    df['DT'] = pd.to_datetime(df['DT'])
    df = df.set_index('DT').sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq='h')
    df = df.reindex(full)
    df.index.name = 'DT'
    return df


def clip_outliers(df):
    """Blank out readings outside their physical range so they get re-imputed."""
    for c, (lo, hi) in BOUNDS.items():
        if c in df.columns:
            df[c] = df[c].where(df[c].between(lo, hi))
    return df


def climatology_fill(s, max_interp_gap=6, n_year=3, n_day=2,
                     block_hours=24*14, seed=0, **kw):
    """Fill NaN with a smooth seasonal/daily backbone plus real residual blocks
       resampled from the measured period (matches its texture and spread).

       n_year / n_day: harmonic terms for the yearly / daily cycle.
       block_hours:    length of each resampled residual block (texture knob).
    """
    s = s.astype(float)

    # Short gaps: smooth interpolation between real points only.
    out = s.interpolate('time', limit=max_interp_gap, limit_area='inside')

    # Seasonal + daily backbone via least-squares harmonic fit.
    # t = days since the start of the series.
    t = s.index.values.astype('int64') / 1e9 / 86400.0
    t = t - t[0]
    cols = [np.ones_like(t)]
    for k in range(1, n_year + 1):                 # annual + sub-annual shape
        w = 2*np.pi*k/365.25; cols += [np.sin(w*t), np.cos(w*t)]
    for k in range(1, n_day + 1):                  # diurnal cycle
        w = 2*np.pi*k/1.0;    cols += [np.sin(w*t), np.cos(w*t)]
    X = np.column_stack(cols)

    # Fit the backbone on observed points, then evaluate it everywhere.
    obs = s.notna().values
    beta, *_ = np.linalg.lstsq(X[obs], s.values[obs], rcond=None)
    seasonal = X @ beta

    # Real weather scatter = observed minus backbone.
    resid_obs = (s.values - seasonal)[obs]

    # Tile random contiguous residual blocks across the whole timeline so the
    # filled regions carry the same variability as the measured one.
    rng = np.random.default_rng(seed)
    n = len(s)
    resid_full = np.empty(n)
    i = 0
    while i < n:
        start = rng.integers(0, max(1, len(resid_obs) - block_hours))
        chunk = resid_obs[start:start + block_hours][:n - i]   # trim final block
        resid_full[i:i + len(chunk)] = chunk
        i += len(chunk)

    # Backbone + scatter, used only where the value was missing.
    fill = pd.Series(seasonal + resid_full, index=s.index)
    out = out.where(out.notna(), fill)

    # Enforce hard physical limits the scatter may have crossed.
    if s.name in HARD_FLOOR:
        lo, hi = HARD_FLOOR[s.name]
        out = out.clip(lower=lo, upper=hi)
    return out


def cumulative_to_rate(s):
    """P_rain reads as a cumulative gauge; convert to an hourly increment.
       Negative jumps are counter resets -> set to 0."""
    rate = s.diff()
    rate[rate < 0] = 0.0
    rate.iloc[0] = 0.0
    return rate


def fix_env_nans(df, rain_is_cumulative=True, add_noise=True):
    """Clean and fill every environmental column.
       Returns (filled_df, mask) where mask marks each originally-missing cell."""
    df = clip_outliers(df)
    mask_synth = pd.DataFrame(False, index=df.index, columns=ENV_COLS)

    for c in ENV_COLS:
        if c not in df.columns:
            continue
        mask_synth[c] = df[c].isna()           # record which cells we fabricate

        if c == 'P_rain_mm/h' and rain_is_cumulative:
            # Convert the cumulative gauge to a rate, then fill the rate.
            rate = cumulative_to_rate(df[c])
            df[c] = climatology_fill(rate, add_noise=add_noise)
        else:
            df[c] = climatology_fill(df[c], add_noise=add_noise)

    return df, mask_synth










"""
Fill NaN in the PV / electrical columns the way fix_env_nans fills the
environmental ones -- same structure (clip outliers -> mask -> per-column
fill -> return df, mask) -- but with a fill strategy suited to solar data.

WHY NOT THE HARMONIC CLIMATOLOGY USED FOR ENV
---------------------------------------------
Environmental variables (T, RH, pressure) are smooth, so a harmonic backbone
plus tiled residual blocks reconstructs them well. PV columns are different:
power, voltage and current are ZERO every night and switch on/off sharply at
sunrise/sunset. A 2-term daily harmonic cannot draw that edge, and tiling
residual blocks across the timeline pastes daytime scatter onto night hours,
inventing power at 2 a.m. that a non-negativity clip can't remove.

THE FILL USED HERE: DIURNAL CLIMATOLOGY BOOTSTRAP
-------------------------------------------------
Each missing cell is filled by sampling a real observed value from the SAME
hour-of-day and SAME month. This preserves, with no modelling assumptions:
    * night = 0 (night-hour buckets are full of ~0 readings),
    * the daily sunrise->noon->sunset ramp,
    * the seasonal envelope (month bucket),
    * realistic cloud-driven scatter and the PF 0/100 bimodality.
Sparse buckets fall back to the same hour across all months, then to the
column's global observed pool.

The pre/post-measurement stretches are synthesized for recovery; they match
the statistics of the measured period. fix_pv_nans returns a boolean mask
flagging every fabricated cell.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

# Bounds/limits are keyed by MEASUREMENT TYPE (the part after the unit prefix,
# e.g. 'PV1_P_pv' -> 'P_pv'), so any number of units -- PV1/PV2, or PV1/2/3,
# or more -- resolves the same physical range automatically.

def _measure(col):
    """'PV1_P_pv' -> 'P_pv', 'PV12_Freq' -> 'Freq'. Strips the PV<n>_ prefix."""
    parts = col.split('_', 1)
    return parts[1] if len(parts) == 2 else col

# Plausible physical ranges. Values outside -> NaN, then re-imputed.
PV_BOUNDS = {
    'P_pv': (0, 500),
    'P_dc': (0, 500),
    'V_dc': (0, 1500),
    'V_i':  (0, 1500),
    'PF':   (0, 100),
    'Freq': (55, 65),     # kills the 0 Hz dropouts
}

# Hard limits clamped AFTER the fill so sampled scatter can't cross them.
PV_HARD = {
    'P_pv': (0, None),
    'P_dc': (0, None),
    'V_dc': (0, None),
    'V_i':  (0, None),
    'PF':   (0, 100),
    'Freq': (55, 65),
}


def clip_pv_outliers(df, pv_cols):
    """Blank out readings outside their physical range so they get re-imputed."""
    for c in pv_cols:
        lo, hi = PV_BOUNDS.get(_measure(c), (None, None))
        if c in df.columns and lo is not None:
            df[c] = df[c].where(df[c].between(lo, hi))
    return df


def diurnal_bootstrap_fill_old (s, max_interp_gap=6, min_pool=10, seed=0):
    """Fill NaN by resampling observed values from the same (month, hour) bucket.

    Parameters
    ----------
    s : pandas.Series
        Hourly series indexed by a DatetimeIndex.
    max_interp_gap : int
        Short gaps (<= this many hours) are time-interpolated between real
        points first; only longer gaps are bootstrapped.
    min_pool : int
        A (month, hour) bucket with fewer than this many observations is
        widened to (hour, all months) so the sample stays representative.
    seed : int
        RNG seed for reproducible fills.
    """
    s = s.astype(float)

    # Short gaps: smooth interpolation between real points only.
    out = s.interpolate('time', limit=max_interp_gap, limit_area='inside')

    need = out.isna()
    if not need.any():
        return out

    rng     = np.random.default_rng(seed)
    month   = s.index.month.values
    hour    = s.index.hour.values
    obs     = s.notna().values
    vals    = s.values

    # Build sampling pools from the OBSERVED points only.
    pool_mh = defaultdict(list)   # (month, hour) -> observed values
    pool_h  = defaultdict(list)    # hour          -> observed values
    for v, m, h, o in zip(vals, month, hour, obs):
        if o:
            pool_mh[(m, h)].append(v)
            pool_h[h].append(v)
    pool_mh = {k: np.asarray(v) for k, v in pool_mh.items()}
    pool_h  = {k: np.asarray(v) for k, v in pool_h.items()}
    global_obs = vals[obs]

    filled = out.values.copy()
    for i in np.where(need.values)[0]:
        m, h = month[i], hour[i]
        pool = pool_mh.get((m, h))
        if pool is None or len(pool) < min_pool:      # widen if bucket is thin
            pool = pool_h.get(h)
        if pool is None or len(pool) == 0:            # last resort
            pool = global_obs
        filled[i] = pool[rng.integers(0, len(pool))]

    return pd.Series(filled, index=s.index)


def diurnal_bootstrap_fill(s, min_pool=10, seed=0):
    """Fill NaN by resampling observed values from the same (month, hour) bucket.

    Parameters
    ----------
    s : pandas.Series
        Hourly series indexed by a DatetimeIndex.
    min_pool : int
        A (month, hour) bucket with fewer than this many observations is
        widened to (hour, all months) so the sample stays representative.
    seed : int
        RNG seed for reproducible fills.

    Notes
    -----
    No time-interpolation step is used. Linear interpolation across a NaN
    gap that spans sunset -> sunrise draws a straight line between the last
    evening reading and the next morning reading, producing a triangle of
    fake power through the night. The hour-bucket bootstrap below preserves
    night = 0 automatically: night-hour pools are dominated by ~0 readings,
    so any NaN at e.g. 02:00 samples back to 0.
    """
    s = s.astype(float)

    need = s.isna()
    if not need.any():
        return s

    rng = np.random.default_rng(seed)
    month = s.index.month.values
    hour = s.index.hour.values
    obs = s.notna().values
    vals = s.values

    # Build sampling pools from the OBSERVED points only.
    pool_mh = defaultdict(list)   # (month, hour) -> observed values
    pool_h = defaultdict(list)    # hour          -> observed values
    for v, m, h, o in zip(vals, month, hour, obs):
        if o:
            pool_mh[(m, h)].append(v)
            pool_h[h].append(v)
    pool_mh = {k: np.asarray(v) for k, v in pool_mh.items()}
    pool_h = {k: np.asarray(v) for k, v in pool_h.items()}
    global_obs = vals[obs]

    filled = vals.copy()
    for i in np.where(need.values)[0]:
        m, h = month[i], hour[i]
        pool = pool_mh.get((m, h))
        if pool is None or len(pool) < min_pool:      # widen if bucket is thin
            pool = pool_h.get(h)
        if pool is None or len(pool) == 0:            # last resort
            pool = global_obs
        filled[i] = pool[rng.integers(0, len(pool))]

    return pd.Series(filled, index=s.index)



def fix_pv_nans(df, pv_cols, seed=0):
    """Clean and fill the given PV / electrical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Already on a gap-free hourly DT index (use load() first).
    pv_cols : list of str
        The PV columns to fix, e.g. ['PV1_P_pv', ..., 'PV2_Freq'] or a
        three-unit list including PV3_*. Physical limits are resolved from
        each column's measurement type, so any unit count works.
    seed : int
        RNG seed for reproducible fills.

    Returns
    -------
    (filled_df, mask)
        mask marks each originally-missing cell.
    """
    df = clip_pv_outliers(df, pv_cols)
    present = [c for c in pv_cols if c in df.columns]
    mask_synth = pd.DataFrame(False, index=df.index, columns=present)

    for c in present:
        mask_synth[c] = df[c].isna()                  # record fabricated cells
        s = diurnal_bootstrap_fill(df[c], seed=seed)
        lo, hi = PV_HARD.get(_measure(c), (None, None))
        if lo is not None or hi is not None:
            s = s.clip(lower=lo, upper=hi)
        df[c] = s

    return df, mask_synth




















"""
Plot one or more selected columns against a time/index axis, IEEE style.

Each selected column gets its own panel sized 12 x 3 inches at 100 dpi,
stacked vertically and sharing the x-axis.

Parameters
----------
df : pandas.DataFrame
    The data. `x` may be either the index (e.g. the 'DT' datetime index)
    or one of the columns.
Selected_Columns : str or list of str
    Column name(s) to plot on the y-axis. A single string is allowed.
x : str, optional
    The column or index selected as the x-axis. If it matches the
    index (default 'DT'), the index is used; otherwise the matching
    column is used.

Returns
-------
(fig, axes)
    The Matplotlib figure and the array of axes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_selected_columns(df, Selected_Columns, x='DT'):
    
    # ---- IEEE-style rcParams ----
    mpl.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman', 'DejaVu Serif'],
        'font.size':        8,
        'axes.linewidth':   0.6,
        'lines.linewidth':  0.6,
        'xtick.direction':  'in',
        'ytick.direction':  'in',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'axes.grid':        True,
        'grid.linewidth':   0.3,
        'grid.alpha':       0.4,
    })

    # Allow a single column name to be passed as a string
    if isinstance(Selected_Columns, str):
        Selected_Columns = [Selected_Columns]

    # Resolve the x-axis: index if x is the index name, else a column
    if x == df.index.name:
        x_vals = df.index
    elif x in df.columns:
        x_vals = pd.to_datetime(df[x]) if x.lower() in ('dt', 'timestamp') else df[x]
    else:
        raise KeyError(f"'{x}' is neither the index name nor a column in df.")

    n = len(Selected_Columns)

    # 12 x 3 per panel, 100 dpi
    fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), dpi=100, sharex=True)
    if n == 1:
        axes = [axes]  # keep iteration consistent for a single panel

    for ax, col in zip(axes, Selected_Columns):
        ax.plot(x_vals, df[col], color='gray')
        ax.set_ylabel(col, fontsize=8)
        ax.margins(x=0.01)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel(x, fontsize=8)
    fig.align_ylabels(axes)
    fig.tight_layout()

    plt.show()
    return fig, axes