"""
dataAnalysis.py
================

A small analysis/plotting library for site meteorological + PV inverter
data (hourly resolution).

Main entry point
----------------
    plot_diurnal_patterns(df, columns=None, dt_col='DT', ...)

Given a DataFrame with a datetime column and one row per hour, this builds
an IEEE-style, two-column-width figure that shows, for every requested
variable, the diurnal (0-23 h) pattern:

    * every individual day as a thin grey line (the background "cloud")
    * the hour-of-day Min / Max envelope as a shaded band + dashed lines
      (Min in blue, Max in red)
    * optionally, the hour-of-day Mean as a bold black line
      (drawn ONLY when show_years=False -- see note below)
    * optionally, one colored line per calendar year showing that year's
      own mean diurnal pattern (set show_years=True), so year-to-year
      shape changes are easy to compare

Note on the Mean line
---------------------
When show_years=True the overall (all-days) Mean is essentially the average
of the per-year means and simply overlaps them, so it is deliberately NOT
drawn in that mode. The bold black Mean line therefore appears only in the
plain "all-days" figure (show_years=False).

Saving
------
Saving is controlled by three arguments:
    save_fig  : master on/off switch (default False -> nothing is written)
    save_dir  : output directory (created if missing)
    save_name : base filename (no extension)
When save_fig=True the figure is written as BOTH `<save_dir>/<save_name>.pdf`
and `<save_dir>/<save_name>.png`. Keeping save_name a plain token (e.g.
'diurnal_patterns') gives clean filenames; put any human-readable caption in
`fig_caption` instead (shown on the figure only when show_caption=True).

Typical usage
-------------
    import pandas as pd
    from dataAnalysis import plot_diurnal_patterns

    Site_02_raw = pd.read_csv('Site_02_Data.gzip', compression='gzip')
    rename_columns = {
        'timestamp': 'DT',
        'PV1_P_pv' : 'P_PV1_kW',
        'PV2_P_pv' : 'P_PV2_kW',
        'PV3_P_pv' : 'P_PV3_kW',
    }
    Site_02 = Site_02_raw.rename(columns=rename_columns)[[
        'DT', 'P_PV1_kW', 'P_PV2_kW', 'P_PV3_kW', 'T_env_C', 'H_env_%',
        'V_wind_m/s', 'P_rain_mm/h', 'D_env_g/m^3', 'P_atm_hPa',
    ]]

    fig = plot_diurnal_patterns(
        Site_02,
        show_individual_days=True,
        show_stats=True,
        show_years=True,
        save_fig=True,
        save_dir='../Export/Figure/Site-02/',
        show_caption=True,
        fig_caption="Fig. 1. Diurnal patterns at Site 02 (2021-2026)",
        figsize=(7.4, 6.2),
    )
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

__all__ = [
    "plot_diurnal_patterns",
    "plot_diurnal_pattern_single",
    "remove_isolated_spikes",
    "get_year_color_map",
]

# --------------------------------------------------------------------------
# IEEE-style plotting defaults
# --------------------------------------------------------------------------
# "Times New Roman" is listed first and preferred if installed; otherwise
# "Liberation Serif" (metric-compatible) or "DejaVu Serif" render reliably.
# savefig.bbox="tight" is important here: the stacked legend boxes sit ABOVE
# the figure and the caption BELOW it, and a tight bbox crops the saved file
# to include both instead of clipping them.
IEEE_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}

# IEEE double-column (full page width) figure width in inches.
IEEE_TWO_COLUMN_WIDTH_IN = 7.16

# --------------------------------------------------------------------------
# Default axis-label formatting (LaTeX/mathtext) for known variable names.
# Falls back to the raw column name for anything not listed here; pass your
# own `labels` dict to plot_diurnal_patterns to override/extend this map.
# --------------------------------------------------------------------------
DEFAULT_LABELS = {
    "P_PV1_kW":     r"$P_{pv}^{1}$ (kW)",
    "P_PV2_kW":     r"$P_{pv}^{2}$ (kW)",
    "P_PV3_kW":     r"$P_{pv}^{3}$ (kW)",
    "T_env_C":      r"$\mathcal{T}_{env}$ ($^{\circ}$C)",
    "H_env_%":      r"$\mathcal{H}_{env}$ (%)",
    "V_wind_m/s":   r"$\mathcal{V}_{wind}$ (m/s)",
    "P_rain_mm/h":  r"$\mathcal{P}_{rain}$ (mm/h)",
    "D_env_g/m^3":  r"$\mathcal{D}_{env}$ (g/m$^3$)",
    "P_atm_hPa":    r"$\mathcal{P}_{atm}$ (hPa)",
}


def _prepare_hourly_pivot(df, column, dt_col="DT"):
    """Return a (date x hour) pivot table for a single variable.

    Rows = individual calendar days, columns = hour-of-day (0-23). This
    layout makes it trivial to draw one grey line per day and to compute
    the per-hour mean / min / max across all days.
    """
    tmp = df[[dt_col, column]].copy()
    tmp[dt_col] = pd.to_datetime(tmp[dt_col])
    tmp["_date"] = tmp[dt_col].dt.date
    tmp["_hour"] = tmp[dt_col].dt.hour
    pivot = tmp.pivot_table(index="_date", columns="_hour", values=column, aggfunc="mean")
    # Ensure all 24 hours are present even if some are missing for a day.
    pivot = pivot.reindex(columns=range(24))
    return pivot


def _prepare_yearly_hourly_pivot(df, column, dt_col="DT"):
    """Return a (year x hour) pivot table for a single variable.

    Rows = calendar year, columns = hour-of-day (0-23), values = the mean
    of `column` over every day of that year at that hour. This is what
    lets each year be drawn as its own diurnal-pattern line.
    """
    tmp = df[[dt_col, column]].copy()
    tmp[dt_col] = pd.to_datetime(tmp[dt_col])
    tmp["_year"] = tmp[dt_col].dt.year
    tmp["_hour"] = tmp[dt_col].dt.hour
    pivot = tmp.pivot_table(index="_year", columns="_hour", values=column, aggfunc="mean")
    pivot = pivot.reindex(columns=range(24))
    return pivot


def get_year_color_map(years, cmap_name="tab10"):
    """Build a consistent {year: color} mapping for a list of years.

    Exposed publicly so the same mapping can be reused across figures
    (e.g. so 2023 is the same color in every panel and every call).
    """
    years = sorted(set(years))
    cmap = plt.get_cmap(cmap_name)
    n_colors = getattr(cmap, "N", 10)
    return {y: cmap(i % n_colors) for i, y in enumerate(years)}


# Distinct dash patterns cycled across years. Combined with distinct colors,
# these keep the per-year lines separable even where they lie on top of one
# another: where two years coincide, the different dashes interleave so both
# remain visible (a solid-on-solid overlap would simply hide the lower line).
# DEFAULT_YEAR_LINESTYLES = [
#     "-",                                 # solid
#     (0, (5, 1.5)),                       # dashed
#     (0, (1, 1.3)),                       # dotted
#     (0, (6, 1.5, 1, 1.5)),               # dash-dot
#     (0, (3, 1.3, 1, 1.3, 1, 1.3)),       # dash-dot-dot
#     (0, (9, 2)),                         # long dash
# ]

DEFAULT_YEAR_LINESTYLES = [
    "-",                                 # solid
    "-",                                 # solid
    "-",                                 # solid
    "-",                                 # solid
    "-",                                 # solid
    "-",                                 # solid
]


def get_year_style_map(years, styles=None):
    """Build a consistent {year: linestyle} mapping (mirrors the color map).

    Ensures a given year keeps the same dash pattern in every panel and in
    the shared legend.
    """
    years = sorted(set(years))
    styles = list(styles) if styles else DEFAULT_YEAR_LINESTYLES
    return {y: styles[i % len(styles)] for i, y in enumerate(years)}


def remove_isolated_spikes(
    df, columns, dt_col="DT", factor=5.0, k_jump=4.0, min_jump=None,
    method="interpolate", verbose=True,
):
    """Remove isolated single-sample spikes (sensor glitches) from columns.

    Some channels (e.g. wind speed, rain rate) contain isolated one-hour
    spikes that are physically implausible -- a lone ~56 m/s reading between
    two ~0.5 m/s hours, or a lone ~11 mm/h reading between two dry hours.
    These blow up the Min/Max envelope and the y-axis while the real signal
    is "almost 0". This routine flags and repairs them.

    Detection (isolation test)
    --------------------------
    A sample x[i] is flagged as a spike when it exceeds the LARGER of its two
    temporal neighbours both:
        * by more than `factor` times            (x[i] > factor * max(neigh)), and
        * by an absolute margin `jump`           (x[i] - max(neigh) > jump).
    The absolute margin defaults to a data-adaptive value,
    `jump = k_jump * <99th percentile of the column>`, so the same call works
    for very different scales (m/s vs mm/h) without hand-tuning. Because both
    conditions must hold, sustained real events (whose neighbours are also
    elevated) are left untouched -- only lone spikes are caught.

    Parameters
    ----------
    df : DataFrame (will be time-sorted internally on `dt_col`).
    columns : list of column names to clean.
    dt_col : datetime column used to order samples for the neighbour test.
    factor : relative threshold (spike must exceed neighbours by this factor).
    k_jump : multiplier for the data-adaptive absolute margin (ignored if
        `min_jump` is given).
    min_jump : fixed absolute margin overriding the adaptive one, for all
        columns. Leave None to use `k_jump * q99` per column.
    method : how to repair flagged samples -- "interpolate" (linear, fills the
        one-hour gap ~= neighbour level; the default) or "nan" (mark missing).
    verbose : print a one-line report per column.

    Returns
    -------
    (cleaned_df, report) where `report[col]` is a dict with n_removed,
    max_before and max_after.
    """
    out = df.sort_values(dt_col).reset_index(drop=True).copy()
    report = {}
    for col in columns:
        s = out[col]
        jump = min_jump if min_jump is not None else k_jump * s.quantile(0.99)
        # larger of the two immediate neighbours (NaN at the series ends)
        neigh_max = pd.concat([s.shift(1), s.shift(-1)], axis=1).max(axis=1)
        mask = (s > neigh_max * factor) & ((s - neigh_max) > jump)
        n = int(mask.sum())
        info = {"n_removed": n, "max_before": float(s.max())}
        if n:
            if method == "nan":
                out.loc[mask, col] = np.nan
            else:  # linear interpolation across the flagged samples
                repaired = s.mask(mask).interpolate(limit_direction="both")
                out[col] = repaired
        info["max_after"] = float(out[col].max())
        report[col] = info
        if verbose:
            print(f"[despike] {col}: removed {n} isolated spike(s); "
                  f"max {info['max_before']:.2f} -> {info['max_after']:.2f}")
    return out, report


def plot_diurnal_pattern_single(
    ax, df, column, dt_col="DT", ylabel=None,
    show_individual_days=True, show_stats=True, show_years=False,
    day_line_color="0.6", day_line_alpha=0.35, mean_color="black",
    min_color="#1f77b4", max_color="#d62728", band_color="0.75",
    band_alpha=0.35, max_days_to_draw=400, year_colors=None,
    year_cmap="tab10", year_styles=None, year_linestyles=None,
    year_linewidth=1.4, year_alpha=0.8,
):
    """Draw a single diurnal-pattern panel onto an existing Axes.

    Parameters
    ----------
    ax : matplotlib Axes to draw into.
    df : DataFrame containing at least [dt_col, column].
    column : the variable to plot.
    dt_col : name of the datetime column.
    ylabel : y-axis label (defaults to DEFAULT_LABELS[column] or `column`).
    show_individual_days : draw the thin grey per-day traces.
    show_stats : draw the all-days Min/Max/Min-Max-range envelope (and the
        Mean line, but only when show_years=False -- see note below).
    show_years : draw one line per calendar year (each year's mean diurnal
        pattern). Lets you compare how the shape shifts year to year.
    max_days_to_draw : cap on individual day-lines drawn (randomly
        subsampled) to keep multi-year datasets fast and legible. None =
        draw every day.
    year_colors : optional {year: color} dict for consistent coloring
        across panels/figures (see get_year_color_map). If None, colors
        are assigned from `year_cmap`.
    """
    pivot = _prepare_hourly_pivot(df, column, dt_col=dt_col)
    hours = np.arange(24)

    # --- individual days (grey background lines) -----------------------
    if show_individual_days and len(pivot) > 0:
        draw_pivot = pivot
        if max_days_to_draw is not None and len(pivot) > max_days_to_draw:
            draw_pivot = pivot.sample(n=max_days_to_draw, random_state=0)
        ax.plot(hours, draw_pivot.T.values, color=day_line_color,
                alpha=day_line_alpha, linewidth=0.4, zorder=1)

    # --- all-days statistics (Min / Max / shaded range) ----------------
    if show_stats:
        mean_line = pivot.mean(axis=0, skipna=True)
        min_line = pivot.min(axis=0, skipna=True)
        max_line = pivot.max(axis=0, skipna=True)

        # shaded min-max band
        ax.fill_between(hours, min_line.values, max_line.values,
                        color=band_color, alpha=band_alpha, zorder=2,
                        label="Min-Max range")
        # min / max dashed lines (distinct colors so they're easy to tell apart)
        ax.plot(hours, max_line.values, color=max_color, linestyle="--",
                linewidth=1.0, zorder=3, label="Max")
        ax.plot(hours, min_line.values, color=min_color, linestyle="--",
                linewidth=1.0, zorder=3, label="Min")
    else:
        mean_line = None

    # --- per-year mean diurnal lines -----------------------------------
    # Each year gets both a distinct color AND a distinct dash pattern so
    # the near-coincident lines stay tellable apart (see DEFAULT_YEAR_LINESTYLES).
    if show_years:
        year_pivot = _prepare_yearly_hourly_pivot(df, column, dt_col=dt_col)
        years = list(year_pivot.index)
        if year_colors is None:
            year_colors = get_year_color_map(years, cmap_name=year_cmap)
        if year_styles is None:
            year_styles = get_year_style_map(years, year_linestyles)
        for y in years:
            ax.plot(hours, year_pivot.loc[y].values,
                    color=year_colors.get(y, "grey"),
                    linestyle=year_styles.get(y, "-"),
                    linewidth=year_linewidth, alpha=year_alpha,
                    zorder=5, label=str(y))

    # --- overall (all-days) Mean line ----------------------------------
    # Only drawn when the per-year lines are NOT shown: with show_years=True
    # the all-days mean is essentially the average of the per-year means, so
    # it would just overlap them. Drawn last (highest zorder) with a white
    # halo so it stands out against the grey daily cloud.
    if show_stats and not show_years:
        mean_artist, = ax.plot(hours, mean_line.values, color=mean_color,
                               linestyle="-", linewidth=1.8, zorder=10,
                               label="Mean", solid_capstyle="round")
        mean_artist.set_path_effects([pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])

    # --- axis cosmetics ------------------------------------------------
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day")
    if ylabel is None:
        ylabel = DEFAULT_LABELS.get(column, column)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.5)


def plot_diurnal_patterns(
    df, columns=None, dt_col="DT", ncols=3, figsize=(7.4, 6.2),
    show_individual_days=True, show_stats=True, show_years=False,
    max_days_to_draw=400, labels=None, min_color="#1f77b4",
    max_color="#d62728", year_colors=None, year_cmap="tab10",
    year_styles=None, year_linestyles=None, year_linewidth=1.4, year_alpha=0.8,
    suptitle=None, fig_caption=None, show=True, show_caption=False,
    stats_legend_title="Across all days", year_legend_title="Yearly mean",
    save_dir="../Export/Figure/Site-02/", save_name="diurnal_patterns",
    save_fig=False, missing=None,
):
    """Plot diurnal (hour-of-day) patterns for one or more variables.

    Produces a grid of subplots (one per variable) sized for an IEEE
    two-column-width figure, with two stacked single-row legend boxes at
    the top (all-days statistics, and per-year means).

    Parameters
    ----------
    df : DataFrame with a datetime column `dt_col` and one row per hour.
    columns : columns to plot. Defaults to every column except `dt_col`.
    dt_col : name of the datetime column (default 'DT').
    ncols : number of subplot columns in the grid.
    figsize : (width, height) in inches. Default (7.4, 6.2). Pass None to
        auto-size to IEEE double-column width with height scaled by rows.
    show_individual_days : draw the grey per-day traces.
    show_stats : draw the all-days Min/Max/Min-Max-range envelope (plus the
        Mean line when show_years=False).
    show_years : draw one line per calendar year (each year's mean diurnal
        pattern). Colors are consistent across all panels and the legend.
    max_days_to_draw : cap on grey day-lines per panel (subsampled).
    labels : optional {column: y-axis label} dict overriding DEFAULT_LABELS.
    min_color, max_color : colors for the Min / Max dashed lines.
    year_colors : optional {year: color} dict; built from `year_cmap` if None.
    year_cmap : colormap used to color years (default 'tab10', up to 10 years).
    year_styles : optional {year: linestyle} dict; built from `year_linestyles`
        if None. Giving each year a distinct dash pattern (on top of a distinct
        color) keeps the near-coincident year lines separable where they overlap.
    year_linestyles : optional list of dash patterns cycled across years
        (defaults to DEFAULT_YEAR_LINESTYLES: solid, dashed, dotted, dash-dot,
        dash-dot-dot, long-dash).
    year_linewidth : width of the per-year lines (default 1.4).
    year_alpha : transparency of the per-year lines (default 0.8) so overlaps
        show through rather than fully hiding the line underneath.
    suptitle : optional title drawn above the legends.
    fig_caption : caption text (e.g. 'Fig. 1. ...'). Shown below the figure
        only when show_caption=True; otherwise it is ignored.
    show : if True (default) call plt.show() at the end.
    show_caption : whether to render `fig_caption` below the figure.
    stats_legend_title : caption for the all-days statistics legend box.
    year_legend_title : caption for the per-year legend box.
    save_dir : directory for saved figures (created if missing).
    save_name : base filename WITHOUT extension. Keep this a plain token;
        put readable text in `fig_caption` instead.
    save_fig : master save switch. When True the figure is written as BOTH
        `<save_dir>/<save_name>.pdf` and `<save_dir>/<save_name>.png`.
    missing : optional list of history files that could not be found; if
        given, they are printed as a warning after saving.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Remove isolated spikes from the data
    df, report = remove_isolated_spikes(df, ['V_wind_m/s', 'P_rain_mm/h'])

    # Default to every column except the datetime column.
    if columns is None:
        columns = [c for c in df.columns if c != dt_col]

    n = len(columns)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    # Auto-size only when the caller explicitly passes figsize=None.
    if figsize is None:
        width = IEEE_TWO_COLUMN_WIDTH_IN
        height = width / ncols * nrows * 0.95
        figsize = (width, height)

    # Merge user label overrides on top of the built-in defaults.
    merged_labels = dict(DEFAULT_LABELS)
    if labels:
        merged_labels.update(labels)

    # Consistent year -> color and year -> linestyle maps, shared across
    # every panel so a given year looks identical in all of them.
    if show_years:
        all_years = pd.to_datetime(df[dt_col]).dt.year.unique()
        if year_colors is None:
            year_colors = get_year_color_map(all_years, cmap_name=year_cmap)
        if year_styles is None:
            year_styles = get_year_style_map(all_years, year_linestyles)

    with plt.rc_context(IEEE_RCPARAMS):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        # Draw one panel per variable.
        for i, column in enumerate(columns):
            plot_diurnal_pattern_single(
                axes_flat[i], df, column, dt_col=dt_col,
                ylabel=merged_labels.get(column, column),
                show_individual_days=show_individual_days,
                show_stats=show_stats, show_years=show_years,
                max_days_to_draw=max_days_to_draw, min_color=min_color,
                max_color=max_color, year_colors=year_colors, year_cmap=year_cmap,
                year_styles=year_styles, year_linewidth=year_linewidth,
                year_alpha=year_alpha,
            )

        # Hide any unused axes in the last row.
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        # ---------------------------------------------------------- legend
        from matplotlib.lines import Line2D

        # Collect the labeled artists from the first panel (all panels share
        # the same set of labels), then split them into the two groups.
        handles, labels_ = axes_flat[0].get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))

        stats_order = ["Mean", "Min", "Max", "Min-Max range"]
        stats_labels = [l for l in stats_order if l in by_label]
        year_labels = sorted((l for l in labels_ if l not in stats_order), key=lambda s: s)

        legend_kwargs = dict(frameon=True, facecolor="0.93", edgecolor="0.4",
                             framealpha=1.0, borderpad=0.5, handlelength=1.6,
                             handletextpad=0.5, columnspacing=1.2, fontsize=7)

        # Two single-row legend boxes, stacked and measured so they sit
        # snugly. The group name is carried by an INVISIBLE leading handle,
        # so it reads as an inline caption while every REAL entry keeps its
        # handle next to its own label (avoids the first marker being pulled
        # onto the group title).
        def _blank():
            return Line2D([], [], linestyle="none", marker="none")

        fig.canvas.draw()                       # a renderer is needed to measure legend height
        inv = fig.transFigure.inverted()
        y_cursor = 1.0                          # top of the figure (figure fraction)
        gap = 0.010                             # vertical gap between the two boxes

        def _add(lbls, group):
            """Add one single-row legend box below the previous one."""
            nonlocal y_cursor
            h = [_blank()] + [by_label[l] for l in lbls]
            t = [f"{group}:"] + list(lbls)
            leg = fig.legend(h, t, loc="upper center", ncol=len(t),
                             bbox_to_anchor=(0.5, y_cursor), **legend_kwargs)
            fig.add_artist(leg)
            fig.canvas.draw()
            hh = leg.get_window_extent().transformed(inv).height
            y_cursor = y_cursor - hh - gap      # move the cursor down for the next box
            return leg

        if stats_labels:
            _add(stats_labels, stats_legend_title)
        if year_labels:
            _add(year_labels, year_legend_title)

        # Reserve exactly the vertical space the legends actually used.
        top_margin = max(0.80, y_cursor - 0.005)

        if suptitle:
            fig.suptitle(suptitle, y=1.03)

        fig.tight_layout(rect=[0, 0, 1, top_margin])

        # Optional caption below the figure (independent of the filename).
        if show_caption and fig_caption is not None:
            fig.text(0.5, -0.01, fig_caption, ha="center", va="top",
                     fontsize=8, style="italic")

        # ------------------------------------------------------------ save
        # When enabled, write BOTH a vector PDF (for the paper) and a
        # high-res PNG (for quick viewing) using the same base name.
        if save_fig:
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.join(save_dir, save_name)
            fig.savefig(f"{base}.pdf")
            fig.savefig(f"{base}.png", dpi=600)
            print(f"Saved: {base}.pdf")
            print(f"Saved: {base}.png")

        # Optional warning about history files that were not found upstream.
        if missing:
            print("\n[!] Missing history files:")
            for m in missing:
                print(f"    - {m}")

        # Respect the `show` flag (no-op in notebooks, which auto-display).
        if show:
            plt.show()

        return fig


if __name__ == "__main__":
    # Simple smoke test / usage example when run directly.
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "Site_02_Data.gzip"

    Site_02_raw = pd.read_csv(path, compression="gzip")
    rename_columns = {
        "timestamp": "DT",
        "PV1_P_pv": "P_PV1_kW",
        "PV2_P_pv": "P_PV2_kW",
        "PV3_P_pv": "P_PV3_kW",
    }
    Site_02 = Site_02_raw.rename(columns=rename_columns)[[
        "DT", "P_PV1_kW", "P_PV2_kW", "P_PV3_kW", "T_env_C", "H_env_%",
        "V_wind_m/s", "P_rain_mm/h", "D_env_g/m^3", "P_atm_hPa",
    ]]

    plot_diurnal_patterns(
        Site_02,
        show_individual_days=True,
        show_stats=True,
        show_years=True,
        save_fig=True,
        save_dir="../Export/Figure/Site-02/",
        show_caption=True,
        fig_caption="Fig. 1. Diurnal patterns at Site 02 (2021-2026)",
        figsize=(7.4, 6.2),
        show=False,
    )