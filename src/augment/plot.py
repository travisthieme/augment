"""
augment.plot
==========================

Purpose
-------
Plotting helpers for visualizing Monte Carlo results from
`augment.core`. Includes:
- 1D posterior histograms (linear / log10)
- histogram-style KDE
- ECDF/CCDF
- corner-plot wrapper (using `corner`) with log10 + unit-aware labels

Conventions
-----------
- If `log10=True`, the joint mask enforces > 0 on keys in `log_keys`.
- Style: solid `tab:blue` median, dashed `tab:blue` for 16th/84th,
  black steps for the density outline.
- Prefer "1 / cm2" style for inverse units in labels (handled upstream).

Dependencies
------------
- numpy
- matplotlib
- astropy
- (optional) corner  (only for `plot_corner`)

Author: Travis Thieme
Created: 2025-11-06
Last updated: 2025-11-06
"""

from typing import Dict, Tuple, Optional
import numpy as np
from astropy.units import Quantity
from .stats import resolve_mask, get_active_indices

__all__ = ["plot_posterior", "plot_ecdf", "plot_corner"]

_DEFAULT_RANGE_QUANTILES = (0.01, 0.99)


def _as_vals(arr: Quantity):
    """
    Helper: convert an Astropy Quantity to a (values, unit) pair.
    If `arr` is already a bare ndarray, returns (arr, None).
    """
    unit = arr.unit if hasattr(arr, "unit") else None
    vals = arr.to_value(unit) if unit is not None else np.asarray(arr)
    return vals, unit


def _unit_slash_str(unit):
    """
    Best-effort 'slash' unit string for inverse-type units (e.g., '1 / cm2').
    Falls back to the unit's native string representation otherwise.
    """
    bases = getattr(unit, "bases", None)
    powers = getattr(unit, "powers", None)
    if bases is None or powers is None:
        return unit.to_string()
    if all(p < 0 for p in powers):
        parts = []
        for b, p in zip(bases, powers):
            p = abs(int(p))
            parts.append(f"{b.to_string()}{p if p != 1 else ''}")
        return "1 / " + " ".join(parts)
    return unit.to_string()


def _resolve_plot_range(data, plot_range=None, range_quantiles=None, *, label="data"):
    finite = np.asarray(data)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError(f"No finite plotted samples are available for {label!r}.")

    if plot_range is not None:
        try:
            lo, hi = plot_range
        except (TypeError, ValueError) as exc:
            raise ValueError(f"plot_range for {label!r} must be a (min, max) pair.") from exc
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"plot_range for {label!r} must be finite and ordered.")
        in_range = (finite >= lo) & (finite <= hi)
        if not np.any(in_range):
            raise ValueError(
                f"plot_range for {label!r} excludes all plotted samples. "
                f"Got range=({lo:.6g}, {hi:.6g}), but sample span is "
                f"({np.nanmin(finite):.6g}, {np.nanmax(finite):.6g})."
            )
        return (lo, hi)

    if range_quantiles is None:
        return None

    qlo, qhi = range_quantiles
    if not (0.0 <= qlo < qhi <= 1.0):
        raise ValueError("range_quantiles must be in ascending order within [0, 1].")
    lo, hi = np.nanquantile(finite, [qlo, qhi])
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
        return None
    return (lo, hi)


def _log_label(pretty: str, unit_text: str, *, bold_labels: bool) -> str:
    if unit_text:
        inner = (
            rf"\mathbf{{{pretty}}} / {unit_text}"
            if bold_labels
            else rf"{pretty} / {unit_text}"
        )
    else:
        inner = rf"\mathbf{{{pretty}}}" if bold_labels else pretty

    if bold_labels:
        return rf"\mathbf{{log}}_{{\mathbf{{10}}}}\mathbf{{(}}{inner}\mathbf{{)}}"
    return rf"\log_{{10}}({inner})"


def plot_posterior(
    out,
    *,
    key: str,
    log10: bool = True,
    kde: bool = False,  # NEW: single API switch
    # mask handling (consistent with corner)
    mask: Optional[np.ndarray] = None,
    keys: Optional[Tuple[str, ...]] = None,
    log_keys: Optional[Tuple[str, ...]] = None,
    # viz controls
    ax=None,
    bins: int = 200,
    plot_range: Optional[Tuple[float, float]] = None,
    range_quantiles: Optional[Tuple[float, float]] = _DEFAULT_RANGE_QUANTILES,
    smooth_sigma: Optional[float] = None,  # used when kde=True
    # ETI / HDI shading
    eti: Optional[Tuple[float, ...]] = (0.68, 0.95),
    hdi: Optional[Tuple[float, ...]] = None,  # back-compat; takes precedence if given
    eti_color: str = "tab:blue",
    eti_alpha_range: Tuple[float, float] = (0.6, 0.2),  # smallest→largest interval alpha
    # labeling / titles
    pretty_label: Optional[str] = None,
    unit_label_overrides=None,
    title: Optional[str] = None,
    title_fmt: str = ".2f",
    # styling (match corner)
    fontsize: int = "medium",
    tick_length: float = 4,
    tick_width: float = 1.6,
    spine_linewidth: float = 1.6,
    bold_labels: bool = True,
    show_all_ticklabels: bool = True,  # API symmetry
    ymin_zero: bool = True,
    show_legend: bool = True,
):
    """
    Unified 1D posterior plot with optional KDE-like smoothing and ETI shading.

    This function draws a one-dimensional posterior density for a single sampled
    variable `key`. It uses the *same* joint/active mask logic as the corner plot
    so that plotted samples and reported summary statistics remain consistent
    across figures and printed outputs.

    Parameters
    ----------
    out : dict
        The result dictionary returned by your sampler (must contain "samples"
        keyed by variable name, each an Astropy Quantity).
    key : str
        Name of the variable to plot.
    log10 : bool, optional
        If True, transform the selected samples via log10 before plotting.
        The joint mask enforces positivity for log-transformed variables.
    kde : bool, optional
        If True, apply Gaussian smoothing to the histogram for a smooth curve.
        If False, draw a step histogram outline.
    mask : np.ndarray, optional
        Boolean array selecting a subset of draws to plot. If None, a mask is
        resolved via `resolve_mask` (possibly using an active mask set earlier).
    keys, log_keys : tuple of str, optional
        Keys used to *construct* the joint mask (if `mask` is not provided).
        If omitted, defaults to `(key,)` and log_keys = keys if `log10=True`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new axes is created.
    bins : int, optional
        Number of histogram bins for the density estimation.
    plot_range : (float, float), optional
        Explicit x-axis range to use for the histogram and displayed limits.
    range_quantiles : (float, float) or None, optional
        If provided, compute the displayed x-axis range from these lower/upper
        quantiles. Defaults to the central 98% to prevent rare tail draws from
        flattening the visible posterior. Set to None to show the full span.
    smooth_sigma : float, optional
        Standard deviation (in *bin units*) of the Gaussian kernel used to
        smooth the histogram when `kde=True`. Ignored if `kde=False`.
    eti : tuple of float, optional
        Equal-tail intervals to shade (fractions in (0,1)), e.g., (0.68, 0.95).
    hdi : tuple of float, optional
        Backward-compatible synonym; if provided, takes precedence over `eti`.
    eti_color : str, optional
        Matplotlib color used for ETI shading and legend entry.
    eti_alpha_range : (float, float), optional
        (alpha for the *smallest* ETI, alpha for the *largest* ETI). If multiple
        ETIs are supplied, alphas are spaced linearly between these values.
    pretty_label : str, optional
        TeX-ready label to display on the x-axis (e.g., r"\\zeta_{\\rm CR}").
        If None, the raw `key` is used.
    unit_label_overrides : dict, optional
        Mapping key -> display-only LaTeX unit label. This changes labels/titles
        but does not convert plotted values.
    title : str, optional
        If provided, used verbatim as the axes title. If None, a corner-style
        numeric title (median with +/− 16/84th) is generated from RAW samples.
    title_fmt : str, optional
        Python format specifier used in the numeric title (e.g., ".2f").
    fontsize, tick_length, tick_width, spine_linewidth, bold_labels, show_all_ticklabels
        Styling knobs to match the look-and-feel of the corner plot.
    ymin_zero : bool, optional
        If True, force the y-axis lower bound to 0.0.
    show_legend : bool, optional
        Whether to show the legend (Median, 16/84th, ETI bands).

    Notes
    -----
    - Quantiles for median / 16th / 84th, and ETI bounds are computed directly
      from the masked, transformed sample vector to guarantee consistency with
      `format_summary` and `plot_corner`.
    - When `kde=True`, smoothing affects only the curve aesthetics; the reported
      quantiles are unaffected (still from RAW samples).
    """
    import matplotlib.pyplot as plt
    from astropy import units as u

    unit_label_overrides = unit_label_overrides or {}

    # Resolve which intervals to shade (ETI takes effect unless HDI provided)
    bands = hdi if (hdi is not None) else (eti if eti is not None else ())
    bands = tuple(sorted(float(b) for b in bands))  # e.g., (0.68, 0.95)

    # If the caller didn't specify keys, use the current key; log_keys toggles with log10
    if keys is None:
        keys = (key,)
    if log_keys is None:
        log_keys = keys if log10 else ()

    # Build or reuse a single source-of-truth mask (joint mask or stored active mask)
    mask = resolve_mask(
        out,
        mask=mask,
        keys=keys,
        log_keys=log_keys,
        default_key=key,
        default_log10=log10,
        use_active=True,
    )

    # Convert → mask → (optional) log-transform
    arr = out["samples"][key]
    unit = arr.unit
    v = arr.to_value(unit)[mask]
    x = np.log10(v) if log10 else v
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No samples to plot after masking (and log filtering).")

    x_range = _resolve_plot_range(x, plot_range, range_quantiles, label=key)

    # Create axes if needed
    if ax is None:
        _, ax = plt.subplots()

    # Histogram density (normalized), then optional smoothing for KDE-like look
    hist, edges = np.histogram(x, bins=int(bins), density=True, range=x_range)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    # Optional Gaussian smoothing (still only affects the curve)
    if kde and (smooth_sigma is not None) and (smooth_sigma > 0):
        half = int(np.ceil(4 * smooth_sigma))
        kx = np.arange(-half, half + 1, dtype=float)
        kern = np.exp(-0.5 * (kx / smooth_sigma) ** 2)
        kern /= kern.sum()
        hist = np.convolve(hist, kern, mode="same")

    # ---- Quantiles from RAW samples (consistent with corner & format_summary) ----
    def _percentiles(arr, ps=(16.0, 50.0, 84.0), method="linear"):
        try:
            return np.percentile(arr, ps, method=method)  # numpy >= 1.22
        except TypeError:
            return np.percentile(arr, ps)  # fallback

    q16, q50, q84 = _percentiles(x, (16.0, 50.0, 84.0), method="linear")

    def _latex_unit_str(unit, label_override=None) -> str:
        if label_override is not None:
            text = str(label_override).strip("$")
            if "\\" in text:
                return text
            if "_{" in text and text.endswith("}"):
                base, sub = text.split("_{", 1)
                return rf"\mathrm{{{base}}}_{{\mathrm{{{sub[:-1]}}}}}"
            if "^" in text or "_" in text or "{" in text:
                return rf"\mathrm{{{text}}}"
            return rf"\mathrm{{{text}}}"
        if unit is None or unit.is_equivalent(u.dimensionless_unscaled):
            return ""

        def _base_str(base):
            try:
                text = base.to_string("latex_inline")
            except Exception:
                text = base.to_string()
            text = text.strip("$")
            if text.startswith(r"\mathrm{") and text.endswith("}"):
                text = text[len(r"\mathrm{") : -1]
            return text

        bases = getattr(unit, "bases", None)
        powers = getattr(unit, "powers", None)
        if not bases or not powers:
            try:
                text = unit.to_string("latex_inline").strip("$")
            except Exception:
                text = unit.to_string()
            if text.startswith(r"\mathrm{") and text.endswith("}"):
                text = text[len(r"\mathrm{") : -1]
            return rf"\mathrm{{{text}}}"

        parts = []
        for base, power in zip(bases, powers):
            base_text = _base_str(base)
            power = int(power) if float(power).is_integer() else power
            if power == 1:
                parts.append(rf"\mathrm{{{base_text}}}")
            else:
                parts.append(rf"\mathrm{{{base_text}}}^{{{power}}}")
        return r"\,".join(parts)

    def _title_label(pretty: str, unit_text: str, *, logspace: bool) -> str:
        if logspace:
            return _log_label(pretty, unit_text, bold_labels=bold_labels)
        return rf"\mathbf{{{pretty}}}" if bold_labels else pretty

    def _axis_label(pretty: str, unit_text: str, *, logspace: bool) -> str:
        if logspace:
            return _title_label(pretty, unit_text, logspace=True)
        if unit_text:
            return (
                rf"\mathbf{{{pretty}}}\,({unit_text})"
                if bold_labels
                else rf"{pretty}\,({unit_text})"
            )
        return rf"\mathbf{{{pretty}}}" if bold_labels else pretty

    def _format_quantile_title(
        value: float, plus: float, minus: float, fmt: str, unit_text: str = ""
    ) -> str:
        formatted = format(value, fmt)
        if "e" not in formatted and "E" not in formatted:
            upper = format(plus, fmt)
            lower = format(minus, fmt)
            suffix = rf"\ {unit_text}" if unit_text else ""
            return rf"{formatted}^{{+{upper}}}_{{-{lower}}}{suffix}"

        mantissa, exponent_text = formatted.lower().split("e", 1)
        exponent = int(exponent_text)
        scale = 10.0**exponent
        decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
        center = f"{value / scale:.{decimals}f}"
        upper = f"{plus / scale:.{decimals}f}"
        lower = f"{minus / scale:.{decimals}f}"
        suffix = rf"\ {unit_text}" if unit_text else ""
        return rf"{center}^{{+{upper}}}_{{-{lower}}}\times 10^{{{exponent}}}{suffix}"

    # Outline: solid for KDE, step for histogram (unchanged)
    if kde:
        ax.plot(ctr, hist, color="tab:blue", linewidth=1.0, zorder=-2)
    else:
        ax.plot(ctr, hist, drawstyle="steps-mid", color="tab:blue", linewidth=1.2, zorder=-2)

    # Median & 16/84 lines from RAW samples (match corner/format_summary)
    ax.axvline(q50, color="black", linestyle="-", linewidth=1.0, label="Median")
    ax.axvline(q16, color="black", linestyle="--", linewidth=1.0, label="16th/84th Percentile")
    ax.axvline(q84, color="black", linestyle="--", linewidth=1.0)

    # --- ETI shading (use RAW-sample equal-tail bounds, anchored exactly) ---
    bands = hdi if (hdi is not None) else (eti if eti is not None else ())
    bands = tuple(sorted(float(b) for b in bands))
    if len(bands) > 0:
        a_hi, a_lo = eti_alpha_range
        alphas = [a_hi] if len(bands) == 1 else np.linspace(a_hi, a_lo, num=len(bands))

        # helper: raw-sample quantiles (same method as format_summary/corner)
        def _percentiles(arr, ps=(16.0, 50.0, 84.0), method="linear"):
            try:
                return np.percentile(arr, ps, method=method)  # numpy >= 1.22
            except TypeError:
                return np.percentile(arr, ps)

        for frac, a in zip(bands, alphas):
            a = float(np.clip(a, 0.0, 1.0))
            tail = (1.0 - float(frac)) / 2.0
            q_lo, q_hi = _percentiles(x, (100.0 * tail, 100.0 * (1.0 - tail)), method="linear")

            # interpolate density at the exact bounds so ETI edges align with the dashed lines
            y_lo = np.interp(q_lo, ctr, hist)
            y_hi = np.interp(q_hi, ctr, hist)

            # build polygon that starts/ends precisely at those quantiles
            inside = (ctr >= q_lo) & (ctr <= q_hi)
            x_fill = np.concatenate(([q_lo], ctr[inside], [q_hi]))
            y_fill = np.concatenate(([y_lo], hist[inside], [y_hi]))

            ax.fill_between(
                x_fill,
                0.0,
                y_fill,
                color=eti_color,
                alpha=a,
                label=f"{int(frac*100)}% ETI",
                zorder=-4,
            )

    # Pretty x-label like corner (TeX label if provided)
    name = pretty_label or key
    unit_text = _latex_unit_str(unit, unit_label_overrides.get(key))
    xlabel = rf"${_axis_label(name, unit_text, logspace=log10)}$"
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(
        "Density", fontweight=("bold" if bold_labels else None), fontsize=fontsize
    )  #  + (" [1/dex]" if log10 else "")

    # Corner-style numeric title from RAW-sample quantiles
    em, ep = (q50 - q16), (q84 - q50)
    if title is None:
        title_unit_text = "" if log10 else unit_text
        ax.set_title(
            rf"${_title_label(name, unit_text, logspace=log10)} = "
            rf"{_format_quantile_title(q50, ep, em, title_fmt, title_unit_text)}$",
            fontsize=fontsize,
        )
    else:
        ax.set_title(title)

    # ymin = 0 and inward ticks/spines for consistent styling
    if ymin_zero:
        top = ax.get_ylim()[1]
        ax.set_ylim(0.0, top)
    if x_range is not None:
        ax.set_xlim(*x_range)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=fontsize,
        length=tick_length,
        width=tick_width,
    )
    for sp in ax.spines.values():
        sp.set_linewidth(spine_linewidth)
    if show_legend:
        ax.legend(frameon=False, fontsize="small", borderpad=1.2, ncols=1)

    return ax


def plot_ecdf(
    out,
    *,
    key: str,
    log10: bool = True,
    pretty_label: Optional[str] = None,
    unit_label_overrides=None,
    mask: Optional[np.ndarray] = None,
    keys: Optional[Tuple[str, ...]] = None,
    log_keys: Optional[Tuple[str, ...]] = None,
    ax=None,
    ccdf: bool = False,
    # --- styling (match corner) ---
    ticklabelsize: int = 9,
    tick_length: float = 4,
    tick_width: float = 1.6,
    spine_linewidth: float = 1.6,
    bold_labels: bool = True,
    show_all_ticklabels: bool = True,  # API symmetry
):
    """
    Empirical CDF (or CCDF) plot for a single variable, using the same mask logic.

    Parameters
    ----------
    out : dict
        Result dictionary with a "samples" entry (Astropy Quantities).
    key : str
        Variable to display.
    log10 : bool, optional
        If True, apply log10 to the selected draws before computing the ECDF.
    pretty_label : str, optional
        TeX-ready label to display on the x-axis. If None, the raw `key` is used.
    unit_label_overrides : dict, optional
        Mapping key -> display-only LaTeX unit label. This changes labels/titles
        but does not convert plotted values.
    mask, keys, log_keys : see `plot_posterior`
        Control which draws are included (resolved via `resolve_mask`).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; created if None.
    ccdf : bool, optional
        If True, plot 1 − F(x) instead of F(x).
    ticklabelsize, tick_length, tick_width, spine_linewidth, bold_labels, show_all_ticklabels
        Styling controls to match the corner/posterior look.

    Notes
    -----
    - This is a distribution diagnostic complementary to the histogram/KDE plot.
    - ECDF is computed from the masked, transformed draws and is independent of
      binning choices.
    """
    import matplotlib.pyplot as plt
    from astropy import units as u

    unit_label_overrides = unit_label_overrides or {}

    if keys is None:
        keys = (key,)
    if log_keys is None:
        log_keys = keys if log10 else ()

    mask = resolve_mask(
        out,
        mask=mask,
        keys=keys,
        log_keys=log_keys,
        default_key=key,
        default_log10=log10,
        use_active=True,
    )

    arr = out["samples"][key]
    unit = arr.unit
    v = arr.to_value(unit)[mask]
    x = np.log10(v) if log10 else v
    x = np.sort(x)
    n = x.size
    if n == 0:
        raise ValueError("No samples to plot after masking (and log filtering).")

    y = np.linspace(1 / n, 1.0, n)
    if ccdf:
        y = 1.0 - y

    if ax is None:
        _, ax = plt.subplots()
    ax.step(x, y, where="post")

    def _latex_unit_str(unit, label_override=None) -> str:
        if label_override is not None:
            text = str(label_override).strip("$")
            if "\\" in text:
                return text
            if "_{" in text and text.endswith("}"):
                base, sub = text.split("_{", 1)
                return rf"\mathrm{{{base}}}_{{\mathrm{{{sub[:-1]}}}}}"
            if "^" in text or "_" in text or "{" in text:
                return rf"\mathrm{{{text}}}"
            return rf"\mathrm{{{text}}}"
        if unit is None or unit.is_equivalent(u.dimensionless_unscaled):
            return ""

        def _base_str(base):
            try:
                text = base.to_string("latex_inline")
            except Exception:
                text = base.to_string()
            text = text.strip("$")
            if text.startswith(r"\mathrm{") and text.endswith("}"):
                text = text[len(r"\mathrm{") : -1]
            return text

        bases = getattr(unit, "bases", None)
        powers = getattr(unit, "powers", None)
        if not bases or not powers:
            try:
                text = unit.to_string("latex_inline").strip("$")
            except Exception:
                text = unit.to_string()
            if text.startswith(r"\mathrm{") and text.endswith("}"):
                text = text[len(r"\mathrm{") : -1]
            return rf"\mathrm{{{text}}}"

        parts = []
        for base, power in zip(bases, powers):
            base_text = _base_str(base)
            power = int(power) if float(power).is_integer() else power
            if power == 1:
                parts.append(rf"\mathrm{{{base_text}}}")
            else:
                parts.append(rf"\mathrm{{{base_text}}}^{{{power}}}")
        return r"\,".join(parts)

    def _label(pretty: str, unit_text: str, *, logspace: bool) -> str:
        if logspace:
            return _log_label(pretty, unit_text, bold_labels=bold_labels)
        if unit_text:
            return (
                rf"\mathbf{{{pretty}}}\,({unit_text})"
                if bold_labels
                else rf"{pretty}\,({unit_text})"
            )
        return rf"\mathbf{{{pretty}}}" if bold_labels else pretty

    name = pretty_label or key
    unit_text = _latex_unit_str(unit, unit_label_overrides.get(key))
    ax.set_xlabel(
        rf"${_label(name, unit_text, logspace=log10)}$",
    )
    ax.set_ylabel("1 − F(x)" if ccdf else "F(x)", fontweight=("bold" if bold_labels else None))
    ax.set_title(rf"{'CCDF' if ccdf else 'ECDF'} of ${_label(name, unit_text, logspace=log10)}$")

    # Inward ticks & spine widths
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=ticklabelsize,
        length=tick_length,
        width=tick_width,
    )
    for sp in ax.spines.values():
        sp.set_linewidth(spine_linewidth)

    return ax


def plot_corner(
    out,
    *,
    keys: Optional[Tuple[str, ...]] = None,
    log_keys: Optional[Tuple[str, ...]] = None,
    unit_overrides=None,
    unit_label_overrides=None,
    label_map=None,
    max_points: Optional[int] = None,  # None = no subsampling
    seed: int = 42,
    bins: int = 40,
    smooth: float = 1.0,
    mask: Optional[np.ndarray] = None,
    plot_range=None,
    range_quantiles: Optional[Tuple[float, float]] = _DEFAULT_RANGE_QUANTILES,
    title_fmt: str = ".2f",
    shade_quantile_band: bool = True,
    shade_quantiles: Tuple[float, float] = (0.16, 0.84),
    shade_color: str = "tab:blue",
    shade_alpha: float = 0.25,
    # styling...
    fontsize="medium",
    tick_length=4,
    tick_width=1.6,
    spine_linewidth=1.6,
    bold_labels=True,
    show_all_ticklabels=True,
):
    """
    Corner plot (pairwise posteriors + 1D diagonals) with consistent masking.

    Parameters
    ----------
    out : dict
        Result dict with "samples" (Astropy Quantities for each key).
    keys : tuple of str or None, optional
        Variables to include (column order defines the grid). If None, all
        keys in ``out["samples"]`` are plotted in insertion order.
    log_keys : tuple of str or None, optional
        Subset of `keys` to be displayed in log10 space (positivity enforced).
        If None, no variables are log-transformed.
    unit_overrides : dict, optional
        Mapping key -> unit to cast samples before plotting (if needed).
    unit_label_overrides : dict, optional
        Mapping key -> display-only LaTeX unit label. This changes labels/titles
        but does not convert plotted values.
    label_map : dict, optional
        Mapping key -> TeX/pretty label. If absent, raw key is used.
    max_points : int, optional
        If provided, subsample the masked draws for speed. If an active mask
        with subsampled indices is already set upstream, that is respected.
    seed : int, optional
        RNG seed for any internal subsampling.
    bins, smooth : int, float
        Passed through to `corner.corner` for binning and smoothing.
    mask : np.ndarray, optional
        Explicit mask to use; otherwise resolved via `resolve_mask`.
    plot_range : list, optional
        Explicit axis ranges passed to `corner.corner`. Use one `(min, max)`
        pair per key.
    range_quantiles : (float, float), optional
        If provided, compute `plot_range` from these lower/upper quantiles for
        each plotted variable. Defaults to `(0.01, 0.99)`, framing the central
        98% of each marginal posterior so rare tail draws do not dominate the
        panel limits. Set to None to show the full span.
    title_fmt : str, optional
        Format string for diagonal titles (corner’s native titles).
    shade_quantile_band : bool, optional
        If True, fill the diagonal histograms between `shade_quantiles`.
    shade_quantiles : (float, float), optional
        Lower/upper quantiles to shade on the diagonal (fractions in [0,1]).
    shade_color, shade_alpha : optional
        Color and alpha used for the shaded quantile band.
    fontsize, tick_length, tick_width, spine_linewidth, bold_labels, show_all_ticklabels
        Styling knobs for a compact, publication-ready layout.

    Notes
    -----
    - Titles on the diagonal are generated by `corner` from the plotted
      samples; if you set an active mask upstream, the titles and contours
      remain consistent with all other plots and printed summaries.
    - If you want a *single* mask to drive multiple figures, call your
      `set_active_mask(out, keys, log_keys, ...)` once before plotting.
    """
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from astropy import units as u

    rng = np.random.default_rng(seed)
    S = out["samples"]
    if keys is None:
        keys = tuple(S.keys())
    else:
        keys = tuple(keys)
    if log_keys is None:
        log_keys = ()
    else:
        log_keys = tuple(log_keys)
    label_map = label_map or {}
    unit_overrides = unit_overrides or {}
    unit_label_overrides = unit_label_overrides or {}
    units = {k: (unit_overrides.get(k) or S[k].unit) for k in keys}

    # One joint mask for everything (or reuse active one)
    mask = resolve_mask(out, mask=mask, keys=keys, log_keys=log_keys, use_active=True)

    # Prefer stored subsample indices (if an active mask with subsampling was set upstream)
    idx = get_active_indices(out)
    if idx is None:
        idx = np.where(mask)[0]

    def _latex_unit_str(unit, label_override=None) -> str:
        if label_override is not None:
            text = str(label_override).strip("$")
            if "\\" in text:
                return text
            if "_{" in text and text.endswith("}"):
                base, sub = text.split("_{", 1)
                return rf"\mathrm{{{base}}}_{{\mathrm{{{sub[:-1]}}}}}"
            if "^" in text or "_" in text or "{" in text:
                return rf"\mathrm{{{text}}}"
            return rf"\mathrm{{{text}}}"
        if unit is None or unit.is_equivalent(u.dimensionless_unscaled):
            return ""

        def _base_str(base):
            try:
                text = base.to_string("latex_inline")
            except Exception:
                text = base.to_string()
            text = text.strip("$")
            if text.startswith(r"\mathrm{") and text.endswith("}"):
                text = text[len(r"\mathrm{") : -1]
            return text

        bases = getattr(unit, "bases", None)
        powers = getattr(unit, "powers", None)
        if not bases or not powers:
            try:
                text = unit.to_string("latex_inline").strip("$")
            except Exception:
                text = unit.to_string()
            if text.startswith(r"\mathrm{") and text.endswith("}"):
                text = text[len(r"\mathrm{") : -1]
            return rf"\mathrm{{{text}}}"

        parts = []
        for base, power in zip(bases, powers):
            base_text = _base_str(base)
            power = int(power) if float(power).is_integer() else power
            if power == 1:
                parts.append(rf"\mathrm{{{base_text}}}")
            else:
                parts.append(rf"\mathrm{{{base_text}}}^{{{power}}}")
        return r"\,".join(parts)

    def _title_label(pretty: str, unit_text: str, *, logspace: bool) -> str:
        if logspace:
            return _log_label(pretty, unit_text, bold_labels=bold_labels)
        else:
            label = pretty
        return rf"\mathbf{{{label}}}" if bold_labels else label

    def _axis_label(pretty: str, unit_text: str, *, logspace: bool) -> str:
        if logspace:
            return _title_label(pretty, unit_text, logspace=True)
        if unit_text:
            return (
                rf"\mathbf{{{pretty}}}\,({unit_text})"
                if bold_labels
                else rf"{pretty}\,({unit_text})"
            )
        return rf"\mathbf{{{pretty}}}" if bold_labels else pretty

    # Build data matrix and labels using SAME indices
    cols, labels, title_labels = [], [], []
    for k in keys:
        v = S[k].to_value(units[k])[idx]
        pretty = label_map.get(k, k)
        unit_text = _latex_unit_str(units[k], unit_label_overrides.get(k))
        if k in log_keys:
            v = np.log10(v)
            labels.append(rf"${_axis_label(pretty, unit_text, logspace=True)}$")
        else:
            labels.append(rf"${_axis_label(pretty, unit_text, logspace=False)}$")
        title_labels.append(_title_label(pretty, unit_text, logspace=k in log_keys))
        cols.append(v)
    data = np.vstack(cols).T

    if plot_range is None and range_quantiles is not None:
        plot_range = []
        for i in range(data.shape[1]):
            plot_range.append(
                _resolve_plot_range(data[:, i], None, range_quantiles, label=keys[i])
            )

    if isinstance(plot_range, dict):
        plot_range = [plot_range.get(k) for k in keys]

    if plot_range is not None:
        if len(plot_range) != len(keys):
            raise ValueError("plot_range must contain one range per key.")
        plot_range = list(plot_range)
        for i, (k, rng_i) in enumerate(zip(keys, plot_range)):
            if rng_i is None:
                continue
            try:
                plot_range[i] = _resolve_plot_range(data[:, i], rng_i, None, label=k)
            except ValueError as exc:
                transform = "log10 " if k in log_keys else ""
                raise ValueError(
                    f"{exc} The plotted column contains {transform}sample values."
                ) from exc

    def _format_quantile_title(x: np.ndarray, fmt: str, unit_text: str = "") -> str:
        q16, q50, q84 = np.nanpercentile(x, [16.0, 50.0, 84.0])
        plus = q84 - q50
        minus = q50 - q16
        formatted = format(q50, fmt)
        if "e" not in formatted and "E" not in formatted:
            upper = format(plus, fmt)
            lower = format(minus, fmt)
            suffix = rf"\ {unit_text}" if unit_text else ""
            return rf"{formatted}^{{+{upper}}}_{{-{lower}}}{suffix}"

        mantissa, exponent_text = formatted.lower().split("e", 1)
        exponent = int(exponent_text)
        scale = 10.0**exponent
        decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
        center = f"{q50 / scale:.{decimals}f}"
        upper = f"{plus / scale:.{decimals}f}"
        lower = f"{minus / scale:.{decimals}f}"
        suffix = rf"\ {unit_text}" if unit_text else ""
        if exponent == 0:
            return rf"{center}^{{+{upper}}}_{{-{lower}}}{suffix}"
        return rf"{center}^{{+{upper}}}_{{-{lower}}}\times 10^{{{exponent}}}{suffix}"

    fig = corner.corner(
        data,
        labels=labels,
        bins=bins,
        smooth=smooth,
        quantiles=[0.16, 0.50, 0.84],
        range=plot_range,
        show_titles=False,
        title_fmt=title_fmt,
        color="tab:blue",
        scale_hist=False,
        plot_datapoints=False,
        fill_contours=True,
        contour_kwargs={"colors": "black", "linewidths": 0.8},
        label_kwargs={"fontsize": fontsize, "fontweight": "normal"},
        title_kwargs={"fontsize": fontsize},
    )

    def _fill_under_hist_steps(ax, edges, hist, qlo, qhi, *, color, alpha, zorder, scale=1.0):
        if hist.size == 0 or edges.size < 2:
            return False
        y0 = 0.0
        ymin = ax.get_ylim()[0]
        if ymin > 0.0:
            y0 = ymin
        added = False
        for left, right, height in zip(edges[:-1], edges[1:], hist):
            if height <= 0:
                continue
            if right <= qlo or left >= qhi:
                continue
            ol = max(left, qlo)
            oright = min(right, qhi)
            if oright <= ol:
                continue
            h = height * scale
            rect = Rectangle(
                (ol, y0),
                oright - ol,
                h,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=zorder,
                clip_on=True,
            )
            ax.add_patch(rect)
            added = True
        return added

    def _fill_under_step_line(ax, xd, yd, qlo, qhi, *, color, alpha, zorder, scale=1.0):
        xd = np.asarray(xd)
        yd = np.asarray(yd)
        if xd.size < 2 or yd.size < 2:
            return False
        added = False
        for i in range(len(xd) - 1):
            x0 = xd[i]
            x1 = xd[i + 1]
            if x1 == x0:
                continue  # vertical segment
            yv = yd[i]
            if yv <= 0:
                continue
            left = min(x0, x1)
            right = max(x0, x1)
            if right <= qlo or left >= qhi:
                continue
            ol = max(left, qlo)
            oright = min(right, qhi)
            if oright <= ol:
                continue
            h = yv * scale
            rect = Rectangle(
                (ol, 0.0),
                oright - ol,
                h,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=zorder,
                clip_on=True,
            )
            ax.add_patch(rect)
            added = True
        return added

    def _fill_under_step_vertices(ax, verts, qlo, qhi, *, color, alpha, zorder):
        verts = np.asarray(verts)
        if verts.size == 0:
            return False
        added = False
        for (x0, y0), (x1, y1) in zip(verts[:-1], verts[1:]):
            if y0 != y1:
                continue  # only horizontal segments
            if y0 <= 0:
                continue
            left = min(x0, x1)
            right = max(x0, x1)
            if right <= qlo or left >= qhi:
                continue
            ol = max(left, qlo)
            oright = min(right, qhi)
            if oright <= ol:
                continue
            rect = Rectangle(
                (ol, 0.0),
                oright - ol,
                y0,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=zorder,
                clip_on=True,
            )
            ax.add_patch(rect)
            added = True
        return added

    # Style diagonal quantile lines (from corner): dashed black, median solid
    medians = np.percentile(data, 50.0, axis=0)
    for ax in fig.axes:
        spec = ax.get_subplotspec()
        if spec is None or spec.rowspan.start != spec.colspan.start:
            continue
        i = spec.rowspan.start
        if i < data.shape[1]:
            unit_text = (
                ""
                if keys[i] in log_keys
                else _latex_unit_str(
                    units[keys[i]],
                    unit_label_overrides.get(keys[i]),
                )
            )
            ax.set_title(
                rf"${title_labels[i]} = {_format_quantile_title(data[:, i], title_fmt, unit_text)}$",
                fontsize=fontsize,
            )
        # Quantile lines are vertical (constant x)
        for line in ax.lines:
            xd = line.get_xdata()
            if len(xd) > 1 and np.allclose(xd, xd[0]):
                x0 = float(xd[0])
                is_median = (i < medians.size) and np.isclose(x0, medians[i], rtol=1e-6, atol=1e-9)
                line.set(color="black", linestyle="-" if is_median else "--", linewidth=1.0)

        # Optional: shade quantile band under the diagonal histogram
        if shade_quantile_band and (i < data.shape[1]):
            x = data[:, i]
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            qlo, qhi = np.nanquantile(x, shade_quantiles)
            if not (np.isfinite(qlo) and np.isfinite(qhi)) or qlo == qhi:
                continue
            # Shade under-histogram area using the same binning as the diagonal
            if isinstance(bins, (list, tuple, np.ndarray)):
                bins_i = bins[i] if len(bins) > i else bins[0]
            else:
                bins_i = bins
            hist, edges = np.histogram(x, bins=int(bins_i), density=True)
            bin_mask = (edges[:-1] < qhi) & (edges[1:] > qlo)
            ymin, ymax = ax.get_ylim()
            drew = False
            if np.any(bin_mask):
                # Prefer the actual diagonal histogram artist for correct scaling
                line_z = None
                source = "hist"
                xd = yd = None

                for line in ax.lines:
                    xd = np.asarray(line.get_xdata())
                    if len(xd) > 1 and not np.allclose(xd, xd[0]):
                        yd = np.asarray(line.get_ydata())
                        line_z = line.get_zorder()
                        source = "line"
                        break

                # Try polygon-style histogram patch (histtype="step")
                if source != "line":
                    best_n = 0
                    best_verts = None
                    best_z = None
                    for patch in ax.patches:
                        if isinstance(patch, Rectangle):
                            continue
                        if not hasattr(patch, "get_path"):
                            continue
                        verts = patch.get_path().vertices
                        if verts is None or len(verts) < 6:
                            continue
                        if len(verts) > best_n:
                            best_n = len(verts)
                            best_verts = verts
                            best_z = patch.get_zorder()
                    if best_verts is not None:
                        line_z = best_z
                        source = "patch"

                # Keep fill above the axes patch but below the histogram line
                base_z = (line_z - 0.1) if line_z is not None else 2.0
                z = max(base_z, 1.1)

                if source == "line":
                    drew = _fill_under_step_line(
                        ax,
                        xd,
                        yd,
                        qlo,
                        qhi,
                        color=shade_color,
                        alpha=shade_alpha,
                        zorder=z,
                        scale=1.0,
                    )
                elif source == "patch":
                    drew = _fill_under_step_vertices(
                        ax, best_verts, qlo, qhi, color=shade_color, alpha=shade_alpha, zorder=z
                    )
                else:
                    x_step = edges
                    y_step = np.r_[hist, hist[-1]]
                    drew = _fill_under_step_line(
                        ax,
                        x_step,
                        y_step,
                        qlo,
                        qhi,
                        color=shade_color,
                        alpha=shade_alpha,
                        zorder=z,
                        scale=1.0,
                    )

    # Styling & compact spacing
    for ax in fig.axes:
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            labelsize=fontsize,
            length=tick_length,
            width=tick_width,
            pad=0.5,
        )
        for sp in ax.spines.values():
            sp.set_linewidth(spine_linewidth)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.95, wspace=0.05, hspace=0.05)
    return fig
