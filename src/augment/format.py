"""
augment.format
=======================

Purpose
-------
String-formatting utilities for sampler output. Centralizes how we print:
- asymmetric linear-space values: `c (+u, -l) unit`
- log10, positive-only summaries: `val (+up, -lo) unit`
- optional fraction of non-positive draws: `[f: xx.xx%]`

Notes
-----
- Use this module for ALL printed output so figures, logs, and notebooks
  stay consistent.
- Inverse units are rendered as `1 / cm2` when possible.

Dependencies
------------
- numpy
- astropy

Author: Travis Thieme
Created: 2025-11-06
Last updated: 2025-11-06
"""

from astropy import units as u
from astropy.units import Quantity
import numpy as np

__all__ = ["format_asym", "format_summary", "format_summary_table"]


def _fmt_sci(x: float, sigfigs: int = 4) -> str:
    """
    Format a float in scientific notation with a given number of significant figures.

    Parameters
    ----------
    x : float
        Value to format.
    sigfigs : int, optional
        Number of significant figures to display (default: 4).

    Returns
    -------
    str
        String like `'1.2345e+03'`.

    Notes
    -----
    - This helper is used by `format_asym` to keep numeric formatting consistent
      across central values and error terms.
    """
    return f"{x:.{sigfigs}e}"

def _unit_slash_str(unit: u.UnitBase) -> str:
    """
    Render units with all-negative powers in a "slash" form: e.g., `1 / cm2`.

    Parameters
    ----------
    unit : astropy.units.UnitBase
        Unit to render.

    Returns
    -------
    str
        A human-friendly unit string. If all powers are negative, returns the
        inverse-style `'1 / cm2 s'` representation; otherwise falls back to
        `unit.to_string()`.

    Examples
    --------
    >>> _unit_slash_str((u.cm**-2))
    '1 / cm2'
    >>> _unit_slash_str(u.K)
    'K'
    """
    # NOTE: Only stringification logic — no unit conversions happen here.
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

def format_asym(
    central_value: Quantity,
    upper_error: Quantity,
    lower_error: Quantity,
    *,
    sigfigs: int = 4,
    unit_style: str = "slash"
) -> str:
    """
    Format an asymmetric value with units: `c (+u, -l) unit`.

    Parameters
    ----------
    central_value : Quantity
        Central (median or best-fit) value with unit.
    upper_error : Quantity
        +error (same unit as `central_value`).
    lower_error : Quantity
        -error (same unit as `central_value`).
    sigfigs : int, optional
        Significant figures for scientific-notation formatting (default: 4).
    unit_style : {"slash", other}, optional
        If "slash", render inverse-like units as `'1 / cm2'`; otherwise use
        `unit.to_string()`.

    Returns
    -------
    str
        A string like `'2.561e+15 (+1.362e+15, -5.082e+14) 1 / cm2'`.

    Notes
    -----
    - No unit conversion is performed; values are reported in the input unit.
    - This function is unit-aware and will omit units if the value is dimensionless.
    """
    unit = central_value.unit
    c  = central_value.to_value(unit)  # keep everything in the same unit for consistent formatting
    up = upper_error.to_value(unit)
    lo = lower_error.to_value(unit)
    s  = f"{_fmt_sci(c, sigfigs)} (+{_fmt_sci(up, sigfigs)}, -{_fmt_sci(lo, sigfigs)})"
    if unit.is_equivalent(u.dimensionless_unscaled):
        return s
    ustr = _unit_slash_str(unit) if unit_style == "slash" else unit.to_string()
    return f"{s} {ustr}"

def format_summary(
    entry: dict,
    *,
    logspace: bool = False,
    samples: Quantity = None,          # REQUIRED when using mask or target_unit
    positive_only: bool = False,       # only for linear mode
    mask: np.ndarray = None,           # NEW: ensures consistency with corner
    target_unit: u.UnitBase = None,    # optional unit conversion before summarizing
    sigfigs: int = 3,                  # linear mode digits
    unit_style: str = "slash",
    decimals_value: int = 2,           # log mode digits
    decimals_error: int = 2,           # log mode digits
    include_fraction: bool = True,     # fraction of (masked) samples <= 0
    show_all_excluded: bool=False,
) -> str:
    """
    Format a sampler summary line that matches your plotting conventions.

    Two operating modes
    -------------------
    1) **Masked / unit-aware recomputation path** (recommended):
       Triggered when any of `mask`, `target_unit`, or (`logspace` and `samples`)
       is provided. In this path, the function:
         - Converts `samples` to `target_unit` (if given),
         - Applies `mask` (if given),
         - Computes the 16/50/84 percentiles **from the masked draws**,
         - Prints in either linear or log10 space.
       This guarantees exact agreement with figures that used the same mask.

    2) **Legacy 'entry-as-is' path**:
       If `logspace=False` and no `mask/target_unit` are provided, the function
       formats `entry["median"], entry["plus"], entry["minus"]` directly via
       `format_asym`. This preserves backward compatibility with precomputed
       summaries but will not reflect any active mask.

    Parameters
    ----------
    entry : dict
        A summary dict with keys `median`, `plus`, `minus` (Quantities).
        Used only in the legacy path (see above).
    logspace : bool, optional
        If True, compute 16/50/84 percentiles in log10 space on the **positive-only**
        masked draws, and print numbers in dex.
    samples : Quantity, optional
        Full per-draw samples for the variable being printed. **Required** if
        you pass a `mask`, request a `target_unit`, or set `logspace=True`.
    positive_only : bool, optional
        For linear mode only. If True, drop non-positive values before computing
        percentiles (helps avoid skew from zeros/negatives).
    mask : np.ndarray(bool), optional
        Boolean mask selecting the same subset of draws used in your plots.
    target_unit : Unit, optional
        Convert `samples` to this unit before computing percentiles.
    sigfigs : int, optional
        Significant figures for the linear mode (uses scientific notation).
    unit_style : {"slash", other}, optional
        How to print units (see `format_asym`).
    decimals_value : int, optional
        Digits after the decimal for the **median** in log mode.
    decimals_error : int, optional
        Digits after the decimal for the **errors** in log mode.
    include_fraction : bool, optional
        If True, append the fraction of masked samples that are ≤ 0 as
        `[f: xx.xx%]`. Useful for diagnosing zero-heavy posteriors.
    show_all_excluded : bool, optional
        If True, the fraction is for all masked values. 
        If False, the fraction is for masked values that were <=0.

    Returns
    -------
    str
        A formatted string suitable for logs, tables, or figure captions.

    Raises
    ------
    ValueError
        If required inputs are missing (e.g., `samples` not provided while
        requesting `logspace=True` or `mask/target_unit`), or if no finite /
        positive samples remain after masking.

    Examples
    --------
    Linear (masked) printing:
        >>> s = format_summary(
        ...     entry, logspace=False, samples=arrQ, mask=active_mask,
        ...     target_unit=u.cm**-2, positive_only=True, sigfigs=3
        ... )
        '2.580e+15 (+1.374e+15, -5.900e+14) 1 / cm2'

    Log10 (masked, with fraction):
        >>> s = format_summary(
        ...     entry, logspace=True, samples=arrQ, mask=active_mask,
        ...     decimals_value=2, decimals_error=2, include_fraction=True
        ... )
        '-15.67 (+0.62, -0.71) 1 / s [f: 36.68%]'
    """
    # ---- masked/target-unit path (recommended) ----
    if (mask is not None) or (target_unit is not None) or (logspace and samples is not None):
        if samples is None:
            raise ValueError("format_summary: `samples` is required when using mask/target_unit/logspace=True.")

        unit = target_unit or samples.unit
        arr0 = samples.to_value(unit)

        # finite baseline for robust denominators
        finite0 = np.isfinite(arr0)
        den = int(finite0.sum()) or 1

        # align / apply mask
        if mask is not None:
            if mask.shape != arr0.shape:
                raise ValueError("format_summary: `mask` must be same shape as `samples`.")
            applied_mask = (mask & finite0)
            if show_all_excluded:
                # fraction of finite draws we DROPPED via the mask
                frac_masked = 100.0 * float(np.sum(finite0 & (~mask))) / den
            else:
                # fraction of finite draws that were DROPPED *and* were non-positive
                dropped_nonpos = (~mask) & finite0 & (arr0 <= 0.0)
                frac_masked = 100.0 * float(np.sum(dropped_nonpos)) / den
        else:
            applied_mask = finite0
            frac_masked = 0.0

        arr = arr0[applied_mask]
        if arr.size == 0:
            raise ValueError("No finite samples to summarize after masking.")

        # ---- summaries ----
        if logspace:
            pos = arr[arr > 0.0]
            if pos.size == 0:
                raise ValueError("No positive samples available for log-space summary.")
            logp = np.log10(pos)
            q16, q50, q84 = np.percentile(logp, [16.0, 50.0, 84.0])
            base = f"{q50:.{decimals_value}f} (+{(q84-q50):.{decimals_error}f}, -{(q50-q16):.{decimals_error}f})"
            if not u.dimensionless_unscaled.is_equivalent(unit):
                base += " " + (_unit_slash_str(unit) if unit_style == "slash" else unit.to_string())
        else:
            vv = arr
            if positive_only:
                vv = vv[vv > 0.0]
                if vv.size == 0:
                    raise ValueError("No positive samples for linear summary with positive_only=True.")
            q16, q50, q84 = np.percentile(vv, [16.0, 50.0, 84.0])
            center = (q50) * unit
            minus  = (q50 - q16) * unit
            plus   = (q84 - q50) * unit
            base   = format_asym(center, plus, minus, sigfigs=sigfigs, unit_style=unit_style)

        if include_fraction:
            base += f" [f: {frac_masked:.2f}%]"
        return base

    # ---- legacy linear path (no mask/target_unit) ----
    if not logspace:
        return format_asym(entry["median"], entry["plus"], entry["minus"],
                           sigfigs=sigfigs, unit_style=unit_style)

    raise ValueError("format_summary(logspace=True) needs `samples=` (and optionally `mask=`) for consistency.")


def format_summary_table(
    out: dict,
    *,
    keys=None,
    log_keys=(),
    target_units=None,
    mask: np.ndarray = None,
    positive_only: bool = False,
    sigfigs: int = 3,
    unit_style: str = "slash",
    decimals_value: int = 2,
    decimals_error: int = 2,
    include_fraction: bool = False,
    value_align: str = "right",
    tablefmt: str = "markdown",
) -> str:
    """
    Format all or selected sampler outputs as a compact table.

    Parameters
    ----------
    out : dict
        Sampler output containing ``out["summary"]`` and, for log-space,
        masked, or unit-converted summaries, ``out["samples"]``.
    keys : iterable of str or None, optional
        Output keys to include. Defaults to all keys in ``out["summary"]``.
    log_keys : iterable of str, optional
        Keys to summarize in log10 space. These require ``out["samples"]``.
    target_units : dict or None, optional
        Mapping from key to target unit before summarizing.
    mask : np.ndarray(bool), optional
        Mask applied consistently to every row.
    positive_only : bool, optional
        For linear rows, drop non-positive samples before computing percentiles.
    sigfigs, unit_style, decimals_value, decimals_error
        Formatting controls for value and error columns.
    include_fraction : bool, optional
        If True, include an ``f`` column with the percentage of finite samples
        that are non-positive before any positive/log filtering. Requires
        ``out["samples"]`` for each included key.
    value_align : {"left", "right"}, optional
        Alignment for the ``Value`` column.
    tablefmt : {"markdown", "plain"}, optional
        Output table style.

    Returns
    -------
    str
        A formatted table suitable for printing or displaying in a notebook.
    """
    if keys is None:
        keys = tuple(out["summary"].keys())
    else:
        keys = tuple(keys)
    if value_align not in {"left", "right"}:
        raise ValueError("value_align must be 'left' or 'right'.")
    log_set = set(log_keys)
    target_units = target_units or {}
    samples = out.get("samples", {})

    def _unit_str(unit, *, logspace=False) -> str:
        base = "dimensionless" if unit.is_equivalent(u.dimensionless_unscaled) else (
            _unit_slash_str(unit) if unit_style == "slash" else unit.to_string()
        )
        return f"log10({base})" if logspace else base

    def _fmt_linear_value(value: float) -> str:
        return _fmt_sci(value, sigfigs)

    def _fmt_log_value(value: float) -> str:
        return f"{value:+.{decimals_value}f}"

    def _fmt_log_error(value: float) -> str:
        return f"{value:.{decimals_error}f}"

    def _sample_percentiles(key, sample, unit, *, logspace):
        arr0 = sample.to_value(unit)
        finite = np.isfinite(arr0)
        if mask is not None:
            if mask.shape != arr0.shape:
                raise ValueError("format_summary_table: `mask` must be same shape as each sample array.")
            finite &= mask
        arr = arr0[finite]
        if arr.size == 0:
            raise ValueError(f"No finite samples to summarize for {key!r}.")
        frac_nonpos = 100.0 * float(np.sum(arr <= 0.0)) / float(arr.size)
        if logspace:
            arr = arr[arr > 0.0]
            if arr.size == 0:
                raise ValueError(f"No positive samples available for log-space summary of {key!r}.")
            arr = np.log10(arr)
        elif positive_only:
            arr = arr[arr > 0.0]
            if arr.size == 0:
                raise ValueError(f"No positive samples available for linear summary of {key!r}.")
        q16, q50, q84 = np.percentile(arr, [16.0, 50.0, 84.0])
        return q50, q84 - q50, q50 - q16, frac_nonpos

    rows = []
    for key in keys:
        entry = out["summary"][key]
        needs_samples = (
            key in log_set
            or mask is not None
            or key in target_units
            or positive_only
            or include_fraction
        )
        sample = samples.get(key)
        if needs_samples and sample is None:
            raise ValueError(
                f"format_summary_table: samples for {key!r} are required. "
                "Run sampler.run(..., return_samples=True)."
            )
        unit = target_units.get(key)
        if unit is None:
            unit = sample.unit if sample is not None else entry["median"].unit

        if needs_samples:
            value, plus, minus, frac_nonpos = _sample_percentiles(
                key,
                sample,
                unit,
                logspace=key in log_set,
            )
        else:
            value = entry["median"].to_value(unit)
            plus = entry["plus"].to_value(unit)
            minus = entry["minus"].to_value(unit)
            frac_nonpos = 0.0

        if key in log_set:
            value_str = _fmt_log_value(value)
            err_str = f"+{_fmt_log_error(plus)}/-{_fmt_log_error(minus)}"
        else:
            value_str = _fmt_linear_value(value)
            err_str = f"+{_fmt_linear_value(plus)}/-{_fmt_linear_value(minus)}"

        row = {
            "parameter": key,
            "value": value_str,
            "error": err_str,
            "unit": _unit_str(unit, logspace=key in log_set),
            "fraction": f"{frac_nonpos:.2f}%",
        }
        rows.append(row)

    if tablefmt == "markdown":
        headers = ["Parameter", "Value", "+Error/-Error", "Unit"]
        if include_fraction:
            headers.append("f")
        separators = ["---", "---:" if value_align == "right" else "---", "---", "---"]
        if include_fraction:
            separators.append("---")
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(separators) + " |",
        ]
        for row in rows:
            cells = [f"`{row['parameter']}`", row["value"], row["error"], row["unit"]]
            if include_fraction:
                cells.append(row["fraction"])
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    if tablefmt == "plain":
        columns = [
            ("Parameter", "parameter"),
            ("Value", "value"),
            ("+Error/-Error", "error"),
            ("Unit", "unit"),
        ]
        if include_fraction:
            columns.append(("f", "fraction"))
        widths = {
            field: max(len(header), *(len(row[field]) for row in rows))
            for header, field in columns
        }
        lines = [
            "  ".join(
                f"{header:>{widths[field]}}" if field == "value" and value_align == "right"
                else f"{header:<{widths[field]}}"
                for header, field in columns
            ),
            "  ".join("-" * widths[field] for _, field in columns),
        ]
        lines.extend(
            "  ".join(
                f"{row[field]:>{widths[field]}}" if field == "value" and value_align == "right"
                else f"{row[field]:<{widths[field]}}"
                for _, field in columns
            )
            for row in rows
        )
        return "\n".join(lines)

    raise ValueError("tablefmt must be 'markdown' or 'plain'.")
