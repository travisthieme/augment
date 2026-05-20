"""
augment.stats
===========================

Purpose
-------
Core statistics and masking utilities used throughout
`augment`. This module provides a *single source of truth*
for which samples are included in plots and summaries, plus compact,
unit-aware summary statistics that match the numbers shown in figures.

Includes
--------
- **make_joint_mask(...)** — build one boolean mask that enforces finiteness
  for all requested variables and strict positivity for those intended to be
  displayed in log10 space.
- **make_joint_indices(...)** — return the joint mask and a (possibly)
  subsampled set of indices for performance-sensitive plots.
- **set_active_mask(...)** — persist the joint mask (and optional subsample)
  inside the `out` dict for reuse across all plots/formatters.
- **get_active_mask(...) / get_active_indices(...)** — retrieve the stored
  mask or subsampled indices if present.
- **resolve_mask(...)** — consistent mask resolution with clear precedence:
  explicit mask → active mask → freshly computed joint mask.
- **summarize_for_plot(...)** — “Option 2” summaries that agree with
  corner/posterior titles and formatted printouts (median with 16th/84th,
  in linear or log10 space, unit aware).

Conventions
-----------
- **Joint mask**: finite for all `keys`; strictly `> 0` for variables
  listed in `log_keys` (so `log10` is well-defined downstream).
- **Active mask**: stored in `out["_active_mask"]` as a dict with
  `mask`, `idx` (optional subsample), and the `keys/log_keys` used to
  build it. Plots call `resolve_mask(..., use_active=True)` to reuse it.
- **Mask precedence** (`resolve_mask`):
  1) explicit `mask` argument (highest priority),
  2) stored active mask (if `use_active=True`),
  3) freshly computed via `make_joint_mask(keys, log_keys)`; if `keys` are
     missing, falls back to `default_key` and `default_log10`.
- **Summaries (Option 2 semantics)**:
  - *log10 space*: filter to positive samples → take `log10` →
    report median and asymmetric errors `(q84−median, median−q16)` **in dex**.
    Also return **linear** `median/q16/q84` in `target_unit`.
  - *linear space*: (optionally) filter to positive samples and compute the
    same percentile triplet in linear units.
  - All summaries respect any mask passed in (or the active mask if used).
- **Units**: samples are converted to `target_unit` **before** filtering and
  percentile computation, ensuring consistent values across plots/prints.
- **Subsampling**: when `max_points` is provided, `make_joint_indices`
  selects a reproducible, without-replacement subset for speed while keeping
  the joint constraints intact.

Dependencies
------------
- numpy
- astropy

Author: Travis Thieme
Created: 2025-11-06
Last updated: 2025-11-06
"""

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
from astropy.units import Quantity
from astropy import units as u

def make_joint_mask(out: Dict, keys: Iterable[str], log_keys: Iterable[str]) -> np.ndarray:
    """
    Build a **joint boolean mask** over the sample index that enforces:
      • Finite values for every variable in `keys`
      • Strict positivity (> 0) for variables listed in `log_keys` (so that
        a log10 transform is safe later)

    Parameters
    ----------
    out : dict
        Sampler result dict containing `out["samples"][key]` for each variable,
        where each entry is an Astropy `Quantity` with shape (nsamples,).
    keys : Iterable[str]
        Variable names that must be **finite** simultaneously.
    log_keys : Iterable[str]
        Subset of `keys` that must also be **> 0**.

    Returns
    -------
    np.ndarray (bool, shape = nsamples)
        Joint mask selecting draws that pass all finite/positivity checks.

    Raises
    ------
    ValueError
        If the resulting mask has no `True` entries (i.e., no samples survive).

    Notes
    -----
    • This is the fundamental “consistent subset” definition used by all plots
      and formatters. Using it everywhere ensures that the same samples drive
      corner plots, 1-D posteriors, and printed summaries.
    """
    mask = None
    log_set = set(log_keys)
    for k in keys:
        q: Quantity = out["samples"][k]
        v = q.to_value(q.unit)
        good = np.isfinite(v)          # finite constraint
        if k in log_set:
            good &= (v > 0.0)          # positivity for log-variables
        mask = good if mask is None else (mask & good)
    if mask is None or mask.sum() == 0:
        raise ValueError("No samples left after applying the joint mask.")
    return mask

def make_joint_indices(out: Dict, keys: Iterable[str], log_keys: Iterable[str],
                       *, max_points: Optional[int] = None, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mask, idx) where idx may be a subsampled set of mask indices.

    Parameters
    ----------
    out, keys, log_keys : see `make_joint_mask`
    max_points : int or None, optional
        If provided and the number of `True` entries in the mask exceeds
        `max_points`, a **without-replacement** random subsample of that size
        is chosen for `idx`. If None, all passing indices are returned.
    seed : int, optional
        RNG seed for reproducible subsampling.

    Returns
    -------
    mask : np.ndarray (bool, shape = nsamples)
        The full joint mask.
    idx : np.ndarray (int, shape ≤ sum(mask))
        Either all indices where `mask` is True, or a subsampled selection.

    Notes
    -----
    • Subsampling speeds up heavy plots (e.g., corners) while keeping the joint
      constraints. Downstream code should always use `idx` to slice values that
      will be plotted, and may still keep `mask` for metadata.
    """
    mask = make_joint_mask(out, keys, log_keys)
    idx = np.where(mask)[0]
    if max_points is not None and idx.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    return mask, idx

def set_active_mask(out: Dict, keys: Iterable[str], log_keys: Iterable[str],
                    *, max_points: Optional[int] = None, seed: int = 42) -> np.ndarray:
    """
    Compute and **store** the active mask (and optional subsample) into `out`.
    Returns the boolean mask (length = nsamples).

    Side Effects
    ------------
    Creates/overwrites `out["_active_mask"]` with:
      {
        "mask": np.ndarray(bool, nsamples),   # full joint mask
        "idx":  np.ndarray(int, ≤ sum(mask)), # possibly-subsampled indices
        "keys": tuple(keys),
        "log_keys": tuple(log_keys),
      }

    Parameters
    ----------
    out, keys, log_keys : see `make_joint_indices`
    max_points, seed : see `make_joint_indices`

    Returns
    -------
    np.ndarray (bool)
        The full joint mask.

    Notes
    -----
    • Once an active mask is set, plotting utilities can call `resolve_mask(...,
      use_active=True)` and automatically reuse the same selection, ensuring
      that all figures and printed values agree exactly.
    """
    mask, idx = make_joint_indices(out, keys, log_keys, max_points=max_points, seed=seed)
    out["_active_mask"] = {"mask": mask, "idx": idx, "keys": tuple(keys), "log_keys": tuple(log_keys)}
    return mask

def get_active_mask(out: Dict) -> Optional[np.ndarray]:
    """Return the stored active mask if present, else None.

    Returns
    -------
    np.ndarray(bool) or None
        The mask stored by `set_active_mask`, or None if not set.

    Tip
    ---
    • Use this in formatters/printers to guarantee numerical agreement with
      figures that used the same active selection.
    """
    info = out.get("_active_mask")
    return None if info is None else info["mask"]

def get_active_indices(out: Dict) -> Optional[np.ndarray]:
    """Return the stored active indices (possibly subsampled) if present, else None.

    Returns
    -------
    np.ndarray(int) or None
        The `idx` array created by `make_joint_indices` and stored via
        `set_active_mask`. If no active mask was set, returns None.

    Notes
    -----
    • Prefer `idx` over `np.where(mask)[0]` in plotting to maintain any
      subsampling choice made in `set_active_mask`.
    """
    info = out.get("_active_mask")
    return None if info is None else info["idx"]

def resolve_mask(out: Dict, *,
                 mask: Optional[np.ndarray] = None,
                 keys: Optional[Iterable[str]] = None,
                 log_keys: Optional[Iterable[str]] = None,
                 default_key: Optional[str] = None,
                 default_log10: bool = False,
                 use_active: bool = True) -> np.ndarray:
    """
    Resolve which mask to use, following this priority:
      1) Use an explicit `mask` if provided.
      2) If `use_active=True` and an active mask is stored in `out`, use it.
      3) Otherwise, **compute a new joint mask** from `keys`/`log_keys`.
         If those are None, fall back to `default_key` plus `default_log10`.

    Parameters
    ----------
    out : dict
        Sampler result dict.
    mask : np.ndarray(bool) or None
        Explicit mask to use (highest priority).
    keys, log_keys : iterable or None
        Variables to enforce finite/positive constraints on when computing a
        new joint mask. See `make_joint_mask`.
    default_key : str or None
        If `keys` is None, use `(default_key,)` instead. Required if neither
        `mask` nor `keys` is provided.
    default_log10 : bool, optional
        If True and `log_keys` is None, set `log_keys = keys` (typical for
        plotting a single log-transformed variable).
    use_active : bool, optional
        If True, prefer any mask previously saved via `set_active_mask`.

    Returns
    -------
    np.ndarray(bool)
        The mask to use.

    Raises
    ------
    ValueError
        If neither `mask` nor `keys` nor `default_key` is sufficient to build
        a mask.

    Notes
    -----
    • This function centralizes mask resolution so that all plotting/printing
      code can simply call `resolve_mask(..., use_active=True)` and be confident
      that they’re using the same selection strategy.
    """
    if mask is not None:
        return mask
    if use_active:
        m = get_active_mask(out)
        if m is not None:
            return m
    if keys is None:
        if default_key is None:
            raise ValueError("resolve_mask: need mask or keys or default_key.")
        keys = (default_key,)
    if log_keys is None:
        log_keys = keys if default_log10 else ()
    return make_joint_mask(out, keys, log_keys)

def summarize_for_plot(
    out: Dict,
    key: str,
    *,
    target_unit: Optional[u.UnitBase] = None,
    space: str = "log10",
    positive_only: bool = True,
    use_precomputed: bool = True,
    mask: Optional[np.ndarray] = None,   # <— NEW
) -> Dict[str, object]:
    """
    Compute plot-ready **center and asymmetric errors** for a single variable,
    optionally in log10 space, and **consistent with the active/joint mask**.

    Summary definition (Option 2 behavior)
    --------------------------------------
    • In log space:
        – Filter to positive samples.
        – Take log10 of the filtered samples.
        – Return median and (q84 − median, median − q16) **in dex**.
        – Also return the corresponding **linear** median/q16/q84 in `target_unit`.
      This matches the corner plot titles and masked printouts.

    • In linear space:
        – Optionally filter to positive samples (`positive_only=True`).
        – Return median and asymmetric errors in linear units.

    Parameters
    ----------
    out : dict
        Sampler result dict with `out["samples"][key]` as an Astropy Quantity.
    key : str
        Variable name to summarize.
    target_unit : Unit or None, optional
        Unit to convert the samples to **before** any log/percentile operations.
        Defaults to the samples' native unit.
    space : {"log10", "linear"}, optional
        Whether to summarize in log10 space (dex) or linear space.
    positive_only : bool, optional
        If `space="linear"`, whether to drop non-positive samples before
        summarizing (avoids skew from ≤0 draws if desired).
    use_precomputed : bool, optional
        If True and `out["summary_log10_pos"][key]` exists (your precomputed
        Option-2 summary on positive draws), that entry is used to ensure exact
        agreement with stored results (after unit conversion).
    mask : np.ndarray(bool) or None, optional
        If provided, applied **after** unit conversion and finite filtering to
        guarantee the same sample subset used in figures.

    Returns
    -------
    dict
        For `space="log10"`:
            {
              "value": float,        # median in dex
              "err_minus": float,    # dex
              "err_plus": float,     # dex
              "unit": Unit,          # target_unit
              "median": Quantity,    # linear median (target_unit)
              "q16": Quantity,       # linear 16th percentile
              "q84": Quantity,       # linear 84th percentile
              "frac_nonpos": float,  # fraction of ≤0 samples (before log filtering)
            }
        For `space="linear"`:
            Same keys, but "value"/errors are in linear units.

    Raises
    ------
    ValueError
        If, after filtering and masking, there are no usable samples (e.g.,
        all non-positive for a log10 summary).

    Notes
    -----
    • `frac_nonpos` is computed **before** positivity filtering, useful to
      annotate diagnostics (e.g., “[f: xx.xx%]” in formatted outputs).
    • For log10 summaries, the returned linear percentiles are reconstructed
      from the log-space percentiles to keep both views in sync.
    """
    # 1) choose source: precomputed summary (if available) or raw samples
    samples: Quantity = out["samples"][key]
    unit0 = samples.unit
    tgt = target_unit or unit0

    # Convert samples first (important for log10)
    vals = samples.to_value(tgt)
    vals = vals[np.isfinite(vals)]
    if mask is not None:
        vals = vals[mask]  # <— enforce the same joint view
    if vals.size == 0:
        raise ValueError(f"No finite samples for {key}.")

    frac_nonpos = float(np.mean(vals <= 0.0)) if vals.size else 0.0

    if space.lower() == "log10":
        # Prefer precomputed Option-2 summary if present and unit matches
        if use_precomputed and "summary_log10_pos" in out and key in out["summary_log10_pos"]:
            # Precomputed entries are linear bounds; convert to target unit
            ent = out["summary_log10_pos"][key]
            med_q = ent["median"].to(tgt)
            q16_q = ent["q16"].to(tgt)
            q84_q = ent["q84"].to(tgt)
            # Convert to dex for plotting numbers
            med_log = float(np.log10(med_q.value))
            q16_log = float(np.log10(q16_q.value))
            q84_log = float(np.log10(q84_q.value))
            return {
                "value": med_log,
                "err_minus": med_log - q16_log,
                "err_plus": q84_log - med_log,
                "unit": tgt,
                "median": med_q,
                "q16": q16_q,
                "q84": q84_q,
                "frac_nonpos": frac_nonpos,
            }

        # Compute Option-2 directly from raw samples
        pos = vals[vals > 0.0]
        if pos.size == 0:
            raise ValueError(f"No positive samples for log10 summary of {key}.")
        logp = np.log10(pos)
        q16, q50, q84 = np.percentile(logp, [16.0, 50.0, 84.0])
        # Also return linear versions for convenience
        med_lin = (10.0**q50) * tgt
        q16_lin = (10.0**q16) * tgt
        q84_lin = (10.0**q84) * tgt
        return {
            "value": float(q50),
            "err_minus": float(q50 - q16),
            "err_plus": float(q84 - q50),
            "unit": tgt,
            "median": med_lin,
            "q16": q16_lin,
            "q84": q84_lin,
            "frac_nonpos": frac_nonpos,
        }

    # --- linear space path ---
    arr = vals
    if positive_only:
        arr = arr[arr > 0.0]
        if arr.size == 0:
            raise ValueError(f"No positive samples for linear summary of {key}.")
    q16, q50, q84 = np.percentile(arr, [16.0, 50.0, 84.0])
    return {
        "value": float(q50),
        "err_minus": float(q50 - q16),
        "err_plus": float(q84 - q50),
        "unit": tgt,
        "median": (q50) * tgt,
        "q16": (q16) * tgt,
        "q84": (q84) * tgt,
        "frac_nonpos": frac_nonpos,
    }

def print_aligned_summaries(out, keys, mask=None, *, lin_sigfigs=4, log_decimals=2, include_fraction=False):
    from astropy import units as u
    import numpy as np

    rows = []
    # First pass: gather summaries, build linear strings, collect widths for alignment
    for k in keys:
        m   = mask if mask is not None else resolve_mask(out, default_key=k, default_log10=True, use_active=True)
        lin = summarize_for_plot(out, k, space="linear", positive_only=True, mask=m)
        log = summarize_for_plot(out, k, space="log10",  positive_only=True, mask=m)

        # linear column (no units)
        lin_str = (f"{lin['value']:.{lin_sigfigs}e} "
                   f"(+{lin['err_plus']:.{lin_sigfigs}e}, -{lin['err_minus']:.{lin_sigfigs}e})")

        unit_str = "none" if u.dimensionless_unscaled.is_equivalent(log["unit"]) else log["unit"].to_string()

        # optional masked ≤0 fraction (of finite draws)
        fmask_str = ""
        if include_fraction:
            arr0   = out["samples"][k].to_value(out["samples"][k].unit)
            finite = np.isfinite(arr0)
            fmask  = 100.0 * np.sum((~m) & finite & (arr0 <= 0.0)) / max(1, np.sum(finite))
            fmask_str = f" | f: {fmask:5.2f}%"

        rows.append({
            "key": k, "m": m, "lin_str": lin_str, "log": log, "unit_str": unit_str, "fmask_str": fmask_str
        })

    # Widths for pretty alignment
    name_w = max(len(r["key"]) for r in rows)
    lin_w  = max(len(r["lin_str"]) for r in rows)
    unit_w = max(len(r["unit_str"]) for r in rows)

    # Decimal-alignment widths for LOG column (median, +err, -err)
    def intw_fmt(x, d, signed=False):
        s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
        return len(s.split('.')[0])
    val_iw = max(intw_fmt(r["log"]["value"],     log_decimals, signed=True)  for r in rows)
    ep_iw  = max(intw_fmt(r["log"]["err_plus"],  log_decimals, signed=False) for r in rows)
    em_iw  = max(intw_fmt(r["log"]["err_minus"], log_decimals, signed=False) for r in rows)

    # Helpers to align decimal points
    align_val = lambda x: (lambda s: f"{s.split('.')[0]:>{val_iw}}.{s.split('.')[1]}")(f"{x:+.{log_decimals}f}")
    align_err = lambda x,w: (lambda s: f"{s.split('.')[0]:>{w}}.{s.split('.')[1]}")(f"{x:.{log_decimals}f}")

    # Build log strings with aligned decimals, then compute width
    for r in rows:
        log = r["log"]
        r["log_str"] = (f"{align_val(log['value'])} "
                        f"(+{align_err(log['err_plus'], ep_iw)}, -{align_err(log['err_minus'], em_iw)})")
    log_w = max(len(r["log_str"]) for r in rows)

    # Print aligned rows
    for r in rows:
        print(f"{r['key']:<{name_w}} : {r['lin_str']:<{lin_w}} | "
              f"{r['log_str']:<{log_w}} [units: {r['unit_str']:<{unit_w}}]{r['fmask_str']}")