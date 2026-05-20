"""
augment.io
=======================

Purpose
-------
I/O helpers for configuring samplers and exporting results. This module
centralizes:
- Loading variable specifications from JSON/CSV into `VariableSpec`s
  (and optionally registering them onto an `Sampler`).
- Saving summaries, error budgets, and raw samples to disk in simple,
  human- and machine-readable formats.

Includes
--------
- specs_from_dicts(...)   → build `VariableSpec`s from a list of dicts
- load_specs_json(...)    → read list of variable dicts from a JSON file
- load_specs_csv(...)     → read list of variable dicts from a CSV file
- save_summary_csv(...)   → write the `out["summary"]` dict to a CSV
- save_error_budget_csv(...) → write `out["error_budget"]` to a CSV
- save_samples_npz(...)   → write `out["samples"]` (Quantities) to a .npz
                            with unit metadata preserved

Conventions
-----------
- Quantity parsing accepts either true Astropy `Quantity`, a numeric + unit
  string pair, or a single string like `"1.23 1 / cm2"`.
- When saving summaries to CSV, values are emitted as `"value unit"` strings.
- The `.npz` samples file contains two metadata arrays:
  `__keys__` (variable names) and `__units__` (string units), aligned by index.

Dependencies
------------
- json, csv, numpy
- astropy.units
- local: VariableSpec, Sampler

Author: Travis Thieme
Created: 2025-11-06
Last updated: 2025-11-06
"""

from typing import List, Dict, Optional
import csv, json
from astropy import units as u
from astropy.units import Quantity
from .core import VariableSpec, Sampler

__all__ = [
    "specs_from_dicts",
    "load_specs_json",
    "load_specs_csv",
    "save_summary_csv",
    "save_error_budget_csv",
    "save_samples_npz",
]


def _q(val, unit_str: Optional[str]) -> Quantity:
    """
    Internal helper: coerce various inputs into an Astropy `Quantity`.

    Accepted forms
    --------------
    - Numeric + explicit unit string:
        `_q(1.23, "cm-2")` → `<Quantity 1.23 1 / cm2>`
    - Single string with value and unit (separated by whitespace):
        `_q("1.23 1 / cm2", None)` → `<Quantity 1.23 1 / cm2>`
    - Already a `Quantity`: returned unchanged.

    Parameters
    ----------
    val : float|int|str|Quantity
        The value to convert.
    unit_str : str or None
        Unit string when `val` is numeric. Ignored if `val` is a string/Quantity.

    Returns
    -------
    Quantity
        Value with unit.

    Raises
    ------
    ValueError
        If `val` is numeric and `unit_str` is missing; or if a string is
        given without a unit token.
    TypeError
        If the input type is not supported.

    Notes
    -----
    - This function does **not** attempt unit compatibility checks beyond
      constructing a `Unit` from the provided string. Such checks are deferred
      to `VariableSpec.validate()`.
    """
    if isinstance(val, (int, float)):
        if unit_str is None:
            raise ValueError("Unit string required when passing numeric values.")
        return val * u.Unit(unit_str)
    if isinstance(val, str):
        # allow "1.23 1 / cm2" — split on first whitespace
        parts = val.strip().split(None, 1)
        if len(parts) == 1:
            raise ValueError("String quantity must include a unit, e.g. '1.23 cm-2'.")
        num = float(parts[0])
        return num * u.Unit(parts[1])
    if hasattr(val, "unit"):
        return val
    raise TypeError("Unsupported quantity format.")


def specs_from_dicts(dicts: List[Dict], *, sampler: Optional[Sampler] = None) -> List[VariableSpec]:
    """
    Build `VariableSpec` objects from a list of dictionaries. Optionally,
    register each spec onto an existing sampler.

    Expected dictionary keys
    ------------------------
    Required:
      - "name": str
      - "central_value": Quantity|float|str
      - "upper_error":   Quantity|float|str
      - "lower_error":   Quantity|float|str
    Optional (sampling behavior):
      - "dist": {"split_normal","normal","lognormal"}   (default: "split_normal")
      - "frac": bool   (treat errors as fractional if True)
      - "lower_bound": Quantity|float|str
      - "upper_bound": Quantity|float|str
    Optional (units when values are numeric):
      - "central_unit": str
      - "error_unit":   str  (if omitted and `frac=False`, falls back to central_unit)

    Parameters
    ----------
    dicts : list of dict
        Variable specifications.
    sampler : Sampler or None, optional
        If provided, each spec is immediately added to this sampler via
        `sampler.add(...)` after validation.

    Returns
    -------
    list[VariableSpec]
        All constructed and validated specifications.

    Raises
    ------
    KeyError / ValueError / TypeError
        For missing required fields, invalid unit strings, or incompatible
        types. Unit compatibility is ultimately enforced by `spec.validate()`.

    Notes
    -----
    - When `frac=True`, error units must be dimensionless. This helper passes
      through the values; `VariableSpec.validate()` enforces the rule.
    """
    specs = []
    for d in dicts:
        name = d["name"]
        central = d.get("central_value")
        upper = d.get("upper_error")
        lower = d.get("lower_error")
        dist = d.get("dist", "split_normal")
        frac = bool(d.get("frac", False))
        lower_b = d.get("lower_bound")
        upper_b = d.get("upper_bound")

        # units (optional if quantities provided)
        cu = d.get("central_unit")
        eu = d.get("error_unit")

        central_q = _q(central, cu) if not hasattr(central, "unit") else central
        upper_q = (
            _q(upper, eu if eu is not None else (cu if not frac else None))
            if not hasattr(upper, "unit")
            else upper
        )
        lower_q = (
            _q(lower, eu if eu is not None else (cu if not frac else None))
            if not hasattr(lower, "unit")
            else lower
        )

        lower_bq = (
            _q(lower_b, cu) if (lower_b is not None and not hasattr(lower_b, "unit")) else lower_b
        )
        upper_bq = (
            _q(upper_b, cu) if (upper_b is not None and not hasattr(upper_b, "unit")) else upper_b
        )

        spec = VariableSpec(
            name=name,
            central_value=central_q,
            upper_error=upper_q,
            lower_error=lower_q,
            dist=dist,
            lower=lower_bq,
            upper=upper_bq,
            frac=frac,
        )
        spec.validate()
        specs.append(spec)
        if sampler is not None:
            # Mirror the spec registration on the sampler so downstream `.run(...)`
            # has everything it needs without separate manual `add(...)` calls.
            sampler.add(
                name,
                central_q,
                upper_error=upper_q,
                lower_error=lower_q,
                dist=dist,
                lower=lower_bq,
                upper=upper_bq,
                frac=frac,
            )
    return specs


def load_specs_json(path: str, *, sampler: Optional[Sampler] = None) -> List[VariableSpec]:
    """
    Load a list of variable dictionaries from a JSON file and convert them
    to `VariableSpec`s (optionally registering them on a sampler).

    Parameters
    ----------
    path : str
        Path to the JSON file (must contain a top-level list of dicts).
    sampler : Sampler or None, optional
        If provided, specs are added to this sampler after validation.

    Returns
    -------
    list[VariableSpec]
        Constructed specifications.

    Raises
    ------
    ValueError
        If the JSON root is not a list.

    Notes
    -----
    - The JSON structure mirrors the keys accepted by `specs_from_dicts(...)`.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of variable dicts.")
    return specs_from_dicts(data, sampler=sampler)


def load_specs_csv(path: str, *, sampler: Optional[Sampler] = None) -> List[VariableSpec]:
    """
    Load variable dictionaries from a CSV file into `VariableSpec`s.

    CSV expectations
    ----------------
    - Header row with columns corresponding to the keys described in
      `specs_from_dicts(...)`. Common columns:
        name, central_value, central_unit, upper_error, lower_error,
        error_unit, dist, frac, lower_bound, upper_bound
    - `frac` may be "True"/"False", "1"/"0", "y"/"n" (case-insensitive).

    Parameters
    ----------
    path : str
        Path to the CSV file.
    sampler : Sampler or None, optional
        If provided, specs are added to this sampler after validation.

    Returns
    -------
    list[VariableSpec]
        Constructed specifications.

    Notes
    -----
    - Each CSV row is converted to a dict and passed to `specs_from_dicts(...)`.
    """
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # Convert "True"/"False" to bool for frac
            if "frac" in row and isinstance(row["frac"], str):
                row["frac"] = row["frac"].strip().lower() in ("1", "true", "t", "yes", "y")
            rows.append(row)
    return specs_from_dicts(rows, sampler=sampler)


def save_summary_csv(summary: Dict[str, Dict[str, Quantity]], path: str) -> None:
    """
    Save the summary dictionary to CSV.

    Input schema
    ------------
    summary : dict
        Mapping from output name → dict of fields. Typical fields include
        "median", "q16", "q84", "plus", "minus" (Quantities).

    Output format
    -------------
    - First column: "output" (the output name).
    - Other columns: sorted union of all field keys across outputs.
    - Quantity values are serialized as `"value unit"` strings; non-quantities
      are stringified with `str(...)`.

    Parameters
    ----------
    summary : dict
        The `out["summary"]` mapping from `Sampler.run(...)`.
    path : str
        CSV destination path.

    Notes
    -----
    - Consumers that need numeric types should parse the `"value unit"` strings
      back into Quantities (e.g., split once on whitespace from the right).
    """
    import csv

    # Collect all keys present across outputs to make a single header
    keys = set()
    for _, d in summary.items():
        keys.update(d.keys())
    fieldnames = ["output"] + sorted(keys)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name, d in summary.items():
            row = {"output": name}
            for k, q in d.items():
                if hasattr(q, "unit"):
                    row[k] = f"{q.value} {q.unit}"
                else:
                    row[k] = str(q)
            w.writerow(row)


def save_error_budget_csv(eb: Dict[str, float], path: str) -> None:
    """
    Save an error-budget mapping (variable → Spearman ρ) to CSV.

    Parameters
    ----------
    eb : dict[str, float]
        Mapping from variable name to Spearman rank correlation coefficient.
    path : str
        CSV destination path.

    Output columns
    --------------
    - "variable"
    - "spearman_rho"
    """
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variable", "spearman_rho"])
        for k, v in eb.items():
            w.writerow([k, v])


def save_samples_npz(samples: Dict[str, Quantity], path: str) -> None:
    """
    Save raw per-draw samples to a compressed NumPy `.npz`, preserving units.

    Layout of the NPZ
    -----------------
    - One array per key in `samples` (unitless numeric array).
    - `__keys__`  : object array listing the variable names in order.
    - `__units__` : object array listing string unit representations aligned
                    with `__keys__`.

    Parameters
    ----------
    samples : dict[str, Quantity]
        Mapping from variable name to Quantity array of draws.
    path : str
        Destination `.npz` path.

    Notes
    -----
    - To load and reconstruct Quantities later:
        >>> import numpy as np, astropy.units as u
        >>> data = np.load("file.npz", allow_pickle=True)
        >>> keys  = data["__keys__"]
        >>> units = data["__units__"]
        >>> arrays = {k: data[k] * u.Unit(u_str) if u_str else data[k]
        ...           for k, u_str in zip(keys, units)}
    """
    import numpy as np

    arrays = {}
    meta = {}
    for k, q in samples.items():
        unit = q.unit if hasattr(q, "unit") else None
        arrays[k] = q.to_value(unit) if unit is not None else np.asarray(q)
        meta[k] = str(unit) if unit is not None else ""
    np.savez_compressed(
        path,
        **arrays,
        __units__=np.array(list(meta.values()), dtype=object),
        __keys__=np.array(list(samples.keys()), dtype=object),
    )
