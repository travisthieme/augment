"""
augment.core
=====================

Purpose
-------
Core sampling utilities for propagating asymmetric uncertainties using
Monte Carlo draws. This module defines the main `Sampler`
class and the common logic for:
- registering named input variables (with asymmetric errors)
- drawing unit-aware samples
- running a user-supplied model on those samples
- collecting posterior samples and summary statistics
- computing rank-correlation error budgets

Key features
------------
- Unit-aware: internally uses `astropy.units.Quantity`
- Asymmetric errors: (central, +upper_error, -lower_error)
- Positive-only support via lower bounds
- Optional error budget against a single output key
- Returns both raw samples and a structured `summary` dict

Inputs / outputs
----------------
- Inputs: dict[str, Quantity] describing central + errors
- Model: callable that returns a Quantity or dict[str, Quantity]
- Outputs: dict with keys:
    - "samples": dict[str, Quantity]
    - "summary": dict[str, dict[str, Quantity]]
    - "error_budget": dict[str, float] (Spearman ρ vs chosen output)

Dependencies
------------
- numpy
- scipy
- astropy

Notes
-----
- All random draws should be reproducible with `seed=...`.
- Summary uses linear-space quantiles by default; log-space printing is
  implemented in `astro_asymmetric.format`.

Author: Travis Thieme
Created: 2025-11-06
Last updated: 2025-11-06
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Callable
import numpy as np
from scipy.stats import spearmanr as _scipy_spearmanr
from astropy import units as u
from astropy.units import Quantity

__all__ = ["VariableSpec", "Sampler", "spearman_r"]

@dataclass
class VariableSpec:
    """
    Specification for a random variable with asymmetric uncertainties.

    Parameters
    ----------
    name : str
        Variable name (keyword when calling the model function).
    central_value : Quantity
        Central value with units (or dimensionless Quantity). Must be finite.
    upper_error : Quantity
        Upper 1-sigma uncertainty. If `frac=True`, interpret as *fractional* (dimensionless).
    lower_error : Quantity
        Lower 1-sigma uncertainty. If `frac=True`, interpret as *fractional* (dimensionless).
    dist : {"split_normal","normal","lognormal"}
        - "split_normal": two-piece Gaussian using lower/upper sigmas in linear space (default).
        - "normal": symmetric Gaussian in linear space with sigma = 0.5*(upper+lower).
        - "lognormal": symmetric Gaussian in log-space (uses fractional sigma; asymmetry ignored).
    lower : Optional[Quantity]
        Hard lower bound. If provided, samples are clipped at this bound.
    upper : Optional[Quantity]
        Hard upper bound. If provided, samples are clipped at this bound.
    frac : bool
        If True, `upper_error` and `lower_error` are fractional (dimensionless) uncertainties
        relative to `central_value`. E.g., upper_error=0.2 means +20%.

    Notes
    -----
    - `validate()` enforces dimensional compatibility and sanity checks (finite central value).
    - Bounds (`lower`, `upper`) are applied *after* drawing from the core distribution.
    - For `lognormal`, the effective fractional sigma is derived from the provided errors.
    """

    name: str
    central_value: Quantity
    upper_error: Quantity
    lower_error: Quantity
    dist: str = "split_normal"
    lower: Optional[Quantity] = None
    upper: Optional[Quantity] = None
    frac: bool = False

    def validate(self) -> None:
        """
        Sanity-check this specification:
        - Ensure central_value and errors are Quantities.
        - Ensure finite central_value.
        - Ensure units are compatible (errors with central_value unless `frac=True`).
        - Ensure bounds (if any) share compatible units.
        """
        if not isinstance(self.central_value, Quantity):
            raise TypeError(f"{self.name}: central_value must be an astropy Quantity.")
        if not np.isfinite(self.central_value.to_value(self.central_value.unit)):
            raise ValueError(f"{self.name}: central_value must be finite.")
        if not isinstance(self.upper_error, Quantity):
            raise TypeError(f"{self.name}: upper_error must be an astropy Quantity.")
        if not isinstance(self.lower_error, Quantity):
            raise TypeError(f"{self.name}: lower_error must be an astropy Quantity.")
        if self.frac:
            # When fractional, errors must be dimensionless.
            if not self.upper_error.unit.is_equivalent(u.dimensionless_unscaled):
                raise TypeError(f"{self.name}: upper_error must be dimensionless when frac=True.")
            if not self.lower_error.unit.is_equivalent(u.dimensionless_unscaled):
                raise TypeError(f"{self.name}: lower_error must be dimensionless when frac=True.")
        else:
            # Otherwise, errors must convert cleanly to central_value's unit.
            _ = self.upper_error.to(self.central_value.unit)
            _ = self.lower_error.to(self.central_value.unit)
        if self.lower is not None:
            _ = self.central_value.to(self.lower.unit)
        if self.upper is not None:
            _ = self.central_value.to(self.upper.unit)


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Spearman rank correlation coefficient (ρ) with SciPy.

    Parameters
    ----------
    x, y : np.ndarray (1D)
        Arrays of equal length. Non-finite pairs are dropped.

    Returns
    -------
    float
        Spearman ρ in [-1, 1], or `nan` if insufficient variation/length.

    Notes
    -----
    - Uses `nan_policy='omit'` to ignore non-finite pairs.
    - Returns `nan` if arrays are too short or constant.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size or x.size < 3:
        return float("nan")
    r, _ = _scipy_spearmanr(x, y, nan_policy="omit")
    return float(r) if np.isfinite(r) else float("nan")


class Sampler:
    """
    Astropy-units-aware Monte Carlo sampler for models with asymmetric measurement uncertainties.

    Parameters
    ----------
    n_samples : int, optional
        Number of Monte Carlo draws to generate for each registered variable.
    seed : int or None, optional
        Seed for the internal NumPy Generator (reproducibility).
    vectorized : bool, optional
        Reserved for future use; kept for API symmetry. (Currently not altering behavior.)

    Usage
    -----
    >>> sampler = Sampler(n_samples=200000, seed=42)
    >>> sampler.add("N_H2", 1e22 * u.cm**-2, upper_error=2e21*u.cm**-2, lower_error=1e21*u.cm**-2)
    >>> def model(N_H2): return 0.5 * N_H2
    >>> out = sampler.run(model, return_samples=True)
    >>> out["summary"]["model"]["median"]
    <Quantity ...>

    Notes
    -----
    - Inputs are registered via `.add(...)` and validated with `VariableSpec.validate()`.
    - `.run(model, ...)` draws samples, invokes your `model`, collects outputs,
      summarizes them (median / q16 / q84), and optionally computes an error budget.
    """

    def __init__(self, n_samples: int = 200000, seed: Optional[int] = 42, vectorized: bool = True):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        self.n_samples = int(n_samples)
        self.vectorized = bool(vectorized)
        self._vars: Dict[str, VariableSpec] = {}
        self._rng = np.random.default_rng(seed)

    def add(self,
            name: str,
            central_value: Quantity,
            *,
            upper_error: Quantity,
            lower_error: Quantity,
            dist: str = "split_normal",
            lower: Optional[Quantity] = None,
            upper: Optional[Quantity] = None,
            frac: bool = False) -> None:
        """
        Register a variable to be sampled. All values must be Quantities.

        Parameters
        ----------
        name : str
            Variable name (used as keyword when calling the model).
        central_value, upper_error, lower_error : Quantity
            Central value and asymmetric uncertainties. If `frac=True`, the
            errors must be dimensionless fractional uncertainties.
        dist : {"split_normal","normal","lognormal"}
            Choice of sampling distribution in *linear* or *log* space.
        lower, upper : Quantity or None
            Optional hard bounds; samples are clipped to [lower, upper].
        frac : bool
            Interpret errors as fractional when True.

        Notes
        -----
        - This method constructs a `VariableSpec`, runs `.validate()`, and stores
          it internally under `name`.
        """
        spec = VariableSpec(
            name=name,
            central_value=central_value,
            upper_error=upper_error,
            lower_error=lower_error,
            dist=dist,
            lower=lower,
            upper=upper,
            frac=frac,
        )
        spec.validate()
        self._vars[name] = spec

    def variables(self) -> Dict[str, VariableSpec]:
        """
        Return a shallow copy of the registered variable specifications.

        Returns
        -------
        dict
            Mapping from variable name to `VariableSpec`.
        """
        return dict(self._vars)

    # ---- sampling kernels ----
    def _sample_split_normal(self, spec: VariableSpec) -> Quantity:
        """
        Draw samples from a two-piece (split) normal distribution in linear space.

        Notes
        -----
        - Positive z → uses upper sigma; negative z → uses lower sigma.
        - If `frac=True`, errors are interpreted as fractional and scaled by `central_value`.
        - Bounds (if provided) are enforced after drawing.
        """
        c = spec.central_value
        unit = c.unit
        if spec.frac:
            sp = spec.upper_error.to_value(u.dimensionless_unscaled)
            sm = spec.lower_error.to_value(u.dimensionless_unscaled)
            sigma_plus  = sp * c.to_value(unit)
            sigma_minus = sm * c.to_value(unit)
        else:
            sigma_plus  = spec.upper_error.to_value(unit)
            sigma_minus = spec.lower_error.to_value(unit)
        z = self._rng.standard_normal(self.n_samples)
        sig = np.where(z >= 0.0, sigma_plus, sigma_minus)
        samples = c.to_value(unit) + z * sig
        if spec.lower is not None:
            low = spec.lower.to_value(unit)
            samples = np.maximum(samples, low)
        if spec.upper is not None:
            up = spec.upper.to_value(unit)
            samples = np.minimum(samples, up)
        return samples * unit

    def _sample_normal(self, spec: VariableSpec) -> Quantity:
        """
        Draw samples from a symmetric normal distribution in linear space.

        Notes
        -----
        - If `frac=True`, sigma = frac_sigma * central_value.
        - Else, sigma = 0.5 * (upper_error + lower_error) in the central_value's unit.
        - Bounds (if provided) are enforced after drawing.
        """
        c = spec.central_value
        unit = c.unit
        if spec.frac:
            frac_sigma = 0.5 * (spec.upper_error.to_value(u.dimensionless_unscaled) +
                                spec.lower_error.to_value(u.dimensionless_unscaled))
            sigma = frac_sigma * c.to_value(unit)
        else:
            sigma = 0.5 * (spec.upper_error.to_value(unit) + spec.lower_error.to_value(unit))
        samples = self._rng.normal(loc=c.to_value(unit), scale=sigma, size=self.n_samples)
        if spec.lower is not None:
            samples = np.maximum(samples, spec.lower.to_value(unit))
        if spec.upper is not None:
            samples = np.minimum(samples, spec.upper.to_value(unit))
        return samples * unit

    def _sample_lognormal(self, spec: VariableSpec) -> Quantity:
        """
        Draw samples assuming a log-normal distribution.

        Notes
        -----
        - Interprets the provided (asymmetric) errors as an effective **fractional sigma**.
          When `frac=False`, this is estimated from linear errors relative to central value.
        - Sampling is performed in log-space with `mu_ln = log(central)` and
          `sigma_ln = log(1 + frac_sigma)` for numerical stability.
        - Bounds (if provided) are enforced after exponentiation.
        """
        c = spec.central_value
        unit = c.unit
        if spec.frac:
            frac_sigma = 0.5 * (spec.upper_error.to_value(u.dimensionless_unscaled) +
                                spec.lower_error.to_value(u.dimensionless_unscaled))
        else:
            # Guard against division by zero with a tiny floor.
            denom = max(c.to_value(unit), 1e-300)
            frac_sigma = 0.5 * (spec.upper_error.to_value(unit) + spec.lower_error.to_value(unit)) / denom
        frac_sigma = max(frac_sigma, 1e-12)
        sigma_ln = np.log1p(frac_sigma)
        mu_ln = np.log(max(c.to_value(unit), 1e-300))
        samples = self._rng.normal(loc=mu_ln, scale=sigma_ln, size=self.n_samples)
        samples = np.exp(samples)
        if spec.lower is not None:
            samples = np.maximum(samples, spec.lower.to_value(unit))
        if spec.upper is not None:
            samples = np.minimum(samples, spec.upper.to_value(unit))
        return samples * unit

    def sample(self) -> Dict[str, Quantity]:
        """
        Draw samples for all registered variables using their specified distributions.

        Returns
        -------
        dict[str, Quantity]
            Mapping from variable name to sampled `Quantity` array (length = `n_samples`).
        """
        out: Dict[str, Quantity] = {}
        for name, spec in self._vars.items():
            if spec.dist == "split_normal":
                arr = self._sample_split_normal(spec)
            elif spec.dist == "normal":
                arr = self._sample_normal(spec)
            elif spec.dist == "lognormal":
                arr = self._sample_lognormal(spec)
            else:
                raise ValueError(f"{name}: unknown dist='{spec.dist}'.")
            out[name] = arr
        return out

    @staticmethod
    def _as_quantity_array(x: Any) -> Quantity:
        """
        Ensure the model output is an Astropy Quantity.

        Raises
        ------
        TypeError
            If the provided object is not a `Quantity`.
        """
        if isinstance(x, Quantity):
            return x
        raise TypeError("Model must return astropy Quantities (or collections thereof).")

    @staticmethod
    def _percentiles(q: Tuple[float, float]) -> Tuple[float, float]:
        """
        Validate and normalize a (q1, q2) percentile pair in the inclusive [0,100] range.
        """
        q1, q2 = float(q[0]), float(q[1])
        if not (0.0 <= q1 < q2 <= 100.0):
            raise ValueError("q must be a pair of percent values in ascending order within [0,100].")
        return q1, q2

    def _summarize_array(self, arr: Quantity, q: Tuple[float, float]) -> Dict[str, Quantity]:
        """
        Summarize a Quantity array with median, q1, q2, and asymmetric errors in **linear space**.

        Parameters
        ----------
        arr : Quantity
            One output variable's samples.
        q : (float, float)
            Percentile bounds (e.g., (16.0, 84.0)).

        Returns
        -------
        dict
            {
              "median": <Quantity>,
              "q{q1}": <Quantity>,
              "q{q2}": <Quantity>,
              "minus": <Quantity>,   # median - q1
              "plus":  <Quantity>,   # q2 - median
            }

        Notes
        -----
        - This is a raw linear-space summary. For log10 printing, see the format layer.
        - `np.nanpercentile`/`np.nanmedian` are used to ignore NaNs if present.
        """
        q1, q2 = self._percentiles(q)
        vals = arr.to_value(arr.unit)
        med = np.nanmedian(vals)
        lo, hi = np.nanpercentile(vals, [q1, q2])
        return {
            "median": med * arr.unit,
            f"q{int(q1)}": lo * arr.unit,
            f"q{int(q2)}": hi * arr.unit,
            "minus": (med - lo) * arr.unit,
            "plus": (hi - med) * arr.unit,
        }

    def run(self,
            model: Callable[..., Any],
            *,
            q: Tuple[float, float] = (16.0, 84.0),
            return_samples: bool = False,
            error_budget_against: Optional[str] = None
            ) -> Dict[str, Any]:
        """
        Sample variables and evaluate the model.

        Workflow
        --------
        1) Draw samples for all registered variables.
        2) Call the user-supplied `model`:
           - Prefer keyword call `model(**samples)`; if that fails due to signature,
             fall back to positional call in registration order.
        3) Normalize model outputs into a dict of `Quantity` arrays (`outputs`).
        4) Build a linear-space percentile summary for each output via `_summarize_array`.
        5) Optionally attach raw `outputs` under `"samples"` if `return_samples=True`.
        6) Optionally compute a **rank-correlation error budget**:
           Spearman ρ between each input variable's draws and the chosen output key.

        Parameters
        ----------
        model : callable
            A function that accepts the sampled variables and returns either:
              - a single `Quantity`,
              - a sequence of `Quantity` (tuple/list),
              - a dict[str, `Quantity`].
        q : (float, float), optional
            Percentiles for asymmetric errors (default: (16, 84)).
        return_samples : bool, optional
            If True, include raw model outputs under `out["samples"]`.
        error_budget_against : str or None, optional
            If provided, compute the Spearman rank correlation coefficient between
            each *input* variable and the specified *output* series. Store results
            under `out["error_budget"]` mapping var_name → ρ.

        Returns
        -------
        dict
            {
              "summary": { output_key: { "median", "q16", "q84", "minus", "plus" } },
              ["samples": { output_key: Quantity[...] }],              # if requested
              ["error_budget": { var_name: float (Spearman ρ) }],      # if requested
            }

        Raises
        ------
        TypeError
            If the model returns unsupported types or non-Quantity outputs.
        KeyError
            If `error_budget_against` is set but not present in the outputs.

        Notes
        -----
        - The error budget is unit-agnostic (rank correlation on numeric arrays).
        - `return_samples=True` lets downstream plotting/formatting use masked,
          log-transformed, or unit-converted views of the same draws for perfect consistency.
        """
        samples = self.sample()
        model_name = getattr(model, "__name__", "model")
        try:
            # Preferred path: keyword arguments using registered variable names.
            result = model(**samples)
        except TypeError:
            # Fallback: positional call in the order variables were registered.
            ordered = [samples[k] for k in self._vars.keys()]
            result = model(*ordered)

        # Normalize outputs into a dict[str, Quantity]
        outputs: Dict[str, Quantity] = {}
        if isinstance(result, Quantity):
            outputs[model_name] = result
        elif isinstance(result, (tuple, list)):
            for i, r in enumerate(result):
                outputs[f"{model_name}[{i}]"] = self._as_quantity_array(r)
        elif isinstance(result, dict):
            for k, r in result.items():
                outputs[str(k)] = self._as_quantity_array(r)
        else:
            raise TypeError("Model returned an unsupported type.")

        # Linear-space percentile summaries for each output key
        summary: Dict[str, Dict[str, Quantity]] = {}
        for k, arr in outputs.items():
            summary[k] = self._summarize_array(arr, q=q)

        out: Dict[str, Any] = {"summary": summary}
        if return_samples:
            out["samples"] = outputs

        # Optional rank-correlation error budget vs a specific output
        if error_budget_against is not None:
            if error_budget_against not in outputs:
                raise KeyError(f"error_budget_against='{error_budget_against}' not in outputs: {list(outputs.keys())}")
            target = outputs[error_budget_against]
            target_vals = target.to_value(target.unit)
            eb: Dict[str, float] = {}
            for vname, spec in self._vars.items():
                arr = samples[vname].to_value(spec.central_value.unit)
                try:
                    rho = spearman_r(arr, target_vals)
                except Exception:
                    rho = np.nan
                eb[vname] = float(rho)
            out["error_budget"] = eb

        return out
