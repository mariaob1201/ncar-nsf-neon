"""Diagnostic plots and metrics for evaluating model misfit against observations."""
from __future__ import annotations
import matplotlib.pyplot as plt

import numpy as np


def residuals_plots(y_obs, y_pred, bins: int = 40, savepath: str | None = None):
    """Compute residual diagnostics for model predictions vs. observations.

    Returns (fig, residuals, metrics, conclusion) where:
      - fig:        matplotlib Figure with obs-vs-pred, residual time series, and residual histogram
      - residuals:  numpy array of (obs - pred) over finite-valued pairs
      - metrics:    dict with bias, mae, rmse, r2, n
      - conclusion: short human-readable summary string

    matplotlib is imported lazily so the module can be imported in headless contexts
    that never call this function.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_obs) & np.isfinite(y_pred)
    obs, pred = y_obs[mask], y_pred[mask]
    residuals = obs - pred

    n = residuals.size
    if n == 0:
        metrics = {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
        return plt.figure(), residuals, metrics, "No finite paired observations to evaluate."

    bias = float(np.mean(residuals))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    metrics = {"n": n, "bias": bias, "mae": mae, "rmse": rmse, "r2": r2}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    lim_lo = float(min(obs.min(), pred.min()))
    lim_hi = float(max(obs.max(), pred.max()))
    axes[0].scatter(pred, obs, s=8, alpha=0.4)
    axes[0].plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Observed")
    axes[0].set_title("Observed vs. Predicted")

    axes[1].plot(residuals, lw=0.7)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Residual (obs - pred)")
    axes[1].set_title("Residual series")

    axes[2].hist(residuals, bins=bins, edgecolor="white")
    axes[2].axvline(0, color="k", lw=0.5)
    axes[2].set_xlabel("Residual")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Residual distribution")

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    quality = "good" if abs(bias) < 0.1 * (rmse if rmse > 0 else 1.0) and (np.isnan(r2) or r2 > 0.7) else "needs improvement"
    conclusion = (
        f"n={n}, bias={bias:.3g}, mae={mae:.3g}, rmse={rmse:.3g}, r2={r2:.3g} -> fit quality: {quality}."
    )

    return fig, residuals, metrics, conclusion
