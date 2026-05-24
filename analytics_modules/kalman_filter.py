"""
Kalman filter calibration for CTSM simulation outputs.
"""
from __future__ import annotations

import numpy as np

from .model_misfit import residuals_plots


def kalman_filter(df, var):
    """Simple scalar Kalman filter merging model predictions with observations."""

    df_clean = df.dropna(subset=[var, 'sim_' + var])

    sim = df_clean[var]
    obs = df_clean['sim_' + var]
    x_est = sim.iloc[0]
    P = 1.0
    Q = 1e-3
    R = 0.1 * np.var(obs - sim)

    kalman_estimates = []

    for i in range(len(obs)):
        z = obs.iloc[i]
        x_pred = sim.iloc[i]

        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x_est = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred

        kalman_estimates.append(x_est)

    kalman_estimates = np.array(kalman_estimates)
    bias = np.mean(kalman_estimates - obs)
    kalman_corrected = kalman_estimates - bias

    df_clean["kalman_" + var] = kalman_estimates
    df_clean["kalman_" + var + "_bias_corrected"] = kalman_corrected

    return df_clean


def kalman_gain_bias(y_obs, y_sim, hours=None, Q_diag=(1e-4, 1e-4, 1e-6, 1e-6), R0_scale=0.1, smooth=True):
    """
    Linear state-space with predictor vector h_t = [1, sim_t, sin(wt), cos(wt)] (last two optional).
        state theta_t = [bias_t, gain_t, s_t, c_t]';  theta_t = theta_{t-1} + w_t,  w~N(0,Q)
        obs   y_t = h_t . theta_t + v_t,              v~N(0,R_t)
    If hours is None -> model uses [1, sim_t] only.
    """
    y_obs, y_sim = np.asarray(y_obs, float), np.asarray(y_sim, float)
    m = np.isfinite(y_obs) & np.isfinite(y_sim)
    y, s = y_obs[m], y_sim[m]
    n = len(y)
    use_harm = hours is not None
    if use_harm:
        h = (np.asarray(hours, int) % 24)[m]
        w = 2 * np.pi / 24.0
        H = np.column_stack([np.ones(n), s, np.sin(w * h), np.cos(w * h)])
        Q = np.diag([Q_diag[0], Q_diag[1], Q_diag[2], Q_diag[3]])
    else:
        H = np.column_stack([np.ones(n), s])
        Q = np.diag([Q_diag[0], Q_diag[1]])
    dim = H.shape[1]

    theta = np.zeros(dim)
    P = np.eye(dim)
    R = R0_scale * np.var(y - s) if np.isfinite(np.var(y - s)) else 1.0
    R = max(R, 1e-8)

    theta_f = np.zeros((n, dim))
    P_f = np.zeros((n, dim))
    K_hist = np.zeros(n)
    innov = np.zeros(n)
    S_hist = np.zeros(n)

    for t in range(n):
        theta_pred = theta
        P_pred = P + Q
        ht = H[t]
        v = y[t] - ht @ theta_pred
        S = float(ht @ P_pred @ ht + R)
        K = (P_pred @ ht) / S
        theta = theta_pred + K * v
        P = (np.eye(dim) - np.outer(K, ht)) @ P_pred

        theta_f[t] = theta
        P_f[t] = np.diag(P)
        K_hist[t] = K[1] if dim > 1 else K[0]
        innov[t] = v
        S_hist[t] = S

        R_est = max(v * v - float(ht @ P_pred @ ht), 1e-10)
        R = 0.95 * R + 0.05 * R_est

    y_cal = np.sum(H * theta_f, axis=1)
    lo = y_cal - 1.96 * np.sqrt(np.maximum(np.sum((H ** 2) * P_f, axis=1), 1e-12))
    hi = y_cal + 1.96 * np.sqrt(np.maximum(np.sum((H ** 2) * P_f, axis=1), 1e-12))

    if smooth:
        theta_s = theta_f.copy()
        Pd = np.diag(Q)
        for t in range(n - 2, -1, -1):
            P_pred_next = np.diag(P_f[t] + Pd)
            J = np.diag(P_f[t]) @ np.linalg.pinv(P_pred_next)
            theta_s[t] = theta_f[t] + (J @ (theta_s[t + 1] - theta_f[t + 1]))
        y_smooth = np.sum(H * theta_s, axis=1)
    else:
        y_smooth = None

    return y_cal, (lo, hi), y_smooth, {"theta_seq": theta_f, "innov": innov, "S": S_hist}


def calibrate_and_evaluate(df, col, method="auto", hour_col=None):
    """Run Kalman calibration and produce before/after misfit diagnostics."""
    d = df.dropna(subset=[col, "sim_" + col]).copy()
    y_obs = d[col].to_numpy(float)
    y_sim = d["sim_" + col].to_numpy(float)
    hours = d[hour_col].to_numpy(int) if (hour_col and hour_col in d) else None

    print("--------------------- Observations vs simulations")
    fig, residuals, metrics_pre, conclusion_pre = residuals_plots(
        y_obs, y_sim, bins=40, savepath=None,
    )
    print(conclusion_pre)
    print(metrics_pre)

    print("--------------------- Observations vs KF Assimilation")
    y_cal, (lo, hi), y_smooth, info = kalman_gain_bias(y_obs, y_sim, hours=hours)
    d["cal_lo"], d["cal_hi"] = lo, hi
    if y_smooth is not None:
        d["cal_smooth"] = y_smooth
    pars = {"kf": "bias+gain" + ("+harmonics" if hours is not None else "")}

    d["cali_sim_" + col] = y_cal
    fig, residuals, metrics_post, conclusion_post = residuals_plots(
        y_obs, d["cali_sim_" + col].to_numpy(float), bins=40, savepath=None,
    )
    print(conclusion_post)
    print(metrics_post)
    return d, {"pre_metrics": metrics_pre, "post_metrics": metrics_post, "method": method, "params": pars}
