# These functions should belong to from utilities import evaluate_misfit
#These functions should belong to from utilities import evaluate_misfit

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xarray as xr

#

"""
This code is to evaluate misfit of the CTSM model simulations against observations of data
"""

def residuals_plots(
    y_obs,
    y_simulations,
    bins: int = 30,
    figsize: tuple = (12, 4),
    alpha: float = 0.6,
    savepath: Optional[str] = None,
    show: bool = True,
    thresholds: Optional[dict] = None,
):
    """
    Plot residual diagnostics (histogram, Q–Q plot, residuals vs fitted) and
    produce a short text conclusion about misfit.

    Returns
    -------
    fig : matplotlib.figure.Figure
    residuals : np.ndarray
    metrics : dict
    conclusion : str
    """
    # ---- Config ----
    thr = {
        "alpha_norm": 0.05,   # normality test significance
        "bias_ratio": 0.10,   # |bias| > 10% of std(obs) => flag
        "rho_abs": 0.20,      # |Spearman rho| >= 0.2 => heteroscedasticity flag
        "rmse_ratio": 0.50,   # RMSE > 50% of std(obs) => large error
    }
    if thresholds:
        thr.update(thresholds)

    # ---- Data cleaning ----
    y_obs = np.asarray(y_obs, dtype=float)
    y_sim = np.asarray(y_simulations, dtype=float)
    mask = np.isfinite(y_obs) & np.isfinite(y_sim)
    y_obs, y_sim = y_obs[mask], y_sim[mask]
    residuals = y_obs - y_sim

    # ---- Metrics ----
    n = residuals.size
    bias = float(np.mean(residuals)) if n else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2))) if n else np.nan
    mae  = float(np.mean(np.abs(residuals))) if n else np.nan
    obs_std = float(np.std(y_obs)) if n else np.nan
    bias_ratio = (abs(bias) / obs_std) if (np.isfinite(obs_std) and obs_std > 0) else np.nan

    # R and R^2 (guard tiny n)
    r = np.nan
    r2 = np.nan
    if n > 1 and np.std(y_obs) > 0 and np.std(y_sim) > 0:
        r = float(np.corrcoef(y_obs, y_sim)[0, 1])
        r2 = r**2

    # Normality of residuals (D’Agostino–Pearson)
    if n >= 8:
        _, p_norm = stats.normaltest(residuals)
    else:
        p_norm = np.nan

    # Heteroscedasticity proxy: |residuals| vs fitted (Spearman)
    if n > 2:
        rho, p_rho = stats.spearmanr(np.abs(residuals), y_sim)
    else:
        rho, p_rho = np.nan, np.nan

    metrics = {
        "n": n,
        "bias": bias,
        "bias_vs_std_obs": bias_ratio,   # unitless
        "rmse": rmse,
        "mae": mae,
        "r": r,
        "r2": r2,
        "p_norm": p_norm,                # residual normality p-value
        "rho_absres_vs_fitted": rho,     # heteroscedasticity proxy
        "p_rho": p_rho,
        "std_obs": obs_std,
    }

    # ---- Simple rule-based conclusion ----
    issues = []
    if np.isfinite(bias_ratio) and bias_ratio > thr["bias_ratio"]:
        issues.append(f"noticeable bias ({bias:.3g}, {bias_ratio:.0%} of obs std)")
    if np.isfinite(rmse) and np.isfinite(obs_std) and obs_std > 0 and (rmse > thr["rmse_ratio"] * obs_std):
        issues.append(f"RMSE large vs variability (RMSE={rmse:.3g}, std(obs)={obs_std:.3g})")
    if np.isfinite(p_norm) and p_norm < thr["alpha_norm"]:
        issues.append("residuals deviate from normality (p<0.05)")
    if np.isfinite(rho) and abs(rho) >= thr["rho_abs"]:
        issues.append(f"heteroscedasticity pattern (|ρ|={abs(rho):.2f})")

    if not issues:
        conclusion = (
            f"Good fit: residuals centered near 0 (bias={bias:.3g}), "
            f"no strong pattern vs fitted (ρ={rho:.2f}), approx. normal "
            f"(p={p_norm:.3g}); R²≈{r2:.2f}."
        )
    else:
        conclusion = (
            "Misfit notes: " + "; ".join(issues) +
            (f" R²≈{r2:.2f}, MAE={mae:.3g}" if np.isfinite(r2) else " inf")
        )

    # ---- Plots ----
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # 1) Histogram
    axes[0].hist(residuals, bins=bins, edgecolor="black")
    axes[0].set_title("Residuals Histogram")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Count")

    # 2) Q–Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q–Q Plot (Residuals)")

    # 3) Residuals vs Fitted
    axes[2].scatter(y_sim, residuals, alpha=alpha, s=18)
    axes[2].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[2].set_title("Residuals vs Fitted")
    axes[2].set_xlabel("Fitted (Simulated)")
    axes[2].set_ylabel("Residual (Obs − Sim)")
    
    # 4) Obs vs Fitted
    axes[3].scatter(y_obs, y_sim, alpha=alpha, s=18)
    axes[3].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[3].set_title("Observed vs Fitted")
    axes[3].set_ylabel("Fitted (Simulated)")
    axes[3].set_xlabel("Observed")

    # Summary line on top
    fig.suptitle(conclusion, fontsize=10, y=1.03)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, residuals, metrics, conclusion



def kalman_filter(df, var):
    '''
    '''
    
    df_clean = df.dropna(subset=[var, 'sim_'+var])
    
    sim = df_clean[var] 
    obs = df_clean['sim_'+var]
    x_est = sim[0]                # Start from the model
    P = 1.0                       # Initial uncertainty, also empirical
    Q = 1e-3                      # Process variance small, empirically defined
    R = 0.1 * np.var(obs - sim)  # Measurement noise: trust model more than obs

    kalman_estimates = []

    for i in range(len(obs)):
        z = obs[i]       # Observation
        x_pred = sim[i]  # Model prediction

        # Prediction uncertainty update
        P_pred = P + Q

        # Kalman gain
        K = P_pred / (P_pred + R)

        # State update
        x_est = x_pred + K * (z - x_pred)

        # Uncertainty update
        P = (1 - K) * P_pred

        kalman_estimates.append(x_est)

    # Bias correction (optional)
    kalman_estimates = np.array(kalman_estimates)
    bias = np.mean(kalman_estimates - obs) #this correction shifts it to have zero mean bias.
    kalman_corrected = kalman_estimates - bias

    # Save to dataframe
    df_clean["kalman_"+var] = kalman_estimates
    df_clean["kalman_"+var+"_bias_corrected"] = kalman_corrected
    
    return df_clean


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 4) Kalman with time-varying bias & gain (+ optional diurnal harmonics) ----------
def kalman_gain_bias(y_obs, y_sim, hours=None, Q_diag=(1e-4, 1e-4, 1e-6, 1e-6), R0_scale=0.1, smooth=True):
    """
    Linear state-space with predictor vector h_t = [1, sim_t, sin(wt), cos(wt)] (last two optional).
        state θ_t = [bias_t, gain_t, s_t, c_t]';  θ_t = θ_{t-1} + w_t,  w~N(0,Q)
        obs   y_t = h_t · θ_t + v_t,            v~N(0,R_t)
    If hours is None -> model uses [1, sim_t] only.
    """
    y_obs, y_sim = np.asarray(y_obs, float), np.asarray(y_sim, float)
    m = np.isfinite(y_obs) & np.isfinite(y_sim)
    y, s = y_obs[m], y_sim[m]
    n = len(y)
    use_harm = hours is not None
    if use_harm:
        h = (np.asarray(hours, int)%24)[m]
        w = 2*np.pi/24.0
        H = np.column_stack([np.ones(n), s, np.sin(w*h), np.cos(w*h)])
        Q = np.diag([Q_diag[0], Q_diag[1], Q_diag[2], Q_diag[3]])
    else:
        H = np.column_stack([np.ones(n), s])
        Q = np.diag([Q_diag[0], Q_diag[1]])
    dim = H.shape[1]

    # init
    theta = np.zeros(dim)
    P = np.eye(dim)
    R = R0_scale * np.var(y - s) if np.isfinite(np.var(y - s)) else 1.0
    R = max(R, 1e-8)

    theta_f = np.zeros((n, dim))
    P_f = np.zeros((n, dim))
    K_hist = np.zeros(n)
    innov = np.zeros(n)
    S_hist = np.zeros(n)

    # filter
    for t in range(n):
        # predict
        theta_pred = theta
        P_pred = P + Q
        ht = H[t]
        # innovation
        v = y[t] - ht @ theta_pred
        S = float(ht @ P_pred @ ht + R)
        K = (P_pred @ ht) / S
        # update
        theta = theta_pred + K * v
        P = (np.eye(dim) - np.outer(K, ht)) @ P_pred

        theta_f[t] = theta
        P_f[t] = np.diag(P)
        K_hist[t] = K[1] if dim>1 else K[0]
        innov[t] = v
        S_hist[t] = S

        # simple adaptive R (EMA)
        R_est = max(v*v - float(ht @ P_pred @ ht), 1e-10)
        R = 0.95*R + 0.05*R_est

    # calibrated
    y_cal = np.sum(H * theta_f, axis=1)
    lo = y_cal - 1.96 * np.sqrt(np.maximum(np.sum((H**2)*P_f, axis=1), 1e-12))
    hi = y_cal + 1.96 * np.sqrt(np.maximum(np.sum((H**2)*P_f, axis=1), 1e-12))

    # RTS smoother (optional)
    if smooth:
        theta_s = theta_f.copy()
        Pd = np.diag(Q)  # for random walk
        for t in range(n-2, -1, -1):
            P_pred_next = np.diag(P_f[t] + Pd)
            J = np.diag(P_f[t]) @ np.linalg.pinv(P_pred_next)
            theta_s[t] = theta_f[t] + (J @ (theta_s[t+1] - theta_f[t+1]))
        y_smooth = np.sum(H * theta_s, axis=1)
    else:
        y_smooth = None

    return y_cal, (lo, hi), y_smooth, {"theta_seq": theta_f, "innov": innov, "S": S_hist}

# ---------- Orchestrator ----------
def calibrate_and_evaluate(df, col, method="auto", hour_col=None):
    '''
    '''
    d = df.dropna(subset=[col, "sim_"+col]).copy()
    y_obs = d[col].to_numpy(float)
    y_sim = d["sim_"+col].to_numpy(float)
    hours = d[hour_col].to_numpy(int) if (hour_col and hour_col in d) else None
    
    print("--------------------- Observations vs simulations")
    fig, residuals, metrics_pre, conclusion_pre = residuals_plots(
        y_obs,
        y_sim,
        bins=40,
        savepath=None,
    )
    print(conclusion_pre)
    print(metrics_pre)

    print("--------------------- Observations vs KF Assimilation")
    y_cal, (lo, hi), y_smooth, info = kalman_gain_bias(y_obs, y_sim, hours=hours)
    d["cal_lo"], d["cal_hi"] = lo, hi
    if y_smooth is not None:
        d["cal_smooth"] = y_smooth
    pars = {"kf": "bias+gain" + ("+harmonics" if hours is not None else "")}

    d["cali_sim_"+col] = y_cal
    fig, residuals, metrics_post, conclusion_post = residuals_plots(
        y_obs,
        d["cali_sim_"+col].to_numpy(float),
        bins=40,
        savepath=None,
    )
    print(conclusion_post)
    print(metrics_post)
    return d, {"pre_metrics": metrics_pre, "post_metrics": metrics_post, "method": method, "params": pars}



######################## time series comparison
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

### something weird here
def compute_fit(df, obs, sim):
    mask = df[obs].notna() & df[sim].notna()
    y_true = df.loc[mask, obs]
    y_pred = df.loc[mask, sim]

    #r2  = r2_score(y_true, y_pred)
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    r2 = r**2
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias}


def comparison(df, label, var, variables_units):
    #Comparison at the time series level of aggregation
    
    plot_var = var
    sim_var  = f"sim_{plot_var}"
    calib_sim_var  = f"cali_sim_{plot_var}"

    plot_var_desc = variables_units[plot_var]['var_name']
    plot_var_unit = variables_units[plot_var]['units']

    # ensure time is datetime & sorted (optional but helpful)
    df_daily = df.copy()
    df_daily = df_daily.sort_values("time")

    fig, ax = plt.subplots(figsize=(13, 5))

    # Plot with pandas but turn OFF legend to avoid the bug
    df_daily.plot(x="time", y=plot_var,   marker="o", ax=ax, color="b", legend=False)
    df_daily.plot(x="time", y=sim_var,    marker="o", ax=ax, color="r", legend=False)
    df_daily.plot(x="time", y=calib_sim_var,    marker="o", ax=ax, color="g", legend=False)
    
    fit_sim  = compute_fit(df_daily, var, "sim_"+var)
    fit_cali = compute_fit(df_daily, var, "cali_sim_"+var)

    print("CLM fit:", fit_sim)
    print("KF_CLM fit:", fit_cali)

    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel(f"{plot_var_desc} [{plot_var_unit}]", fontsize=14)
    ax.legend(["NEON", "CLM", "KF_CLM"], fontsize=12)   # normal Matplotlib legend
    ax.text(0.01, 0.95,
        f"CLM R²={fit_sim['R2']:.2f}, RMSE={fit_sim['RMSE']:.2f}\n"
        f"KF_CLM R²={fit_cali['R2']:.2f}, RMSE={fit_cali['RMSE']:.2f}",
        transform=ax.transAxes, fontsize=12,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    ax.set_title(f"{label}", fontweight="bold", fontsize=16)

    plt.tight_layout()
    plt.show()
    
#-- extract year, month, day, hour information from time
def time_series_comparison(df, label, var):
    '''
        Simply giving formatting to the data and call comparisons function
    '''
    
    variables_units_dict = {
        'EFLX_LH_TOT': {
            'units':"W m$^{-2}$",
            'var_name': "Latent Heat Flux"
            },
        'GPP': {
            'units':"",
            'var_name': "GPP"
            },
        'H2OSOI': {
            'units':"",
            'var_name': "H2OSOI"
            }
        }
    
    df_cal = df.copy()
    df_cal['year'] = df_cal['time'].dt.year
    df_cal['month'] = df_cal['time'].dt.month
    df_cal['day'] = df_cal['time'].dt.day
    df_cal['hour'] = df_cal['time'].dt.hour
    
    df_daily = df_cal.groupby(['year','month','day']).mean().reset_index()
    df_daily['time']=pd.to_datetime(df_daily[["year", "month", "day"]])
    df_daily["time"] = pd.to_datetime(df_daily["time"])
    comparison(
        df_daily,
        label=label,
        var=var,
        variables_units=variables_units_dict
    )    


################# utilities to give formatting to data and subseting to specific depth for H2OSOI

def ctsm_sim_depth(sim_files, var, levsoi=2.5):
    """
    Extract CTSM simulation data for the closest available soil depth level.
    
    Parameters:
    -----------
    sim_files : list or str
        Path(s) to CTSM simulation files
    var : str
        Variable name to extract
    levsoi : float
        Target soil depth level (will find closest available), only .5 increments from 0 to 6m
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with data for the closest soil depth
    """
    ds_ctsm = xr.open_mfdataset(sim_files, decode_times=True, combine='by_coords')
    
    # Convert to DataFrame
    df_ctsm_handle = ds_ctsm[var].to_dataframe().reset_index()
    
    # Define depth mapping based on available depths
    if levsoi > 6:
        print("Depths from 0-6m only")
        ds_ctsm.close()
        return None
    
    elif levsoi == 0:
        # Closest to surface: 0.009999999776482582
        df_ctsm_handle = df_ctsm_handle[df_ctsm_handle['levsoi'] <= 0.02]
    
    elif levsoi == 0.5:
        # Closest: 0.5799999833106995
        df_ctsm_handle = df_ctsm_handle[df_ctsm_handle['levsoi'].between(0.57, 0.59)]
    
    elif levsoi == 1:
        # Closest: 1.059999942779541
        df_ctsm_handle = df_ctsm_handle[df_ctsm_handle['levsoi'].between(1.05, 1.07)]
        
    elif levsoi == 1.5:
        # Closest: 1.3600000143051147
        df_ctsm_handle = df_ctsm_handle[df_ctsm_handle['levsoi'].between(1.35, 1.37)]
        
    elif levsoi == 2:
        # Closest: 2.0799999237060547
        df_ctsm_handle = df_ctsm_handle[df_ctsm_handle['levsoi'].between(2.07, 2.09)]
    
    elif levsoi == 2.5:
        # Exact match: 2.5
        df_ctsm_handle = df_ctsm_handle[df_ctsm_handle['levsoi'] == 2.5]
    
    elif levsoi in [3, 3.5]:
        # Closest: 3.5799999237060547
        df_ctsm_handle = df_ctsm_handle[df_ctsm_handle['levsoi'].between(3.57, 3.59)]
    
    elif levsoi in [4, 4.5]:
        # Closest: 4.269999980926514
        df_ctsm_handle = df_ctsm_handle[df_ctsm_handle['levsoi'].between(4.26, 4.28)]
        
    elif levsoi == 5:
        # Closest: 5.059999942779541
        df_ctsm_handle = df_ctsm_handle[df_ctsm_handle['levsoi'].between(5.05, 5.07)]
    
    elif levsoi == 5.5:
        # Closest: 5.949999809265137
        df_ctsm_handle = df_ctsm_handle[df_ctsm_handle['levsoi'].between(5.94, 5.96)]
    
    else:
        # If no predefined range, find the closest automatically
        available_depths = df_ctsm_handle['levsoi'].unique()
        closest_depth = available_depths[np.argmin(np.abs(available_depths - levsoi))]
        print(f"Using closest available depth: {closest_depth}m for requested {levsoi}m")
        df_ctsm_handle = df_ctsm_handle[
            np.abs(df_ctsm_handle['levsoi'] - closest_depth) < 0.001
        ]
    
    # Close the dataset to free memory
    ds_ctsm.close()
    
    return df_ctsm_handle