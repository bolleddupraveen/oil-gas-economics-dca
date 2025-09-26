"""Arps Decline Curve Analysis (DCA) Python Script """

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math


def arps_rate(t, qi, di, b):

    t = np.array(t, dtype=float)
    if np.any(np.isclose(b, 0.0)):
        return qi * np.exp(-di * t)
    return qi / ((1.0 + b * di * t)**(1.0 / b))

def exp_rate(t, qi, di):
    return qi * np.exp(-di * t)

def harmonic_rate(t, qi, di):
    return qi / (1.0 + di * t)


def fit_arps(t_years, q_obs, p0=None, bounds=None):
    t = np.array(t_years, dtype=float)
    q = np.array(q_obs, dtype=float)

    if p0 is None:
        p0 = [max(q[0], 1.0), 0.3, 0.5]   # qi, di, b
    if bounds is None:
        bounds = ([1e-6, 1e-6, 0.0], [1e8, 5.0, 2.0])

    popt, pcov = curve_fit(arps_rate, t, q, p0=p0, bounds=bounds, maxfev=20000)
    q_fit = arps_rate(t, *popt)
    residuals = q - q_fit
    rss = np.sum(residuals**2)
    tss = np.sum((q - np.mean(q))**2)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    rmse = math.sqrt(rss / max(1, len(q) - len(popt)))
    return popt, pcov, residuals, r2, rmse

def fit_exponential(t_years, q_obs):
    t = np.array(t_years)
    q = np.array(q_obs)
    p0 = [max(q[0],1.0), 0.2]
    popt, pcov = curve_fit(exp_rate, t, q, p0=p0, bounds=([1e-6,1e-6],[1e8,5.0]), maxfev=20000)
    q_fit = exp_rate(t, *popt)
    residuals = q - q_fit
    rss = np.sum(residuals**2)
    tss = np.sum((q - np.mean(q))**2)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    rmse = math.sqrt(rss / max(1, len(q) - len(popt)))
    return popt, pcov, residuals, r2, rmse


def forecast_to_econ(popt, econ_limit=50.0, max_years=50, days_per_month=30):
    qi, di, b = popt
    months = np.arange(0, int(max_years*12)+1)
    t_years = months / 12.0
    rates = arps_rate(t_years, qi, di, b)
    idx = np.where(rates <= econ_limit)[0]
    last_idx = idx[0] if len(idx) > 0 else months[-1]
    months = months[:last_idx+1]
    t_years = months / 12.0
    rates = arps_rate(t_years, qi, di, b)
    monthly_bbl = rates * days_per_month
    cum_bbl = np.cumsum(monthly_bbl)
    df = pd.DataFrame({
        'month_index': months,
        'years': t_years,
        'rate_bbl_day': rates,
        'monthly_bbl': monthly_bbl,
        'cum_bbl': cum_bbl
    })
    return df


def param_std_errors(pcov):
    return np.sqrt(np.abs(np.diag(pcov)))


if __name__ == "__main__":

    t_years = np.linspace(0, 5, 60)  # 5 years monthly points
    true_qi = 1200.0
    true_di = 0.25
    true_b = 0.5
    q_bbl_day = arps_rate(t_years, true_qi, true_di, true_b) * (1 + 0.03 * np.random.randn(len(t_years)))


    p0 = [q_bbl_day[0], 0.25, 0.5]  # initial guess
    bounds = ([1e-3, 1e-6, 0.0], [1e7, 5.0, 2.0])
    popt, pcov, residuals, r2, rmse = fit_arps(t_years, q_bbl_day, p0=p0, bounds=bounds)
    se = param_std_errors(pcov)
    print("Arps fit results:")
    print(f"  qi = {popt[0]:.3f} ± {se[0]:.3f} bbl/day")
    print(f"  Di = {popt[1]:.6f} ± {se[1]:.6f} 1/yr")
    print(f"  b  = {popt[2]:.6f} ± {se[2]:.6f}")
    print(f"  R^2 = {r2:.4f}, RMSE = {rmse:.3f} bbl/day")


    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(t_years, q_bbl_day, 'o', label='Observed')
    t_fine = np.linspace(0, max(t_years), 400)
    plt.plot(t_fine, arps_rate(t_fine, *popt), '-', label='Arps fit', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Time (years)'); plt.ylabel('Rate (bbl/day)')
    plt.title('Rate vs Time (log scale)')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(t_years, residuals, '.-')
    plt.axhline(0, color='k', lw=0.5)
    plt.xlabel('Time (years)'); plt.ylabel('Residual (bbl/day)')
    plt.title('Residuals')
    plt.tight_layout()
    plt.show()


    popt_exp, pcov_exp, res_exp, r2_exp, rmse_exp = fit_exponential(t_years, q_bbl_day)
    print("\nExponential fit:")
    print(f"  qi = {popt_exp[0]:.3f}, Di = {popt_exp[1]:.6f}, R^2={r2_exp:.4f}, RMSE={rmse_exp:.3f}")

    def harmonic_param_rate(t, qi, di): return harmonic_rate(t, qi, di)
    p0_h = [q_bbl_day[0], 0.2]
    popt_h, pcov_h = curve_fit(harmonic_param_rate, t_years, q_bbl_day, p0=p0_h,
                               bounds=([1e-3,1e-6],[1e7,5.0]), maxfev=20000)
    qh_fit = harmonic_param_rate(t_years, *popt_h)
    res_h = q_bbl_day - qh_fit
    rss_h = np.sum(res_h**2)
    tss_h = np.sum((q_bbl_day - np.mean(q_bbl_day))**2)
    r2_h = 1 - rss_h / tss_h
    rmse_h = math.sqrt(rss_h / max(1, len(q_bbl_day)-len(popt_h)))
    print("\nHarmonic fit (b=1):")
    print(f"  qi={popt_h[0]:.3f}, Di={popt_h[1]:.6f}, R^2={r2_h:.4f}, RMSE={rmse_h:.3f}")


    econ_limit = 50.0
    forecast_df = forecast_to_econ(popt, econ_limit=econ_limit, max_years=50)
    print(f"\nForecast to economic limit {econ_limit} bbl/day:")
    print(f"Abandonment in {forecast_df['years'].iloc[-1]:.2f} years")
    print(f"EUR (to abandonment) = {forecast_df['cum_bbl'].iloc[-1]:,.0f} bbl")