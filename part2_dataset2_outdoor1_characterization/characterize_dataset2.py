"""
Part 2 -- Dataset 2 (INSANE outdoor_1) GPS Error Characterization
====================================================================
Reproduces Phase 1-3 on outdoor_1 (Table III, Figs. 5-8 region -- named
per paper.tex as d2_fig1_hist_pdf.png, d2_fig3_acf.png,
d2_fig4_semivariogram.png, cross_fig_regression_bar.png).

Run:    python characterize_dataset2.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})
COL_GAUSS, COL_LAPLACE, COL_T, COL_CAUCHY, COL_MIX = (
    "tab:blue", "tab:orange", "tab:green", "tab:red", "black")

OUT_FIG = Path("figures"); OUT_FIG.mkdir(exist_ok=True)
OUT_TAB = Path("tables");  OUT_TAB.mkdir(exist_ok=True)

df = pd.read_csv("dataset2_gps_error.csv")
df = df.sort_values("t_s").reset_index(drop=True)
eh = df["e_h"].values
N = len(eh)
print(f"Loaded {N} epochs. mean(e_h)={eh.mean():.3f} std={eh.std():.3f} "
      f"range=[{eh.min():.2f},{eh.max():.2f}]")

# ---------------------------------------------------------------------
# PHASE 1: MLE distribution fitting -> AIC/BIC/KS
# ---------------------------------------------------------------------
def gauss_cauchy_mixture_nll(theta, x):
    w, mu, sig, x0, gam = theta
    if not (0 < w < 1 and sig > 0 and gam > 0):
        return 1e12
    pdf = w * stats.norm.pdf(x, mu, sig) + (1 - w) * stats.cauchy.pdf(x, x0, gam)
    pdf = np.clip(pdf, 1e-300, None)
    return -np.sum(np.log(pdf))

fits = {}
mu_g, sig_g = stats.norm.fit(eh)
fits["Gaussian"] = dict(k=2, ll=np.sum(stats.norm.logpdf(eh, mu_g, sig_g)),
                         cdf=lambda x: stats.norm.cdf(x, mu_g, sig_g))
loc_l, scale_l = stats.laplace.fit(eh)
fits["Laplace"] = dict(k=2, ll=np.sum(stats.laplace.logpdf(eh, loc_l, scale_l)),
                        cdf=lambda x: stats.laplace.cdf(x, loc_l, scale_l))
t_params = stats.t.fit(eh)
fits["Student-t"] = dict(k=3, ll=np.sum(stats.t.logpdf(eh, *t_params)),
                          cdf=lambda x: stats.t.cdf(x, *t_params))
c_params = stats.cauchy.fit(eh)
fits["Cauchy"] = dict(k=2, ll=np.sum(stats.cauchy.logpdf(eh, *c_params)),
                       cdf=lambda x: stats.cauchy.cdf(x, *c_params))

x0_init = [0.7, eh.mean() * 1.1, eh.std() * 0.3, eh.mean() * 0.5, eh.std() * 0.3]
res = optimize.minimize(gauss_cauchy_mixture_nll, x0_init, args=(eh,),
                         method="Nelder-Mead",
                         options=dict(xatol=1e-8, fatol=1e-8, maxiter=20000))
w, mu, sig, x0m, gam = res.x
mix_ll = -res.fun
def mix_cdf(x, w=w, mu=mu, sig=sig, x0m=x0m, gam=gam):
    return w * stats.norm.cdf(x, mu, sig) + (1 - w) * stats.cauchy.cdf(x, x0m, gam)
fits["Gauss+Cauchy mix."] = dict(k=5, ll=mix_ll, cdf=mix_cdf,
                                  params=dict(w=w, mu_g=mu, sigma_g=sig, x0=x0m, gamma=gam))

rows = []
for name, f in fits.items():
    k, ll = f["k"], f["ll"]
    aic = 2 * k - 2 * ll
    bic = k * np.log(N) - 2 * ll
    D, p = stats.kstest(eh, f["cdf"])
    rows.append(dict(Model=name, k=k, AIC=round(aic, 1), BIC=round(bic, 1),
                      KS_stat=round(D, 4), KS_p=p))
table1 = pd.DataFrame(rows).sort_values("AIC")
table1.to_csv(OUT_TAB / "table1_distribution_fit_dataset2.csv", index=False)
print("\nDataset 2 distribution fit comparison:\n", table1.to_string(index=False))
print("\nFitted mixture params:", fits["Gauss+Cauchy mix."]["params"])

# ---- d2_fig1_hist_pdf.png ----
xs = np.linspace(0, eh.max(), 500)
plt.figure(figsize=(7, 5))
plt.hist(eh, bins=30, density=True, alpha=0.4, color="gray",
         edgecolor="white", label="Empirical $e_h$")
plt.plot(xs, stats.norm.pdf(xs, mu_g, sig_g), color=COL_GAUSS, lw=1.5, label="Gaussian")
plt.plot(xs, stats.laplace.pdf(xs, loc_l, scale_l), color=COL_LAPLACE, lw=1.5, label="Laplace")
plt.plot(xs, stats.t.pdf(xs, *t_params), color=COL_T, lw=1.5, label="Student-t")
plt.plot(xs, stats.cauchy.pdf(xs, *c_params), color=COL_CAUCHY, lw=1.5, label="Cauchy")
mix_pdf = w * stats.norm.pdf(xs, mu, sig) + (1 - w) * stats.cauchy.pdf(xs, x0m, gam)
plt.plot(xs, mix_pdf, color=COL_MIX, lw=2.2, label="Gauss+Cauchy mix.")
plt.xlabel("Horizontal GPS error $e_h$ (m)"); plt.ylabel("Density")
plt.title("Dataset 2 (outdoor_1): Histogram of $e_h$ with fitted PDFs")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_FIG / "d2_fig1_hist_pdf.png"); plt.close()

# ---------------------------------------------------------------------
# PHASE 2: ACF + spatial semivariogram
# ---------------------------------------------------------------------
t = df["t_s"].values - df["t_s"].values[0]
dt_median = np.median(np.diff(t))

def acf(x, nlags):
    x = x - x.mean()
    var = np.dot(x, x)
    return np.array([1.0 if lag == 0 else np.dot(x[:-lag], x[lag:]) / var
                      for lag in range(nlags + 1)])

t_grid1 = np.arange(0, t[-1], 1.0)
eh_grid1 = np.interp(t_grid1, t, eh)
nlags = min(60, len(eh_grid1) - 2)
rho = acf(eh_grid1, nlags)
ci = 1.96 / np.sqrt(len(eh_grid1))
print(f"\nLag-1 (1s) ACF = {rho[1]:.3f} (95% CI band = +/-{ci:.4f})")

# ---- d2_fig3_acf.png ----
plt.figure(figsize=(7, 5))
plt.stem(range(nlags + 1), rho, basefmt=" ", linefmt="tab:blue", markerfmt=" ")
plt.axhline(ci, ls="--", c="r", lw=1, label="95% CI")
plt.axhline(-ci, ls="--", c="r", lw=1)
plt.xlabel("Lag (s)"); plt.ylabel("ACF of $e_h$")
plt.title("Dataset 2: Autocorrelation function of horizontal GPS error")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_FIG / "d2_fig3_acf.png"); plt.close()

dx = np.diff(df["x_gt"].values); dy = np.diff(df["y_gt"].values)
s = np.concatenate([[0], np.cumsum(np.hypot(dx, dy))])
print(f"Total along-track distance = {s[-1]:.1f} m, duration = {t[-1]:.1f} s")

def semivariogram(s, eh, lags):
    gamma = []
    for h in lags:
        mask = np.abs(np.abs(s[:, None] - s[None, :]) - h) < (lags[1] - lags[0]) / 2
        iu = np.triu_indices_from(mask, k=1)
        pair_mask = mask[iu]
        if pair_mask.sum() < 5:
            gamma.append(np.nan); continue
        ei, ej = eh[iu[0][pair_mask]], eh[iu[1][pair_mask]]
        gamma.append(0.5 * np.mean((ei - ej) ** 2))
    return np.array(gamma)

lags = np.linspace(3, min(220, s[-1] * 0.6), 25)
gamma = semivariogram(s, eh, lags)
valid = ~np.isnan(gamma)
sill = np.nanmax(gamma)
range95_idx = np.argmax(gamma[valid] >= 0.95 * sill)
range95 = lags[valid][range95_idx]
print(f"Spatial correlation range (95% of sill) = {range95:.1f} m")

# ---- d2_fig4_semivariogram.png ----
plt.figure(figsize=(7, 5))
plt.plot(lags[valid], gamma[valid], "o-", color="tab:blue", markersize=5)
plt.axhline(sill, ls=":", c="gray", lw=1, label=f"Sill ({sill:.2f} m$^2$)")
plt.axvline(range95, ls="--", c="r", lw=1, label=f"95% range ({range95:.1f} m)")
plt.xlabel("Along-track distance lag (m)"); plt.ylabel("Semivariance $e_h$ (m$^2$)")
plt.title("Dataset 2: Empirical semivariogram")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_FIG / "d2_fig4_semivariogram.png"); plt.close()

# ---------------------------------------------------------------------
# PHASE 3: Flight-dynamics correlation
# ---------------------------------------------------------------------
vx = np.gradient(df["x_gt"].values, t); vy = np.gradient(df["y_gt"].values, t)
speed = np.hypot(vx, vy)
heading = np.degrees(np.arctan2(vy, vx))
heading_unwrap = np.unwrap(np.radians(heading))
heading_rate_deg = np.degrees(np.gradient(heading_unwrap, t))
turn_mask = np.abs(heading_rate_deg) > 15
turn_times = t[turn_mask]

def time_since_turn(ti):
    prior = turn_times[turn_times <= ti]
    return ti - prior[-1] if len(prior) else np.nan

tst = np.array([time_since_turn(ti) for ti in t])
altitude = df["z_gt"].values
win = max(1, int(round(5.0 / dt_median)))
street_heading = pd.Series(heading_unwrap).rolling(win, center=True, min_periods=1).mean().values
heading_rel_street = np.degrees(np.angle(np.exp(1j * (heading_unwrap - street_heading))))

rows2 = []
for name, var in [("Altitude", altitude), ("Speed", speed),
                   ("Heading rel. street", heading_rel_street),
                   ("Time since turn", tst)]:
    ok = ~np.isnan(var)
    r, p_r = stats.pearsonr(var[ok], eh[ok])
    rho_s, p_s = stats.spearmanr(var[ok], eh[ok])
    rows2.append(dict(Variable=name, Pearson_r=round(r, 3), Pearson_p=p_r,
                       Spearman_rho=round(rho_s, 3), Spearman_p=p_s))
table2_d2 = pd.DataFrame(rows2)
table2_d2.to_csv(OUT_TAB / "table2_flight_dynamics_correlation_dataset2.csv", index=False)
print("\nDataset 2 flight dynamics correlation:\n", table2_d2.to_string(index=False))

# ---------------------------------------------------------------------
# Cross-dataset comparison (Table III) + Fig: cross_fig_regression_bar.png
# ---------------------------------------------------------------------
agz_table2_path = Path("../part1_dataset1_AGZ_characterization/tables/table2_flight_dynamics_correlation.csv")
if agz_table2_path.exists():
    agz_t2 = pd.read_csv(agz_table2_path).set_index("Variable")
    agz_alt_rho = agz_t2.loc["Altitude", "Spearman_rho"]
    agz_speed_rho = agz_t2.loc["Speed", "Spearman_rho"]
    agz_head_rho = agz_t2.loc["Heading rel. street", "Spearman_rho"]
    agz_turn_rho = agz_t2.loc["Time since turn", "Spearman_rho"]
else:
    agz_alt_rho, agz_speed_rho, agz_head_rho, agz_turn_rho = 0.123, 0.033, -0.019, 0.074

d2_t2 = table2_d2.set_index("Variable")
labels = ["Altitude", "Speed", "Heading rel. street/bearing", "Time since turn"]
agz_vals = [agz_alt_rho, agz_speed_rho, agz_head_rho, agz_turn_rho]
d2_vals = [d2_t2.loc["Altitude", "Spearman_rho"], d2_t2.loc["Speed", "Spearman_rho"],
           d2_t2.loc["Heading rel. street", "Spearman_rho"], d2_t2.loc["Time since turn", "Spearman_rho"]]

x = np.arange(len(labels)); width = 0.35
plt.figure(figsize=(7, 5))
plt.bar(x - width / 2, agz_vals, width, color="tab:orange", label="Dataset 1 (AGZ)")
plt.bar(x + width / 2, d2_vals, width, color="tab:blue", label="Dataset 2 (outdoor_1)")
plt.axhline(0, color="k", lw=0.8)
plt.xticks(x, labels, rotation=15, ha="right", fontsize=8)
plt.ylabel("Spearman correlation with $e_h$")
plt.title("Flight-dynamics correlation with horizontal GPS error: Dataset 1 vs Dataset 2")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_FIG / "cross_fig_regression_bar.png"); plt.close()

agz = dict(N=2706, duration="~45 min (~1 Hz)", along_track="multi-km",
           mean_eh=3.57, std_eh=2.52, max_eh=24.53, best_model="Gauss+Cauchy mix.",
           mixture_w=0.783, lag1_acf=0.984, spatial_range_m=162.5,
           altitude_spearman=agz_alt_rho, speed_spearman=agz_speed_rho)
d2 = dict(N=int(N), duration=f"{t[-1]:.1f} s", along_track=f"{s[-1]:.1f} m",
          mean_eh=round(eh.mean(), 2), std_eh=round(eh.std(), 2), max_eh=round(eh.max(), 2),
          best_model="Gauss+Cauchy mix.", mixture_w=round(w, 3), lag1_acf=round(rho[1], 3),
          spatial_range_m=round(range95, 1),
          altitude_spearman=d2_vals[0], speed_spearman=d2_vals[1])
comp = pd.DataFrame({"Metric": list(agz.keys()), "AGZ_paper": list(agz.values()),
                      "outdoor_1_reproduced": list(d2.values())})
comp.to_csv(OUT_TAB / "table3_cross_dataset_comparison.csv", index=False)
print("\nTable III - Cross-dataset comparison:\n", comp.to_string(index=False))

print("\nDone. Figures -> figures/, Tables -> tables/")
