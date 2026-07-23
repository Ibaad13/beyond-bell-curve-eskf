"""
Part 1 -- Dataset 1 (AGZ / Zurich) GPS Error Characterization
================================================================
Reproduces Phase 1-3 of the paper on the AGZ dataset (Table I, Table II,
Figs. 1-4). Output figure filenames match paper.tex \\includegraphics calls
exactly.

Run:    python characterize_dataset1.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
from pathlib import Path

# ---- Please keep consistent figures across all three parts ----
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})
COL_GAUSS, COL_LAPLACE, COL_T, COL_CAUCHY, COL_MIX = (
    "tab:blue", "tab:orange", "tab:green", "tab:red", "black")
COL_BASELINE, COL_OU, COL_IMM, COL_GATING = (
    "tab:blue", "tab:purple", "tab:green", "tab:red")

OUT_FIG = Path("figures"); OUT_FIG.mkdir(exist_ok=True)
OUT_TAB = Path("tables");  OUT_TAB.mkdir(exist_ok=True)

df = pd.read_csv("zurich_gps_error.csv")
df = df.sort_values("t_s").reset_index(drop=True)
eh = df["e_h"].values
N = len(eh)
print(f"Loaded {N} epochs. mean(e_h)={eh.mean():.3f} std={eh.std():.3f} "
      f"range=[{eh.min():.2f},{eh.max():.2f}]")

# ---------------------------------------------------------------------
# PHASE 1: MLE distribution fitting -> AIC/BIC/KS  (Eqs. 3-7, Table I)
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

x0_init = [0.78, eh.mean() * 0.7, eh.std() * 0.5, eh.mean() * 1.5, eh.std() * 0.5]
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
table1.to_csv(OUT_TAB / "table1_distribution_fit.csv", index=False)
print("\nTable I - Distribution fit comparison:\n", table1.to_string(index=False))
print("\nFitted mixture params:", fits["Gauss+Cauchy mix."]["params"])

# ---- Fig 1: Histogram of e_h with fitted PDFs (figures/fig1_hist_pdf.png) ----
xs = np.linspace(0, eh.max(), 500)
plt.figure(figsize=(7, 5))
plt.hist(eh, bins=40, density=True, alpha=0.4, color="gray",
         edgecolor="white", label="Empirical $e_h$")
plt.plot(xs, stats.norm.pdf(xs, mu_g, sig_g), color=COL_GAUSS, lw=1.5, label="Gaussian")
plt.plot(xs, stats.laplace.pdf(xs, loc_l, scale_l), color=COL_LAPLACE, lw=1.5, label="Laplace")
plt.plot(xs, stats.t.pdf(xs, *t_params), color=COL_T, lw=1.5, label="Student-t")
plt.plot(xs, stats.cauchy.pdf(xs, *c_params), color=COL_CAUCHY, lw=1.5, label="Cauchy")
mix_pdf = w * stats.norm.pdf(xs, mu, sig) + (1 - w) * stats.cauchy.pdf(xs, x0m, gam)
plt.plot(xs, mix_pdf, color=COL_MIX, lw=2.2, label="Gauss+Cauchy mix.")
plt.xlabel("Horizontal GPS error $e_h$ (m)"); plt.ylabel("Density")
plt.title("Dataset 1: Histogram of $e_h$ with fitted PDFs")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_FIG / "fig1_hist_pdf.png"); plt.close()

# ---- Fig 2: Q-Q plots vs Gaussian and Student-t (figures/fig2_qq.png) ----
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, dist, name in [(axes[0], stats.norm(mu_g, sig_g), "Q-Q vs Gaussian"),
                        (axes[1], stats.t(*t_params), "Q-Q vs Student-t (fitted)")]:
    (osm, osr), (slope, intercept, r) = stats.probplot(eh, dist=dist)
    ax.scatter(osm, osr, s=14, color=COL_GAUSS if "Gaussian" in name else COL_T,
               alpha=0.7, edgecolors="none")
    ax.plot(osm, slope * osm + intercept, color="red", lw=1)
    ax.set_title(name); ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Ordered values")
plt.suptitle("Fig 2: Quantile-quantile plots")
plt.tight_layout(); plt.savefig(OUT_FIG / "fig2_qq.png"); plt.close()

# ---------------------------------------------------------------------
# PHASE 2: ACF (Eq. 8) + spatial semivariogram (Eq. 9)  -> Fig 3, Fig 4
# ---------------------------------------------------------------------
t = df["t_s"].values - df["t_s"].values[0]
t_grid = np.arange(0, t[-1], 1.0)
eh_grid = np.interp(t_grid, t, eh)

def acf(x, nlags):
    x = x - x.mean()
    var = np.dot(x, x)
    return np.array([1.0 if lag == 0 else np.dot(x[:-lag], x[lag:]) / var
                      for lag in range(nlags + 1)])

nlags = min(100, len(eh_grid) - 2)
rho = acf(eh_grid, nlags)
ci = 1.96 / np.sqrt(len(eh_grid))
print(f"\nLag-1 ACF = {rho[1]:.3f} (95% CI band = +/-{ci:.4f})")

# ---- Fig 3: ACF (figures/fig3_acf.png) ----
plt.figure(figsize=(7, 5))
plt.stem(range(nlags + 1), rho, basefmt=" ", linefmt="tab:blue", markerfmt=" ")
plt.axhline(ci, ls="--", c="r", lw=1, label="95% CI")
plt.axhline(-ci, ls="--", c="r", lw=1)
plt.xlabel("Lag (s)"); plt.ylabel("ACF of $e_h$")
plt.title("Fig 3: Autocorrelation of horizontal GPS error")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_FIG / "fig3_acf.png"); plt.close()

# along-track distance (cumulative arc length of GT trajectory)
dx = np.diff(df["x_gt"].values); dy = np.diff(df["y_gt"].values)
s = np.concatenate([[0], np.cumsum(np.hypot(dx, dy))])

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

lags = np.linspace(5, min(220, s[-1] * 0.4), 25)
gamma = semivariogram(s, eh, lags)
valid = ~np.isnan(gamma)
sill = np.nanmax(gamma)
range95_idx = np.argmax(gamma[valid] >= 0.95 * sill)
range95 = lags[valid][range95_idx]
print(f"Spatial correlation range (95% of sill) = {range95:.1f} m")

# ---- Fig 4: Spatial semivariogram (figures/fig4_semivariogram.png) ----
plt.figure(figsize=(7, 5))
plt.plot(lags[valid], gamma[valid], "o-", color="tab:blue", markersize=5)
plt.axvline(range95, ls="--", c="r", lw=1, label=f"95% range = {range95:.1f} m")
plt.xlabel("Along-track distance lag (m)"); plt.ylabel("Semivariance $e_h$ (m$^2$)")
plt.title("Fig 4: Spatial semivariogram of horizontal GPS error")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_FIG / "fig4_semivariogram.png"); plt.close()

# ---------------------------------------------------------------------
# PHASE 3: Flight-dynamics correlation  -> Table II
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
win = max(1, int(round(5.0 / np.median(np.diff(t)))))
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
table2 = pd.DataFrame(rows2)
table2.to_csv(OUT_TAB / "table2_flight_dynamics_correlation.csv", index=False)
print("\nTable II - Flight dynamics correlation:\n", table2.to_string(index=False))

print("\nDone. Figures -> figures/, Tables -> tables/")
