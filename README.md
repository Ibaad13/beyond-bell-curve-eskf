<div align="center">

# 🌆 Beyond the Bell Curve
## Urban GPS Multipath Characterization & Heavy-Tailed Filtering for MAV Navigation

<p>
Companion repository for the paper

<b>"Beyond the Bell Curve: Characterizing Urban GPS Multipath for MAV Navigation and the Limits of Heavy-Tailed Filtering"</b>
</p>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Paper](https://img.shields.io/badge/Paper-IEEE-red?style=for-the-badge)

---

**📄 Reproduces every figure and table from the paper directly from raw experimental data**

</div>

---

# 📖 Overview

Urban GPS errors are **not Gaussian**.

Instead, they exhibit

- 🎯 Heavy-tailed distributions
- ⏳ Strong temporal correlation
- 🏙️ Spatial correlation caused by urban multipath

This repository investigates these characteristics using two independent datasets and evaluates whether explicitly modeling GPS bias improves navigation performance.

---

# ✨ Highlights

- 📊 Statistical characterization of urban GPS multipath
- 📈 Distribution fitting using MLE
- 📉 QQ plots against Gaussian, Student-t, Laplace, Cauchy and Gaussian+Cauchy mixture
- 📍 Spatial semivariogram analysis
- ⏱ Temporal autocorrelation analysis
- 🚁 Evaluation on two independent MAV datasets
- 🛰 Comparison of four EKF architectures
- 🔬 Fully reproducible figures and tables

---

# 🏆 Main Findings

| Finding | Result |
|----------|--------|
| Best distribution | 🥇 Gaussian + Cauchy Mixture |
| AGZ Lag-1 Correlation | **0.984** |
| Spatial Correlation Range | **≈160 m** |
| Best Filter | **Bias-Augmented OU ESKF** |
| AGZ RMSE Improvement | **26.5%** |
| outdoor_1 p95 Improvement | **41.7%** |

---

# 🗂 Repository Structure

```text
📦 Repository
│
├── 📄 README.md
├── 📄 LICENSE
├── 📄 requirements.txt
│
├── 📚 paper/
│   ├── paper.pdf
│   ├── paper.tex
│   └── figures/
│
├── 📊 part1_dataset1_AGZ_characterization/
│   ├── characterize_dataset1.py
│   ├── zurich_gps_error.csv
│   ├── figures/
│   └── tables/
│
├── 📊 part2_dataset2_outdoor1_characterization/
│   ├── characterize_dataset2.py
│   ├── dataset2_gps_error.csv
│   ├── figures/
│   └── tables/
│
└── 🚁 part3_filtering_results/
    ├── filtering_results.py
    ├── data/
    ├── figures/
    └── tables/
```

---

# 🚀 Quick Start

Install dependencies

```bash
pip install -r requirements.txt
```

Run Part 1

```bash
cd part1_dataset1_AGZ_characterization
python characterize_dataset1.py
```

Run Part 2

```bash
cd ../part2_dataset2_outdoor1_characterization
python characterize_dataset2.py
```

Run Part 3

```bash
cd ../part3_filtering_results
python filtering_results.py
```

---

# 📊 Part I — AGZ Dataset Characterization

### Phase 1 — Distribution Fitting

We fit

- Gaussian
- Student-t
- Laplace
- Cauchy
- Gaussian + Cauchy Mixture

using Maximum Likelihood Estimation and compare

- AIC
- BIC
- KS Statistic

### Result

🏆 Gaussian + Cauchy Mixture consistently achieves the lowest AIC/BIC and best KS statistic.

<p align="center">
<img src="part1_dataset1_AGZ_characterization/figures/fig1_hist_pdf.png" width="46%">
<img src="part1_dataset1_AGZ_characterization/figures/fig2_qq.png" width="46%">
</p>

---

# 📈 Phase 2 — Correlation Analysis

The horizontal GPS error

\[
e_h=\sqrt{e_x^2+e_y^2}
\]

is analyzed using

- Autocorrelation Function (ACF)
- Spatial Semivariogram

### Result

- Lag-1 autocorrelation ≈ **0.984**
- Spatial correlation ≈ **160 m**

<p align="center">
<img src="part1_dataset1_AGZ_characterization/figures/fig3_acf.png" width="46%">
<img src="part1_dataset1_AGZ_characterization/figures/fig4_semivariogram.png" width="46%">
</p>

---

# 🌍 Part II — Cross-Dataset Validation

The complete characterization pipeline is repeated on the INSANE outdoor_1 dataset.

Key observations:

✅ Heavy tails persist

✅ Strong temporal correlation persists

✅ Spatial correlation varies with environment

<p align="center">
<img src="part2_dataset2_outdoor1_characterization/figures/d2_fig1_hist_pdf.png" width="46%">
<img src="part2_dataset2_outdoor1_characterization/figures/cross_fig_regression_bar.png" width="46%">
</p>

---

# 🚁 Part III — Heavy-Tailed Filtering

Four filtering architectures are evaluated.

| Method | Description |
|---------|-------------|
| EKF | Baseline 15-state ESKF |
| Naive Gating | Innovation rejection |
| IMM | Gaussian Sum / IMM |
| Bias-Augmented OU | 18-state ESKF with OU bias model |

---

# 📊 Results

| Metric | AGZ EKF | AGZ OU | outdoor_1 EKF | outdoor_1 OU |
|---------|---------|--------|---------------|--------------|
| RMSE | 5.97 | **4.39** | 4.06 | **3.95** |
| p95 | 11.24 | **6.97** | 8.08 | **4.71** |

---

<p align="center">
<img src="part3_filtering_results/figures/fig9_biasou_horiz_error.png" width="46%">
<img src="part3_filtering_results/figures/fig11_biasou_cdf.png" width="46%">
</p>

<p align="center">
<img src="part3_filtering_results/figures/fig10_estimated_bias.png" width="46%">
<img src="part3_filtering_results/figures/d2_fig12_rmse_bar_comparison.png" width="46%">
</p>

---

# 💡 Why State Augmentation Wins

Traditional innovation-based gating treats multipath as isolated outliers.

However, urban multipath forms **long-lived correlated biases**, meaning:

❌ Rejecting one measurement does not remove the following biased measurements.

Instead, the proposed filter estimates GPS bias directly as a state using an Ornstein-Uhlenbeck process, allowing the EKF to continuously track and compensate for slowly varying multipath errors.

---

# 🔄 Reproducibility

✅ Every figure in the paper can be regenerated.

✅ Every table is computed directly from saved `.npz` files.

✅ Paper figures and repository outputs remain synchronized.

---

# 📚 Citation

If this repository contributes to your research, please cite:

```bibtex
@article{BeyondBellCurve2026,
  title={Beyond the Bell Curve: Characterizing Urban GPS Multipath for MAV Navigation and the Limits of Heavy-Tailed Filtering},
  year={2026}
}
```

---

<div align="center">

⭐ If you found this repository useful, consider starring it!

Made with ❤️ for reproducible MAV navigation research.

</div>
