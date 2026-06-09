# Smooth Transition Regression & Macroeconomic Forecasting in MATLAB

This repository contains a comprehensive collection of MATLAB scripts and econometric procedures developed for descriptive analytics and non-linear regression modeling tasks. 

The project focuses on the simulation of Smooth Transition Regression (STR) models, algorithmic parameter optimization, and real-world macroeconomic forecasting using data from Poland.

## Key Features

- **STR Model Simulation**
  - Simulation of exponential smooth transition regression models.
  - Custom transition variables (e.g., $s_t = \cos(t)$) and regime-switching behavioral analysis.

- **Parameter Optimization & Grid Search**
  - Implementation of comprehensive 2D grid search algorithms to optimize non-linear transition parameters (slope $\lambda$ and threshold $c$).
  - Generation of loss surface heatmaps for visual diagnostics of optimization landscapes.

- **Advanced Estimation Algorithms**
  - Unconstrained parameter estimation using MATLAB's `fminunc` function.
  - Constrained optimization using `fmincon` with strict boundary definitions and Step Tolerance adjustments.

- **Macroeconomic Forecasting & Evaluation**
  - Modeling of Poland's annual GDP growth utilizing lagged unemployment, inflation, and renewable energy consumption.
  - **Data Source:** Macroeconomic data for this task is sourced from **World Bank Open Data**.
  - Comparative analysis between Logistic STR, Exponential STR, and baseline Linear Regression models.
  - Quantitative model evaluation balancing fit and complexity using Residual Sum of Squares (RSS) and Bayesian Information Criterion (BIC).

---

## Tech Stack & Methodology

1. Environment: **MATLAB**
2. Input data:
   - Formatted `.mat` data files (e.g., `Report4_3.mat`).
   - Macroeconomic datasets encompassing GDP, inflation, unemployment, and renewable energy metrics (sourced from World Bank Open Data).
3. Methodology:
   - Smooth Transition Regression (STR)
   - Ordinary Least Squares (OLS) estimation
   - Non-linear numerical optimization
   - Grid search methodology

---

# Repository Contents

1. [code&data](code&data)
   - Contains all MATLAB scripts and helper functions used throughout the project as well as all necessary datasets.
   - Main execution script:
     - [fourth_report.m](code&data/fourth_report.m)

2. [Nowicki_report.pdf](Nowicki_report.pdf)
   - Full report containing:
     - descriptive analytics and STR theory,
     - statistical estimation methods,
     - model verification and heatmaps,
     - real-world GDP modeling results,
     - conclusions based on BIC and RSS.

---

## 🛠 Functions Included
- `lossExp.m`: Evaluates the least squares loss score specifically for the exponential STR model.
- `loss.m`: Computes general loss scores utilized by the optimization algorithms (`fminunc` and `fmincon`).
- `OLS.m`: Computes standard regression coefficients vectors (baseline and differential).
