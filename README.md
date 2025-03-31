# 🔮 Gas Consumption Forecasting with Time Series Models

Forecasting monthly residential gas consumption in California using ARIMA, SARIMA, and ETS models in R.

<img src="https://raw.githubusercontent.com/eledon/Gas_Consumption_Forecasting_with_Sarima/main/david-griffiths-Z3cBD6YZhOg-unsplash.jpg" width="500" height="300"/>

![R](https://img.shields.io/badge/R-TimeSeries-blue?logo=r)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Model](https://img.shields.io/badge/Model-ARIMA%2FSARIMA%2FETS-yellowgreen)
![Data](https://img.shields.io/badge/Data-EIA%20Gov-orange)

---

## 📘 Table of Contents
- [Overview](#overview)
- [Technologies](#technologies)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Model Evaluation](#model-evaluation)
- [Forecast Comparison](#forecast-comparison)
- [Getting Started](#getting-started)
- [Contact](#contact)

---

## 🧭 Overview

This project forecasts monthly residential gas consumption in California using three time series models: **ARIMA**, **SARIMA**, and **ETS**. The models are compared based on forecast accuracy and statistical validity of residuals to identify the most appropriate forecasting method.

---

## 🧪 Technologies

- **Language:** R
- **Libraries:** `forecast`, `ggplot2`, `tseries`, `urca`, `TSA`, `FinTS`, `DescTools`, `fUnitRoots`, `patchwork`, `dplyr`

---

## ❓ Research Question

> Which time series model (ARIMA, SARIMA, or ETS) provides the most accurate and statistically reliable forecasts of monthly residential gas consumption in California?

---

## 📊 Dataset

- **Source:** [U.S. EIA - California Natural Gas Consumption](https://www.eia.gov/dnav/ng/hist/n3010ca2m.htm)
- **Period:** January 1989 – September 2024 (monthly)
- **Unit:** Million Cubic Feet (MCF)
- **Missing data:** One missing value (January 2024) imputed with the historical January mean

---

## 🔍 Exploratory Data Analysis

- **Trend:** Clear upward trend until ~2000s, then stabilization and slight decline
- **Seasonality:** Strong annual cycles with winter peaks and summer lows
- **Distribution:** Right-skewed with high variability (skewness = 0.73)
- **Variance Stabilization:** Log transformation applied
- **Stationarity:** ADF and KPSS tests confirmed non-stationarity; seasonal differencing used
- **ACF/PACF:** Showed strong seasonal autocorrelation

---

## ⚙️ Modeling

### 🔹 ARIMA(2,0,2) — Baseline
- Built on log-transformed data without seasonal components.
- **Purpose:** Serve as a non-seasonal benchmark.
- **Result:** MAPE = 20.86%, underfit due to omission of seasonality.

### 🔹 SARIMA(2,0,1)(0,1,1)[12] — Auto Selected
- Built with seasonal differencing and seasonal MA term.
- **Purpose:** Capture seasonal structure automatically using `auto.arima()`.
- **Result:** MAPE = 6.48%, Theil’s U = 0.49, residuals passed most diagnostic tests.

### 🔹 ETS(M,N,A) — Exponential Smoothing
- Multiplicative error, no trend, additive seasonality.
- **Purpose:** Provide a benchmark from a different model family.
- **Result:** MAPE = 5.56%, lowest error, but **failed residual tests** (autocorrelation, non-normality).

---

## 🧪 Model Evaluation

| Metric           | ARIMA(2,0,2) | SARIMA(2,0,1)(0,1,1)[12] | ETS(M,N,A) |
|------------------|--------------|---------------------------|------------|
| **MAPE**         | 20.86%       | 6.48%                     | **5.56%**  |
| **RMSE**         | 8337         | 5071                      | 4293       |
| **MAE**          | 6851         | 2725                      | 2321       |
| **Residual ACF1**| 0.25         | -0.31                     | -0.53      |
| **Ljung-Box**    | ❌           | ✅                        | ❌         |
| **Jarque-Bera**  | ❌           | ✅                        | ❌         |

---

## ✅ Forecast Comparison

- The **ETS model** achieved the best forecast accuracy but failed key residual diagnostics (autocorrelation, normality).
- **SARIMA** achieved strong forecast performance and had **well-behaved residuals**, making it the **most balanced and reliable model** overall.
- The **ARIMA model** without seasonality underperformed on all metrics and was unable to capture the strong seasonal patterns.

<p align="center">
  <img src="https://github.com/eledon/Gas_Consumption_Forecasting_with_Sarima/blob/main/Forecasts.jpg?raw=true" width="700" alt="Forecast Comparison">
</p>

<p align="center"><em>Figure: Forecast curves for ARIMA, SARIMA, and ETS on the test set</em></p>

---

