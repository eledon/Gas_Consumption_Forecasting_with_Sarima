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
- [Log Transformation](#log-transformation)
- [Modeling](#modeling)
- [Model Evaluation](#model-evaluation)
- [Forecast Comparison](#forecast-comparison)
- [Results Interpretation](#results-interpretation)
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

- **Trend:** Growth until early 2000s, then stabilization  
- **Seasonality:** Strong annual cycles — higher in winter, lower in summer  
- **Distribution:** Right-skewed, long-tailed (skewness = 0.73)  
- **Stationarity:** ADF and KPSS tests confirmed the need for seasonal differencing  
- **ACF/PACF:** Clear seasonal autocorrelation at lag 12

---

## 🔄 Log Transformation

- A **log transformation** was applied to reduce heteroscedasticity and stabilize variance.  
- It improved stationarity and helped satisfy normality assumptions required for ARIMA modeling.  
- After forecasting on the **log scale**, all predicted values were **back-transformed using the exponential function** to evaluate performance in the original scale (MCF).  
- This back-transformation was essential for interpreting the results meaningfully and computing accuracy metrics like MAE and RMSE.

---

## ⚙️ Modeling

### 🔹 ARIMA(2,0,1) — Baseline
- Built on log-transformed data without seasonal terms.  
- **Purpose:** Serve as a benchmark for non-seasonal performance.  
- **Result:** MAPE = 20.86%, residual autocorrelation and skewness.

### 🔹 SARIMA(2,0,1)(0,1,1)[12] — Auto Selected
- Applied on log scale with seasonal differencing.  
- **Purpose:** Capture seasonal patterns using `auto.arima()` with drift.  
- **Result:** MAPE = 6.48%, Theil’s U = 0.49, passed Ljung-Box but **failed Jarque-Bera** test, indicating non-normal residuals.

### 🔹 ETS(M,N,A) — Exponential Smoothing
- Multiplicative error model with additive seasonality.  
- **Purpose:** Provide a robust benchmark from a different modeling family.  
- **Result:** MAPE = 5.56%, lowest forecast error, but failed both **Ljung-Box** and **Jarque-Bera** tests.

---

## 🧪 Model Evaluation

| Metric           | ARIMA(2,0,1) | SARIMA(2,0,1)(0,1,1)[12] | ETS(M,N,A) |
|------------------|--------------|---------------------------|------------|
| **MAPE** (%)     | 20.86        | 6.48                      | **5.56**   |
| **RMSE** (MCF)   | 8337         | 5071                      | 4293       |
| **MAE** (MCF)    | 6851         | 2725                      | 2321       |
| **ACF1**         | 0.25         | -0.31                     | -0.53      |
| **Ljung-Box**    | ❌           | ✅                        | ❌         |
| **Jarque-Bera**  | ❌           | ❌                        | ❌         |
| **MAE/Mean** (%) | 17.1         | 6.8                       | **5.8**    |

**MAE (Mean Absolute Error)**: average size of prediction errors  
**RMSE (Root Mean Squared Error)**: penalizes large errors more heavily  
**MAPE (Mean Absolute Percentage Error)**: average error as a % of actuals  
**ACF1**: autocorrelation of residuals at lag 1 (should be close to 0)  
**MAE/Mean**: contextualizes error relative to average gas consumption (≈ 40,053 MCF)

---

## ✅ Forecast Comparison

- Forecasts were made for the **last 12 months of available data: October 2023 – September 2024**
- Models were trained on log-transformed data and **back-transformed to original scale** for error comparison.
- While ETS yielded the **lowest MAPE**, its residuals showed autocorrelation and deviation from normality.
- SARIMA balanced accuracy with stronger statistical diagnostics, despite failing the Jarque-Bera normality test.
- The basic ARIMA model lacked seasonality and underperformed in both accuracy and residual checks.

<img src="https://github.com/eledon/Gas_Consumption_Forecasting_with_Sarima/blob/main/Forecasts.jpg?raw=true" width="700" alt="Forecast Comparison">

*Figure: Forecast curves for ARIMA, SARIMA, and ETS on the test set (back-transformed)*

---

## 📌 Results Interpretation

Descriptive statistics for the original gas consumption data (in million cubic feet):

- **Min:** 15,058  
- **Max:** 88,358  
- **Mean:** 40,053  
- **Median:** 32,935  

### 🔎 Why we use Descriptive Statistics

Descriptive stats provide a **reference point** for evaluating the scale of forecast errors. For example:

- **ETS MAE = 2,321 MCF**, which is just **5.8% of the mean** — indicating very accurate forecasts.
- Errors also cover a small portion of the data range (~73,300 MCF), suggesting the models are not wildly off even at their worst.

✅ **Conclusion:** Both SARIMA and ETS models provide strong forecasts for monthly gas consumption in California. ETS is most accurate, but SARIMA offers the best balance of accuracy and residual reliability.

---

## 🚀 Getting Started

To reproduce this analysis:

```r
# Install required packages
source("install_packages.R")

# Knit the R Markdown report
rmarkdown::render("Gas_Consumption_in_California.Rmd")


