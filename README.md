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

### 🔹 ARIMA(2,0,2) — Baseline
- Built on log-transformed data without seasonal terms.  
- **Purpose:** Serve as a benchmark for non-seasonal performance.  
- **Result:** MAPE = 20.86%, significant residual autocorrelation.

### 🔹 SARIMA(2,0,1)(0,1,1)[12] — Auto Selected
- Applied on log scale with seasonal differencing.  
- **Purpose:** Capture seasonal patterns using `auto.arima()` with drift.  
- **Result:** MAPE = 6.48%, Theil’s U = 0.49, passed all residual diagnostics.

### 🔹 ETS(M,N,A) — Exponential Smoothing
- Multiplicative error model with additive seasonality.  
- **Purpose:** Provide a robust benchmark from a different modeling family.  
- **Result:** MAPE = 5.56%, lowest forecast error, but failed Ljung-Box and Jarque-Bera tests.

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

- Forecasts were generated on the **log scale** and **back-transformed** for evaluation in original units.
- While ETS yielded the **lowest MAPE**, its residuals violated independence and normality assumptions.
- SARIMA offered strong accuracy **and** clean residual diagnostics — making it the **most balanced and reliable model**.
- The basic ARIMA model lacked seasonality and underperformed as expected.

<img src="https://github.com/eledon/Gas_Consumption_Forecasting_with_Sarima/blob/main/Forecasts.jpg?raw=true" width="700" alt="Forecast Comparison">

*Figure: Forecast curves for ARIMA, SARIMA, and ETS on the test set (back-transformed)*

---

## 📌 Results Interpretation

Descriptive statistics for the original gas consumption data (in million cubic feet):

- **Min:** 15,058  
- **Max:** 88,358  
- **Mean:** 40,053  
- **Median:** 32,935  

### 🔎 Comparing Forecast Errors to Data Scale

| Model   | MAE   | RMSE  | MAPE   |
|---------|--------|--------|--------|
| SARIMA  | 2,725 | 5,071 | 6.48%  |
| ETS     | 2,321 | 4,293 | 5.56%  |

- The **ETS model's MAE** is about **5.8% of the mean** and just **2.6% of the data range** — a strong result.
- **SARIMA** performs nearly as well in error metrics and **adds statistical validity** (i.e., reliable uncertainty estimation).
- **MAPE under 7%** is generally considered excellent in real-world forecasting.

✅ **Conclusion:** Both SARIMA and ETS models deliver high-quality forecasts. SARIMA is preferred for its overall **balance between accuracy and diagnostics**.

---

