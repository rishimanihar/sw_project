# Southwest Airlines Disruption Prediction

This project predicts flight disruptions (delays, cancellations, and diversions) for Southwest Airlines using a machine learning pipeline. It integrates hourly weather data from NOAA with flight performance data from the Bureau of Transportation Statistics (BTS).

## 📊 Model Performance
The model uses an **XGBoost Regressor** trained on 3 years of data (2022-2024) across 25 major US airports.
- **R-squared ($R^2$):** 0.132
- **RMSE:** 0.98 (on a standardized Disruption Index)
- **Key Findings:** Local weather accounts for ~13% of total flight disruption variance. The strongest predictors are "rolling" weather averages (accumulated stress) and time-of-day factors.

## 🛠️ Project Structure
- `main.py`: The end-to-end pipeline script. It handles:
  1.  **Data Scraping:** Downloads hourly weather data from NOAA (BTS data must be downloaded manually).
  2.  **Preprocessing:** Cleans weather data and aggregates flight data.
  3.  **Feature Engineering:** Creates lag features, rolling averages, and time cycles.
  4.  **Modeling:** Trains and evaluates the XGBoost model.
