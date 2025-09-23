This repository contains my end-to-end solution for forecasting wind turbine power output using weather features. The dataset comes from Kaggle (https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting?resource=download) 
and is adapted to simulate conditions in Ilocos Norte, Philippines, home to the country’s largest wind farms.

The project is framed around participation in the Wholesale Electricity Spot Market (WESM), where day-ahead forecasts and short-term balancing are critical to minimize penalties from imbalance settlements.

Project Objectives:
- Exploratory Data Analysis (EDA)
- Analyze distributions of power and weather features.
- Identify seasonality and patterns (e.g., monthly wind output trends).
- Investigate predictive relationships between wind speed, direction, and output.
- Estimate potential MWh lost during anomalous conditions.
Forecasting Models:
- Hour-ahead model for NGCP grid balancing.
- Day-ahead model for WESM bidding.
- Compare performance across horizons using historical and exogenous variables.
Evaluation & Impact:
- Metrics: MAE, RMSE, MAPE for accuracy.
- Visualize forecast errors to highlight financial implications of imbalance.
- Translate results into actionable insights for power plant operators.
Methods & Tools:
- Python (Jupyter Notebook)
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, xgboost
-Forecasting techniques: Time-series regression, Lagged features & weather exogenous variables, Gradient boosting models
