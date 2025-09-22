import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew, kurtosis

sns.set(style="whitegrid")

def load_and_prepare_data(filepath):
    """Load a CSV, parse dates, set index, check columns, and transform wind direction."""
    cols_needed = [
        "Time", "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
        "windspeed_10m", "windspeed_100m", "winddirection_10m", "winddirection_100m",
        "windgusts_10m", "Power"
    ]
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        print(f"Missing columns in {filepath}: {missing_cols}")
        return None
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time").sort_index()
    # Transform wind directions to radians, then sin/cos (circular encoding)
    for col in ["winddirection_10m", "winddirection_100m"]:
        rad = np.deg2rad(df[col])
        df[col + "_sin"] = np.sin(rad)
        df[col + "_cos"] = np.cos(rad)
    return df

def eda(df, loc):
    """Create EDA plots, print distribution stats, and answer guide questions."""
    print(f"\n--- EDA for {loc} ---")
    # Distribution of Power
    plt.figure(figsize=(8,4))
    sns.histplot(df["Power"], bins=50, kde=True)
    plt.title(f"Distribution of Power - {loc}")
    plt.show()
    plt.close()
    print(f"Skewness of Power: {skew(df['Power']):.2f}")
    print(f"Kurtosis of Power: {kurtosis(df['Power']):.2f}")
    # Guide Q1: Distribution
    if skew(df['Power']) > 1:
        print("Power is highly right-skewed, suggesting many low-output periods and rare high-output events.")
    elif skew(df['Power']) < -1:
        print("Power is highly left-skewed, suggesting frequent high-output periods.")
    else:
        print("Power distribution is roughly normal or moderately skewed.")
    # Guide Q2: Seasonality
    df["month"] = df.index.month
    monthly_avg = df.groupby("month")["Power"].mean()
    monthly_avg.plot(marker="o", figsize=(8,4))
    plt.title(f"Average Power by Month - {loc}")
    plt.ylabel("Average Power")
    plt.show()
    plt.close()
    print("Monthly average Power:")
    print(monthly_avg)
    peak_month = monthly_avg.idxmax()
    print(f"Month with highest output: {peak_month} (1=Jan, 12=Dec)")
    # Correlation heatmap
    plt.figure(figsize=(10,8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title(f"Correlation Heatmap - {loc}")
    plt.show()
    plt.close()
    # Guide Q3: Most predictive variables
    corrs = corr["Power"].drop("Power").abs().sort_values(ascending=False)
    print("Top variables correlated with Power:")
    print(corrs.head(5))

def feature_engineering(df):
    """Add lag features, exogenous variables, rolling stats, and targets."""
    for lag in [1,24]:
        df[f"lag_{lag}"] = df["Power"].shift(lag)
    df["rolling_mean_24"] = df["Power"].rolling(window=24).mean()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["target_1"] = df["Power"].shift(-1)      # Hour-ahead target
    df["target_24"] = df["Power"].shift(-24)    # Day-ahead target
    df = df.dropna()
    return df

def get_features_targets(df, horizon):
    """For hour-ahead, horizon=1; for day-ahead, horizon=24"""
    weather_cols = [
        "temperature_2m","relativehumidity_2m","dewpoint_2m",
        "windspeed_10m","windspeed_100m","windgusts_10m",
        "winddirection_10m_sin","winddirection_10m_cos",
        "winddirection_100m_sin","winddirection_100m_cos"
    ]
    lag_cols = ["lag_1","lag_24","rolling_mean_24","hour","dayofweek"]
    X = df[weather_cols + lag_cols]
    y = df[f"target_{horizon}"]
    return X, y

def train_model(X_train, y_train):
    """Train RandomForest with GridSearchCV."""
    param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring="neg_mean_absolute_error")
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def evaluate(y_true, y_pred, kind, loc):
    """Print metrics, show error plots, and estimate financial implication."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{kind} - {loc}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    plt.figure(figsize=(12,4))
    plt.plot(y_true.values[:200], label="Actual")
    plt.plot(y_pred[:200], label="Predicted")
    plt.title(f"{kind} Forecast: Actual vs Predicted (First 200)")
    plt.legend()
    plt.show()
    plt.close()
    # Visualize error
    errors = y_true - y_pred
    plt.figure(figsize=(8,4))
    sns.histplot(errors, bins=50, kde=True)
    plt.title(f"{kind} Forecast Errors (Actual - Predicted) - {loc}")
    plt.show()
    plt.close()
    return mae, rmse, r2, errors

def estimate_lost_energy(y_true, y_pred, timestep_hr=1, wesm_price=5000):
    """Estimate lost MWh and PHP, assuming Power is normalized (0-1) and max power is 1 MW."""
    # Only count under-prediction (missed opportunity)
    lost = (y_true - y_pred).clip(lower=0).sum() * timestep_hr    # in MWh
    cost = lost * wesm_price
    return lost, cost

def main(locations, wesm_price=5000, timestep_hr=1):
    for loc in locations:
        print(f"\nProcessing {loc} ...")
        df = load_and_prepare_data(loc)
        if df is None: continue
        eda(df, loc)
        df = feature_engineering(df)
        split = int(len(df)*0.8)
        # Hour-ahead
        X, y = get_features_targets(df, horizon=1)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        model, params = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        mae, rmse, r2, errors = evaluate(y_test, y_pred, "Hour-ahead", loc)
        lost, cost = estimate_lost_energy(y_test, y_pred, timestep_hr, wesm_price)
        print(f"Hour-ahead lost MWh: {lost:.2f}, Financial loss: PHP {cost:,.2f}")
        # Day-ahead
        Xd, yd = get_features_targets(df, horizon=24)
        Xd_train, Xd_test = Xd[:split], Xd[split:]
        yd_train, yd_test = yd[:split], yd[split:]
        model_d, params_d = train_model(Xd_train, yd_train)
        yd_pred = model_d.predict(Xd_test)
        mae_d, rmse_d, r2_d, errors_d = evaluate(yd_test, yd_pred, "Day-ahead", loc)
        lost_d, cost_d = estimate_lost_energy(yd_test, yd_pred, timestep_hr, wesm_price)
        print(f"Day-ahead lost MWh: {lost_d:.2f}, Financial loss: PHP {cost_d:,.2f}")
        print(f"\nBest params (hour-ahead): {params}")
        print(f"Best params (day-ahead): {params_d}")

if __name__ == "__main__":
    locations = ["Location1.csv", "Location2.csv", "Location3.csv", "Location4.csv"]
    main(locations, wesm_price=5000, timestep_hr=1)
