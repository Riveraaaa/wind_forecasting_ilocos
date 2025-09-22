import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set(style="whitegrid")
locations = ["Location1.csv", "Location2.csv", "Location3.csv", "Location4.csv"]

for loc in locations:
    print(f"\nProcessing {loc}...\n")
    df = pd.read_csv(loc)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time").sort_index()
    
    # 3. EDA
    plt.figure(figsize=(8,4))
    sns.histplot(df["Power"], bins=50, kde=True)
    plt.title(f"Distribution of Power - {loc}")
    plt.show()
    
    df["month"] = df.index.month
    df.groupby("month")["Power"].mean().plot(marker="o", figsize=(8,4))
    plt.title(f"Average Power by Month - {loc}")
    plt.show()
    
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title(f"Correlation Heatmap - {loc}")
    plt.show()
    
    # 4. Feature Engineering
    df["lag1"] = df["Power"].shift(1)
    df["lag24"] = df["Power"].shift(24)
    df["target24"] = df["Power"].shift(-24)
    df = df.dropna()
    
    split = int(len(df)*0.8)
    
    # Hour-ahead forecast
    X_hour = df[["lag1"]]; y_hour = df["Power"]
    X_train, X_test = X_hour[:split], X_hour[split:]
    y_train, y_test = y_hour[:split], y_hour[split:]
    
    model_hour = RandomForestRegressor(random_state=42)
    model_hour.fit(X_train, y_train)
    y_pred_hour = model_hour.predict(X_test)
    
    print(f"Hour-ahead MAE ({loc}):", mean_absolute_error(y_test, y_pred_hour))
    print(f"Hour-ahead RMSE ({loc}):", mean_squared_error(y_test, y_pred_hour, squared=False))
    
    plt.figure(figsize=(12,4))
    plt.plot(y_test.values[:200], label="Actual")
    plt.plot(y_pred_hour[:200], label="Hour-ahead Predicted")
    plt.legend(); plt.title(f"Hour-ahead Forecast (sample) - {loc}")
    plt.show()
    
    # Day-ahead forecast
    X_day = df[["lag24"]]; y_day = df["target24"]
    X_train, X_test = X_day[:split], X_day[split:]
    y_train, y_test = y_day[:split], y_day[split:]
    
    model_day = RandomForestRegressor(random_state=42)
    model_day.fit(X_train, y_train)
    y_pred_day = model_day.predict(X_test)
    
    print(f"Day-ahead MAE ({loc}):", mean_absolute_error(y_test, y_pred_day))
    print(f"Day-ahead RMSE ({loc}):", mean_squared_error(y_test, y_pred_day, squared=False))
    
    plt.figure(figsize=(12,4))
    plt.plot(y_test.values[:200], label="Actual")
    plt.plot(y_pred_day[:200], label="Day-ahead Predicted")
    plt.legend(); plt.title(f"Day-ahead Forecast (sample) - {loc}")
    plt.show()
    
    # Estimate lost MWh and financial cost
    over_hour = (y_pred_hour - y_test).clip(lower=0).sum()/1000
    over_day = (y_pred_day - y_test).clip(lower=0).sum()/1000
    
    print(f"Hour-ahead lost energy ({loc}): {over_hour:.2f} MWh")
    print(f"Day-ahead lost energy ({loc}): {over_day:.2f} MWh")
    
    wesm_price = 5000
    print(f"Financial loss Hour-ahead ({loc}): PHP {over_hour*wesm_price:,.2f}")
    print(f"Financial loss Day-ahead ({loc}): PHP {over_day*wesm_price:,.2f}")
