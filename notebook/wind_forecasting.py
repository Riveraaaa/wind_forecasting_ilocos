# 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# 2. Load data
df = pd.read_csv("notebook/Location1.csv")  # <-- now points to your folder
df["Time"] = pd.to_datetime(df["Time"])
df = df.set_index("Time").sort_index()
df = df.asfreq("H").interpolate(method="time")

# 3. EDA
plt.figure(figsize=(8,4))
sns.histplot(df["Power"], bins=50, kde=True)
plt.title("Distribution of Power")
plt.show()

df["month"] = df.index.month
df.groupby("month")["Power"].mean().plot(marker="o", figsize=(8,4))
plt.title("Average Power by Month")
plt.show()

# 4. Feature engineering
df["hour"] = df.index.hour
df["dow"] = df.index.dayofweek
df["lag1"] = df["Power"].shift(1)
df["lag24"] = df["Power"].shift(24)
df["roll6"] = df["Power"].rolling(6, min_periods=1).mean()
df["roll24"] = df["Power"].rolling(24, min_periods=1).mean()

# Handle wind direction as cyclical features
for col in ["winddirection_10m", "winddirection_100m"]:
    radians = np.deg2rad(df[col])
    df[col+"_sin"] = np.sin(radians)
    df[col+"_cos"] = np.cos(radians)
df = df.drop(columns=["winddirection_10m","winddirection_100m"])

# Weather features
weather_cols = [c for c in df.columns if c not in ["Power","month"]]

# 5. Hour-ahead forecast
features = [c for c in weather_cols if c not in ["month"]] + ["lag1","lag24","roll6","roll24","hour","dow"]
df_model = df.dropna(subset=features+["Power"])

train_size = int(len(df_model)*0.8)
train = df_model.iloc[:train_size]
test = df_model.iloc[train_size:]

X_train, y_train = train[features], train["Power"]
X_test, y_test = test[features], test["Power"]

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xg = xgb.XGBRegressor(n_estimators=200, random_state=42)
xg.fit(X_train, y_train)
y_pred_xg = xg.predict(X_test)

def metrics(y, yhat):
    mae = mean_absolute_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    mape = np.mean(np.abs((y-yhat)/(y+1e-6)))*100
    return {"MAE":mae,"RMSE":rmse,"MAPE%":mape}

print("Hour-ahead RF:", metrics(y_test,y_pred_rf))
print("Hour-ahead XGB:", metrics(y_test,y_pred_xg))

# 6. Day-ahead forecast
df["target24"] = df["Power"].shift(-24)
features_da = features.copy()
df_da = df.dropna(subset=features_da+["target24"])

train_size = int(len(df_da)*0.8)
train_da = df_da.iloc[:train_size]
test_da = df_da.iloc[train_size:]

X_train_da, y_train_da = train_da[features_da], train_da["target24"]
X_test_da, y_test_da = test_da[features_da], test_da["target24"]

xg_da = xgb.XGBRegressor(n_estimators=300, random_state=42)
xg_da.fit(X_train_da, y_train_da)
y_pred_da = xg_da.predict(X_test_da)

print("Day-ahead XGB:", metrics(y_test_da,y_pred_da))

# 7. Visualizations
plt.figure(figsize=(12,4))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(y_pred_rf[:200], label="Hour-ahead RF")
plt.legend(); plt.title("Hour-ahead Forecast (sample)"); plt.show()

plt.figure(figsize=(12,4))
plt.plot(y_test_da.values[:200], label="Actual")
plt.plot(y_pred_da[:200], label="Day-ahead XGB")
plt.legend(); plt.title("Day-ahead Forecast (sample)"); plt.show()

# 8. Estimate lost MWh (over-forecasting periods)
over_hour = (y_pred_rf - y_test).clip(lower=0).sum()/1000
over_day = (y_pred_da - y_test_da).clip(lower=0).sum()/1000
print(f"Hour-ahead lost energy: {over_hour:.2f} MWh")
print(f"Day-ahead lost energy: {over_day:.2f} MWh")

wesm_price = 5000  # PHP/MWh
print(f"Financial loss Hour-ahead: PHP {over_hour*wesm_price:,.2f}")
print(f"Financial loss Day-ahead: PHP {over_day*wesm_price:,.2f}")
