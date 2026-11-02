from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "biogas_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "temperature",
    "ph",
    "methane",
    "pressure",
    "gas_volume",
    "feed_amount",
    "gas_volume_lag_1d",
    "gas_volume_lag_7d",
    "gas_volume_rolling_3d",
    "temperature_rolling_3d",
]
TARGET = "target_gas_volume_24h"


def main() -> None:
    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp").dropna()
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURES], train[TARGET])
    prediction = model.predict(test[FEATURES])

    metrics = {
        "mae": float(mean_absolute_error(test[TARGET], prediction)),
        "rmse": float(mean_squared_error(test[TARGET], prediction) ** 0.5),
        "r2": float(r2_score(test[TARGET], prediction)),
    }
    bundle = {"model": model, "features": FEATURES, "target": TARGET, "metrics": metrics}
    joblib.dump(bundle, MODEL_DIR / "biogas_model.joblib")
    print("Model saved to", MODEL_DIR / "biogas_model.joblib")
    print(metrics)


if __name__ == "__main__":
    main()
