"""An isolated, chronological ML experiment using bundled simulation data."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["temperature", "ph", "methane", "pressure", "gas_volume", "feed_amount",
            "gas_lag_1", "gas_lag_7", "gas_mean_3"]


def prepare_data(raw):
    data = raw.sort_values("timestamp").copy()
    if data.timestamp.duplicated().any() or not data.timestamp.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Latihan ini memerlukan satu catatan setiap hari, tanpa tanggal ganda atau hari yang terlewat.")
    data["gas_lag_1"] = data.gas_volume.shift(1)
    data["gas_lag_7"] = data.gas_volume.shift(7)
    data["gas_mean_3"] = data.gas_volume.rolling(3).mean()
    # Rebuild the target; never use target/lag columns from the source CSV.
    data["target"] = data.gas_volume.shift(-1)
    data["target_date"] = data.timestamp.shift(-1)
    data = data.dropna(subset=FEATURES + ["target", "target_date"])
    if len(data) < 40:
        raise ValueError("Diperlukan setidaknya 40 pasangan data lengkap setelah riwayat dan target disiapkan.")
    split = int(len(data) * .8)
    # Omit the boundary sample so training targets precede test inputs.
    return data.iloc[:split-1].copy(), data.iloc[split:].copy()


def run_experiment(raw):
    train, test = prepare_data(raw)
    predictions = {"Tebakan sederhana": test.gas_volume.to_numpy()}
    models = {
        "Regresi linear (Ridge)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(train[FEATURES], train.target)
        predictions[name] = model.predict(test[FEATURES])
    rows = []
    for name, prediction in predictions.items():
        rows.append({"Metode": name, "MAE (m³)": mean_absolute_error(test.target, prediction),
                     "RMSE (m³)": mean_squared_error(test.target, prediction) ** .5,
                     "R²": r2_score(test.target, prediction)})
    results = pd.DataFrame({"Tanggal prediksi": test.target_date.to_numpy(), "Produksi simulasi": test.target.to_numpy(), **predictions})
    return {"scores": pd.DataFrame(rows), "results": results,
            "train_count": len(train), "test_count": len(test),
            "train_end": train.target_date.max(), "test_start": test.target_date.min()}
