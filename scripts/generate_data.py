from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def main() -> None:
    rng = np.random.default_rng(42)
    periods = 180
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="D")
    hour = np.arange(periods)

    temperature = 35 + 0.45 * np.sin(hour / 30 * 2 * np.pi) + rng.normal(0, 0.12, periods)
    ph = 7.05 + 0.08 * np.sin(hour / 45 * 2 * np.pi) + rng.normal(0, 0.025, periods)
    methane = 54 + 1.8 * np.sin(hour / 60 * 2 * np.pi) + rng.normal(0, 0.35, periods)
    pressure = 101.8 + 1.8 * np.sin(hour / 18 * 2 * np.pi) + rng.normal(0, 0.35, periods)
    feed_amount = 10 + rng.normal(0, 0.55, periods)

    # Hubungan simulasi sederhana: kondisi mendekati optimal meningkatkan gas.
    gas_volume = (
        11.5
        + 0.65 * (temperature - 33)
        + 1.7 * (ph - 6.5)
        + 0.035 * (methane - 45)
        + 0.18 * (feed_amount - 8)
        + rng.normal(0, 0.12, periods)
    )
    gas_volume = np.clip(gas_volume, 2, None)

    # Sisipkan beberapa episode gangguan agar dashboard memiliki contoh peringatan.
    for start in (45, 105):
        end = min(start + 18, periods)
        ph[start:end] -= 0.65
        temperature[start:end] -= 2.3
        methane[start:end] -= 7
        gas_volume[start:end] -= 3.5

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature.round(2),
            "ph": ph.round(2),
            "methane": methane.round(2),
            "pressure": pressure.round(2),
            "gas_volume": gas_volume.round(2),
            "feed_amount": feed_amount.round(2),
        }
    )
    # Fitur riwayat membantu model membaca tren, bukan hanya satu titik sensor.
    data["gas_volume_lag_1d"] = data["gas_volume"].shift(1)
    data["gas_volume_lag_7d"] = data["gas_volume"].shift(7)
    data["gas_volume_rolling_3d"] = data["gas_volume"].rolling(3).mean()
    data["temperature_rolling_3d"] = data["temperature"].rolling(3).mean()
    data["target_gas_volume_24h"] = data["gas_volume"].shift(-1)
    data = data.dropna().reset_index(drop=True)
    data.to_csv(DATA_DIR / "biogas_data.csv", index=False)
    print(f"Saved {len(data)} rows to {DATA_DIR / 'biogas_data.csv'}")


if __name__ == "__main__":
    main()
