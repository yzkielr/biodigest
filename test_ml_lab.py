import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest
from ml_lab import FEATURES, prepare_data, run_experiment


class LabTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.read_csv(Path(__file__).parent / "data/biogas_data.csv", parse_dates=["timestamp"])

    def test_future_values_cannot_change_training(self):
        train, test = prepare_data(self.raw)
        self.assertLess(train.target_date.max(), test.timestamp.min())
        changed = self.raw.copy()
        changed.loc[changed.timestamp >= test.timestamp.min(), "gas_volume"] = 9999
        changed_train, _ = prepare_data(changed)
        pd.testing.assert_frame_equal(train[FEATURES + ["target"]], changed_train[FEATURES + ["target"]])
        with self.assertRaises(ValueError):
            prepare_data(self.raw.drop(index=20))

    def test_baseline_uses_previous_day(self):
        result = run_experiment(self.raw)
        previous = self.raw.set_index("timestamp").gas_volume
        for row in result["results"].to_dict("records"):
            self.assertEqual(row["Tebakan sederhana"], previous.loc[row["Tanggal prediksi"] - pd.Timedelta(days=1)])

    def test_page_train_and_switch_model(self):
        app = AppTest.from_file("app.py").run(timeout=60)
        app.sidebar.selectbox[0].set_value("Latihan ML").run(timeout=60)
        self.assertFalse(app.exception)
        app.button[0].click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 2)
        app.selectbox[0].set_value("Random Forest").run(timeout=60)
        self.assertFalse(app.exception)
        app.sidebar.selectbox[0].set_value("Dashboard monitoring").run(timeout=60)
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
