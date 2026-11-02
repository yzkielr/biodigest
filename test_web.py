import io
import unittest
from app import app
from data_validation import TEMPLATE
from app import demo_data
from ml_core import prepare_data, FEATURES
import pandas as pd

class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_assets_and_scenarios(self):
        for path in ['/', '/static/app.js', '/static/style.css', '/static/plotly.min.js']:
            with self.client.get(path) as response:
                self.assertEqual(response.status_code, 200)
        for scenario, expected in [('stable', 'NORMAL'), ('ph', 'WASPADA'), ('critical', 'KRITIS')]:
            response = self.client.get('/api/demo?scenario=' + scenario)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json['status'], expected)
            self.assertIsInstance(response.json['prediction'], float)
        self.assertEqual(self.client.get('/api/demo?scenario=invalid').status_code, 400)
        self.assertEqual(self.client.get('/api/demo').json['status'], 'NORMAL')

    def test_upload_validation_and_no_prediction(self):
        def upload(raw=TEMPLATE.encode(), confirmed='true', unit='liter'):
            return self.client.post('/api/upload', data={'file': (io.BytesIO(raw), 'data.csv'), 'confirmed': confirmed, 'unit': unit})
        self.assertEqual(upload(confirmed='false').status_code, 400)
        self.assertEqual(upload(b'timestamp,ph\n2026-01-01,7').status_code, 400)
        self.assertEqual(upload(unit='unknown').status_code, 400)
        self.assertEqual(upload(b'a'*(3*1024*1024+1)).status_code, 413)
        response = upload()
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json['prediction'])
        self.assertEqual(response.json['source'], 'unggahan')
        self.assertAlmostEqual(response.json['rows'][0]['gas_volume'], .00001)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_training_and_temporal_split(self):
        result = self.client.post('/api/train')
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(result.json['scores']), 3)
        self.assertEqual(len(result.json['results']), result.json['test_count'])
        self.assertLess(result.json['train_end'], result.json['test_start'])
        self.assertEqual(self.client.post('/api/train').json, result.json)

    def test_no_future_leakage(self):
        raw = demo_data()
        train, test = prepare_data(raw)
        self.assertLess(train.target_date.max(), test.timestamp.min())
        raw.loc[raw.timestamp >= test.timestamp.min(), 'gas_volume'] = 9999
        changed, _ = prepare_data(raw)
        pd.testing.assert_frame_equal(train[FEATURES + ['target']], changed[FEATURES + ['target']])

if __name__ == '__main__':
    unittest.main()
