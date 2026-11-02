import unittest
from pathlib import Path
from unittest.mock import patch

from csv_data import TEMPLATE, validate_csv
from streamlit.testing.v1 import AppTest


class CsvTests(unittest.TestCase):
    def test_template_and_conversion(self):
        frame, errors, _ = validate_csv(TEMPLATE.encode(), 'liter')
        self.assertFalse(errors)
        self.assertAlmostEqual(frame.gas_volume.iloc[0], .00001)

    def test_reject_bad_inputs(self):
        bad = [b'', b'timestamp,ph\n2026-01-01,7',
               TEMPLATE.replace('2026-01-02','2026-01-01').encode(),
               TEMPLATE.replace('35,7.0','35,15').encode(),
               TEMPLATE.replace('35,7.0',',7.0').encode(),
               TEMPLATE.replace('35,7.0','inf,7.0').encode(),
               TEMPLATE.replace('2026-01-01','01/01/2026').encode()]
        for raw in bad:
            with self.subTest(raw=raw):
                frame, errors, _ = validate_csv(raw)
                self.assertIsNone(frame)
                self.assertTrue(errors)

    def test_semicolon_and_gaps(self):
        raw = TEMPLATE.replace(',', ';').replace('7.0','7,0').replace('2026-01-02','2026-01-05')
        frame, errors, warnings = validate_csv(raw.encode())
        self.assertFalse(errors)
        self.assertEqual(frame.ph.iloc[0], 7)
        self.assertTrue(warnings)

    def test_dashboard_demo_and_upload(self):
        app = AppTest.from_file('streamlit_app.py').run(timeout=60)
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Unggah CSV').run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('file_uploader')), 1)
        # Test downstream rendering independently of the browser upload widget.
        frame, _, _ = validate_csv(TEMPLATE.encode())
        for sample in [frame, frame.head(1)]:
            with patch('csv_data.data_panel', return_value=sample):
                app.run(timeout=60)
                self.assertFalse(app.exception)
                self.assertTrue(any('DATA UNGGAHAN' in c.value for c in app.caption))
                self.assertTrue(any('Prediksi belum tersedia' in i.value for i in app.info))


if __name__ == '__main__':
    unittest.main()
