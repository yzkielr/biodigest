"""WSGI entrypoint for Vercel and local development."""
import json
from functools import lru_cache
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, Response
from data_validation import validate_csv, TEMPLATE, COLUMNS
from ml_core import run_experiment, prepare_data, FEATURES
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder='public/static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

def records(frame):
    return json.loads(frame.to_json(orient='records', date_format='iso'))

def demo_data():
    return pd.read_csv(ROOT / 'data/biogas_data.csv', parse_dates=['timestamp'])[COLUMNS].sort_values('timestamp')

@lru_cache(maxsize=1)
def model():
    train, _ = prepare_data(demo_data())
    fitted = RandomForestRegressor(n_estimators=120, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=1)
    fitted.fit(train[FEATURES], train.target)
    return fitted

def summary(data, prediction=None, source='simulasi'):
    row = data.iloc[-1]
    reasons = [message for flag, message in [
        (row.ph < 6.7, 'pH di bawah ambang demo 6,7'),
        (row.temperature < 33.5, 'Suhu di bawah ambang demo 33,5 °C'),
        (row.methane < 48, 'Metana di bawah ambang demo 48%'),
        (row.gas_volume < 9, 'Produksi di bawah ambang demo 9 m³')] if flag]
    status = 'KRITIS' if len(reasons) >= 2 else 'WASPADA' if reasons else 'NORMAL'
    return dict(rows=records(data[COLUMNS]), prediction=prediction, status=status,
                reasons=reasons or ['Tidak ada parameter melewati ambang demo.'], source=source)

@app.get('/')
def index():
    return send_from_directory(ROOT / 'static', 'index.html')

@app.get('/api/demo')
def demo():
    changes = {'stable': {}, 'ph': {'ph': 6.3}, 'critical': {'ph': 6.2, 'temperature': 31., 'methane': 44., 'gas_volume': 8.}}
    scenario = request.args.get('scenario', 'stable')
    if scenario not in changes:
        return jsonify(error='Skenario tidak dikenal.'), 400
    data = demo_data().copy()
    for field, end in changes[scenario].items():
        start = float(data[field].iloc[-7])
        data.loc[data.index[-7:], field] = [start + (end-start)*i/6 for i in range(7)]
    latest = data.iloc[-1].to_dict()
    latest.update(gas_lag_1=data.gas_volume.iloc[-2], gas_lag_7=data.gas_volume.iloc[-8], gas_mean_3=data.gas_volume.iloc[-3:].mean())
    pred = float(model().predict(pd.DataFrame([latest])[FEATURES])[0])
    return jsonify(summary(data, pred))

@app.post('/api/upload')
def upload():
    file = request.files.get('file')
    if not file or not file.filename.lower().endswith('.csv'):
        return jsonify(error='Pilih file CSV.'), 400
    if request.form.get('confirmed') != 'true':
        return jsonify(error='Konfirmasikan satuan dan total gas harian terlebih dahulu.'), 400
    unit = request.form.get('unit', 'm³')
    if unit not in ['m³', 'liter']:
        return jsonify(error='Satuan volume tidak dikenal.'), 400
    raw = file.read()
    if len(raw) > 3 * 1024 * 1024:
        return jsonify(error='Batas unggahan web adalah 3 MB.'), 413
    frame, errors, warnings = validate_csv(raw, unit)
    if errors:
        return jsonify(error=' '.join(errors)), 400
    if len(frame) > 10000:
        return jsonify(error='Maksimal 10.000 catatan per unggahan. Bagi data menjadi beberapa file.'), 400
    return jsonify(**summary(frame, source='unggahan'), warnings=warnings)

@app.get('/api/template')
def template():
    return Response(TEMPLATE, mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=contoh_simulasi.csv'})

@lru_cache(maxsize=1)
def experiment():
    result = run_experiment(demo_data())
    return dict(scores=records(result['scores']), results=records(result['results']),
                train_count=result['train_count'], test_count=result['test_count'],
                train_end=result['train_end'].isoformat(), test_start=result['test_start'].isoformat())

@app.post('/api/train')
def train():
    return jsonify(experiment())

@app.errorhandler(413)
def too_large(error):
    return jsonify(error='File terlalu besar. Batas unggahan web adalah 3 MB.'), 413

@app.after_request
def headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'same-origin'
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
