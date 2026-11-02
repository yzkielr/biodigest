"""Strict daily CSV contract; never impute or overwrite uploaded measurements."""
import csv
import io
import numpy as np
import pandas as pd

COLUMNS = ['timestamp', 'temperature', 'ph', 'methane', 'pressure', 'gas_volume', 'feed_amount']
TEMPLATE = ('timestamp,temperature,ph,methane,pressure,gas_volume,feed_amount\n'
            '2026-01-01,35,7.0,52,101.3,0.010,2\n'
            '2026-01-02,34,6.9,51,101.4,0.009,2\n')


def validate_csv(raw, volume_unit='m³'):
    errors, warnings = [], []
    if len(raw) > 5 * 1024 * 1024:
        return None, ['Ukuran file maksimal 5 MB.'], warnings
    try:
        text = raw.decode('utf-8-sig')
        delimiter = ';' if text.splitlines()[0].count(';') > text.splitlines()[0].count(',') else ','
        header = next(csv.reader(io.StringIO(text), delimiter=delimiter))
        if len(header) != len(set(header)):
            return None, ['Nama kolom tidak boleh berulang.'], warnings
        frame = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str)
    except (UnicodeError, ValueError, IndexError, StopIteration, pd.errors.ParserError):
        return None, ['File tidak dapat dibaca. Simpan sebagai CSV UTF-8 dengan pemisah koma atau titik koma.'], warnings
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        return None, ['Kolom belum tersedia: ' + ', '.join(missing)], warnings
    if frame.empty:
        return None, ['File belum memiliki baris pengukuran.'], warnings
    extras = [c for c in frame.columns if c not in COLUMNS]
    if extras:
        warnings.append('Kolom tambahan diabaikan: ' + ', '.join(extras))
    frame = frame[COLUMNS].copy()
    def report(mask, message):
        if mask.any():
            rows = ', '.join(str(i+2) for i in frame.index[mask][:8])
            errors.append(f'{message} Baris CSV: {rows}' + (' …' if mask.sum() > 8 else ''))
    dates = frame.timestamp.str.strip()
    parsed = pd.to_datetime(dates, format='%Y-%m-%d', errors='coerce')
    report(~dates.str.fullmatch(r'\d{4}-\d{2}-\d{2}', na=False) | parsed.isna(), 'Tanggal harus YYYY-MM-DD dan valid.')
    frame['timestamp'] = parsed
    report(parsed.notna() & parsed.duplicated(keep=False), 'Hanya satu catatan per tanggal; tanggal berulang ditemukan.')
    for col in COLUMNS[1:]:
        numeric = pd.to_numeric(frame[col].str.strip().str.replace(',', '.', regex=False), errors='coerce')
        report(~np.isfinite(numeric), f'{col}: isi angka lengkap, tanpa satuan, NaN, atau infinity.')
        frame[col] = numeric
    for col, low, high in [('ph',0,14),('methane',0,100),('temperature',-273.15,float('inf')),
                            ('pressure',0,float('inf')),('gas_volume',0,float('inf')),('feed_amount',0,float('inf'))]:
        report((frame[col] < low) | (frame[col] > high), f'{col}: nilai di luar rentang fisik yang diterima.')
    if errors:
        return None, errors, warnings
    if not frame.timestamp.is_monotonic_increasing:
        warnings.append('Tanggal diurutkan dari paling lama ke terbaru.')
    frame = frame.sort_values('timestamp').reset_index(drop=True)
    if volume_unit == 'liter':
        frame['gas_volume'] /= 1000
        warnings.append('Volume gas dikonversi dari liter menjadi m³ (1.000 liter = 1 m³).')
    if (frame.timestamp.diff().dropna() != pd.Timedelta(days=1)).any():
        warnings.append('Ada hari yang tidak tercatat. Data tidak diisi atau diinterpolasi otomatis.')
    return frame, errors, warnings
