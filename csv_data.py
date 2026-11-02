"""Strict daily CSV contract; never impute or overwrite uploaded measurements."""
import csv
import io
import numpy as np
import pandas as pd
import streamlit as st

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


def data_panel():
    st.markdown('''<style>
    .upload-hero {padding:28px 30px;background:linear-gradient(110deg,#edf9f5,#f0f6ff);border:1px solid #deeee7;border-radius:20px;margin:6px 0 24px;}
    .upload-hero .tag {color:#137c66;font-size:12px;letter-spacing:2px;font-weight:700;}
    .upload-hero h2 {margin:8px 0;font-size:32px;color:#173a36;}
    .upload-hero p {color:#526773;margin:0;max-width:720px;}
    .upload-step {display:flex;gap:12px;align-items:center;margin:0 0 14px;color:#233f4c;font-weight:650;}
    .upload-step b {display:grid;place-items:center;width:32px;height:32px;border-radius:10px;background:#e5f4ee;color:#11745f;}
    .st-key-csv_upload_zone [data-testid="stFileUploaderDropzone"] {background:#f0faf6;border:2px dashed #36a78a;border-radius:16px;padding:30px 20px;min-height:170px;}
    .st-key-csv_upload_zone [data-testid="stFileUploaderDropzone"] button {background:#137c66;color:#fff;border:1px solid #137c66;border-radius:10px;min-height:48px;padding:10px 24px;font-weight:700;box-shadow:0 5px 14px #137c6625;}
    .st-key-csv_upload_zone [data-testid="stFileUploaderDropzone"] button:hover {background:#095c4a;border-color:#095c4a;}
    .st-key-csv_upload_zone button:focus-visible {outline:3px solid #61b7e5;outline-offset:3px;}
    .st-key-csv_template button {min-height:48px;font-weight:700;border-radius:10px;}
    @media(max-width:700px){.upload-hero{padding:22px}.upload-hero h2{font-size:26px}.st-key-csv_upload_zone [data-testid="stFileUploaderDropzone"]{flex-direction:column;gap:18px;}}
    </style><div class="upload-hero"><div class="tag">DATA CENTER / CSV</div>
<h2>Catatan Anda, jadi visual yang bermakna.</h2>
<p>Masukkan pengukuran harian biodigester. Kami periksa formatnya, lalu tampilkan kondisi dan tren dalam dashboard.</p></div>''', unsafe_allow_html=True)
    guide, upload = st.columns([1, 1.6], gap='large')
    with guide, st.container(border=True):
        st.markdown('<div class="upload-step"><b>1</b> Siapkan catatan Anda</div>', unsafe_allow_html=True)
        st.write('Mulai dari contoh tabel. Isi satu baris untuk satu hari dan satu biodigester.')
        st.download_button('↓  Unduh contoh CSV', TEMPLATE.encode('utf-8'), 'contoh_biodigester.csv', 'text/csv',
                           type='primary', width='stretch', key='csv_template')
        st.caption('Contoh berisi 2 baris simulasi. Ganti dengan pengukuran Anda.')
        st.markdown('**Format singkat**')
        st.write('Tanggal: **2026-01-01**  \nFile: **.csv** · maksimal **5 MB**  \nIsi: **7 kolom pengukuran**')
    with upload, st.container(border=True, key='csv_upload_zone'):
        st.markdown('<div class="upload-step"><b>2</b> Unggah file pengukuran</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader('Tarik file ke area ini atau klik Upload', type=['csv'], key='csv_file', max_upload_size=5)
        st.caption('CSV dengan pemisah koma atau titik koma • Pemeriksaan dimulai otomatis setelah file dipilih.')
        unit = st.selectbox('Satuan volume gas di dalam file', ['m³', 'liter'])
        confirmed = st.checkbox('Saya memastikan kolom lain sesuai panduan satuan dan volume gas adalah total harian.')
    with st.expander('Lihat panduan 7 kolom dan satuan'):
        st.table(pd.DataFrame({'Kolom': COLUMNS, 'Isi': [
            'Tanggal YYYY-MM-DD; satu catatan per hari', 'Suhu °C; rata-rata harian',
            'pH 0–14; rata-rata harian', 'Metana % (0–100); rata-rata harian',
            'Tekanan absolut kPa; rata-rata harian', 'Total gas yang dihasilkan hari itu, bukan meter kumulatif',
            'Total bahan masuk hari itu dalam kg']}))
        st.caption('Angka desimal memakai titik. CSV titik koma juga didukung, termasuk angka desimal koma. Jangan unggah file model atau data pribadi.')
    st.caption('File diproses dalam sesi aplikasi ini; tidak menimpa dataset demo atau melatih ulang model.')
    st.markdown('<div class="upload-step"><b>3</b> Hasil pemeriksaan & pratinjau</div>', unsafe_allow_html=True)
    if uploaded is None:
        st.info('Belum ada file dipilih. Gunakan tombol Upload di area hijau untuk mulai.')
        return None
    frame, errors, warnings = validate_csv(uploaded.getvalue(), unit)
    for message in errors:
        st.error(message)
    for message in warnings:
        st.warning(message)
    if errors:
        st.info('Perbaiki file kemudian unggah kembali. Data belum digunakan oleh dashboard.')
        return None
    st.success(f'{len(frame)} catatan lolos pemeriksaan format dan nilai. Kebenaran sensor dan satuan tetap perlu dikonfirmasi pengguna.')
    st.dataframe(frame, hide_index=True, width='stretch')
    if not confirmed:
        st.info('Konfirmasikan satuan di atas untuk menampilkan data ini di dashboard.')
        return None
    st.info('Data unggahan digunakan untuk monitoring. Prediksi dinonaktifkan karena model saat ini dilatih dengan simulasi dan belum tervalidasi untuk data Anda.')
    return frame
