# Hosting di Vercel

Repositori ini sekarang menjalankan Flask melalui `app.py`, yang mengekspor `app` (WSGI).
HTML dashboard ada di `static/index.html`; CSS/JavaScript ada di `public/static/` agar dilayani sebagai aset statis Vercel. Grafik Plotly tidak memakai CDN pihak ketiga.
Model dan validasi CSV berjalan di Python. Tidak memerlukan API key, database, atau layanan ML lain.

## Pengaturan proyek

- Repository: `yzkielr/biodigest`, branch `main`.
- Root Directory: akar repository (kosong atau `.`).
- Framework Preset: **Flask** (juga diatur dalam `vercel.json`).
- Build Command, Output Directory, Install Command: gunakan default; hapus override Streamlit jika pernah dibuat.
- Versi Python: 3.12 melalui `.python-version`.
- Push ke main memicu deployment bila integrasi Git Vercel aktif. Buka deployment commit terbaru, bukan tombol redeploy pada commit lama.

## Menjalankan web lokal

```powershell
cd C:\1\Bekerja\sucofindo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Buka http://127.0.0.1:8000. Pada lingkungan pengembangan dengan dependensi terisolasi:

```powershell
$env:PYTHONPATH=(Resolve-Path .deploy-deps).Path
.\.venv\Scripts\python.exe app.py
```

## Endpoint

- `GET /api/demo?scenario=stable|ph|critical`: data simulasi, status aturan contoh, prediksi Random Forest.
- `POST /api/upload`: multipart CSV, unit m³/liter, confirmed=true. Maksimum 10.000 catatan dan file 3 MB untuk menyisakan ruang bagi multipart pada batas request hosting.
- `GET /api/template`: contoh CSV dengan dua baris simulasi.
- `POST /api/train`: evaluasi Ridge, Random Forest, dan baseline menggunakan urutan waktu.

Data unggahan tidak disimpan permanen, tidak dipakai melatih model, dan tidak mendapat prediksi.
Model demo dilatih dari bagian latihan data simulasi ketika proses pertama kali digunakan, lalu di-cache dalam memori.
Tidak memuat pickle model lama, sehingga tidak tergantung kompatibilitas paket yang membuat pickle tersebut.
Eksperimen tetap juga di-cache per proses; proses Vercel dapat berganti sehingga cache bukan penyimpanan permanen.
Model demo web memakai konfigurasi eksperimen ML (120 pohon), sehingga angka prediksinya dapat berbeda dari Streamlit lama.

## Versi Streamlit lama

Tersimpan di `streamlit_app.py`. Instal `requirements-streamlit.txt`, lalu jalankan
`python -m streamlit run streamlit_app.py`. Jangan gunakan perintah Streamlit untuk deployment Vercel.

## Verifikasi

`python -m unittest test_web -v` menguji halaman, aset, skenario, unggah/konversi CSV,
penolakan input invalid, dan evaluasi ML. `scripts/check_web_ui.py` memeriksa interaksi browser
(membutuhkan Playwright dan Microsoft Edge lokal; server harus berjalan pada port 8000).

Semua status dan prediksi merupakan demonstrasi, bukan penilaian keselamatan atau validasi performa lapangan.
