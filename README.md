# Smart Biodigester MVP

**Versi web Vercel:** aplikasi utama sekarang menggunakan Flask (`app.py`) dengan dashboard
HTML/CSS/JavaScript. Jalankan `python app.py` dan buka http://127.0.0.1:8000.
Lihat [panduan deployment](DEPLOY.md) untuk instalasi, konfigurasi Vercel, dan endpoint API.
Versi Streamlit sebelumnya disimpan di `streamlit_app.py` untuk penggunaan lokal.

Prototype software untuk lomba: data biodigester -> machine learning -> prediksi -> dashboard.

## Fungsi MVP

- Menghasilkan dataset contoh yang menyerupai data sensor biodigester.
- Melatih model Random Forest untuk memprediksi produksi biogas 24 jam berikutnya.
- Menampilkan status Normal, Waspada, atau Kritis berdasarkan aturan demo, bukan klasifikasi ML.
- Menampilkan data, prediksi, faktor utama, dan rekomendasi di dashboard Streamlit.

## Menjalankan

```bash
cd sucofindo
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-streamlit.txt
python scripts/generate_data.py
python scripts/train_model.py
streamlit run streamlit_app.py
```

Buka URL yang ditampilkan Streamlit, biasanya `http://localhost:8501`.

## Catatan penting

Dataset bawaan adalah data simulasi untuk menguji alur aplikasi, bukan hasil pengukuran lapangan. Ganti `data/biogas_data.csv` dengan data sensor nyata setelah format dari anggota IoT sudah disepakati.

Format data minimum:

`timestamp,temperature,ph,methane,pressure,gas_volume,feed_amount`

## Latihan ML tanpa dataset nyata

Di menu samping, pilih **Halaman → Latihan ML**, lalu klik **Mulai latihan & bandingkan model**.
Halaman ini menggunakan simulasi bawaan untuk membandingkan Ridge dan Random Forest
dengan tebakan produksi besok sama dengan hari ini. Tersedia grafik produksi/prediksi,
grafik kesalahan, penjelasan metrik, dan unduhan hasil CSV berlabel simulasi.
Eksperimen tidak mengganti model dashboard dan hanya disimpan selama sesi berjalan.

Pengujian memakai urutan waktu (sekitar 80% awal untuk latihan, 20% akhir untuk uji),
dengan satu pasangan batas dikeluarkan agar target latihan mendahului input uji.
Target dan fitur riwayat dibangun ulang dari catatan harian. Hasil ini belum membuktikan
akurasi lapangan; pemilihan model lanjutan memerlukan validasi terpisah.

Untuk instalasi lokal yang sudah menggunakan `.deps`:

```powershell
cd C:\1\Bekerja\sucofindo
$env:PYTHONPATH=(Resolve-Path .deps).Path
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Pengembangan selanjutnya

1. Ganti data simulasi dengan data ESP32.
2. Tambahkan InfluxDB atau API sebagai sumber data.
3. Bandingkan Random Forest dengan H2O AutoML.
4. Validasi ambang status dengan anggota biodigester.
5. Tambahkan SHAP setelah model dan dataset nyata stabil.
