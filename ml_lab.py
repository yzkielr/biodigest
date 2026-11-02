"""An isolated, chronological ML experiment using bundled simulation data."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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


def render_lab():
    st.markdown('<div class="eyebrow">BELAJAR DENGAN DATA SIMULASI</div>', unsafe_allow_html=True)
    st.title("Latihan ML")
    st.write("Coba bagaimana komputer belajar dari catatan lama untuk memperkirakan produksi biogas hari berikutnya.")
    st.info("DATA SIMULASI · Hasil di halaman ini hanya menguji alur software. Belum membuktikan akurasi pada biodigester nyata.")
    for col, title, body in zip(st.columns(3),
            ["1 · Belajar dari masa lalu", "2 · Uji pada hari berikutnya", "3 · Bandingkan hasil"],
            ["Model belajar dari bagian awal catatan simulasi.",
             "Bagian akhir catatan disimpan untuk pengujian, tanpa mengacak urutan tanggal.",
             "Lihat apakah ML lebih baik daripada menebak produksi besok sama dengan hari ini."]):
        with col:
            with st.container(border=True):
                st.subheader(title)
                st.write(body)
    st.caption("Menggunakan data demo bawaan. Pilihan skenario monitoring tidak mengubah eksperimen ini.")
    if st.button("Mulai latihan & bandingkan model", type="primary", use_container_width=True):
        with st.spinner("Melatih dua model dan menghitung kesalahan prediksi…"):
            try:
                raw = pd.read_csv(ROOT / "data/biogas_data.csv", parse_dates=["timestamp"])
                st.session_state["ml_experiment"] = run_experiment(raw)
            except (ValueError, OSError) as exc:
                st.error(str(exc))
                return
    if "ml_experiment" not in st.session_state:
        st.write("Klik tombol hijau di atas. Tidak perlu mengunggah dataset atau mengatur model terlebih dahulu.")
        return
    experiment = st.session_state["ml_experiment"]
    scores, results = experiment["scores"], experiment["results"]
    baseline = float(scores.iloc[0]["MAE (m³)"])
    best = scores.iloc[1:].sort_values("MAE (m³)").iloc[0]
    st.subheader("Apa yang kita pelajari?")
    if best["MAE (m³)"] < baseline:
        st.success(f"Pada pengujian simulasi ini, {best['Metode']} memiliki kesalahan rata-rata lebih kecil daripada tebakan sederhana. Hasil ini belum berlaku untuk data nyata.")
    else:
        st.warning("Pada pengujian simulasi ini, kedua model ML belum mengalahkan tebakan sederhana. Artinya, memakai ML belum otomatis menghasilkan prediksi yang lebih baik.")
    cols = st.columns(3)
    cols[0].metric("Catatan untuk belajar", experiment["train_count"])
    cols[1].metric("Hari untuk pengujian", experiment["test_count"])
    cols[2].metric("Kesalahan ML terkecil (MAE)", f"{best['MAE (m³)']:.2f} m³")
    st.caption(f"Target latihan terakhir: {experiment['train_end']:%d/%m/%Y}. Target pengujian pertama: {experiment['test_start']:%d/%m/%Y}. Satu pasangan di batas pembagian dikeluarkan agar target latihan tidak masuk periode input pengujian.")
    st.subheader("Seberapa dekat prediksi dengan produksi simulasi?")
    selected = st.selectbox("Model yang ingin dilihat", scores.Metode.iloc[1:].tolist())
    chart = go.Figure()
    for name, color, dash in [("Produksi simulasi", "#087f65", "solid"), (selected, "#7956c7", "solid"), ("Tebakan sederhana", "#d58b24", "dash")]:
        chart.add_trace(go.Scatter(x=results["Tanggal prediksi"], y=results[name], name=name,
                                  mode="lines+markers", line=dict(color=color, dash=dash),
                                  hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f} m³<extra>%{fullData.name}</extra>"))
    chart.update_layout(template="plotly_white", height=390, xaxis_title="Tanggal produksi yang diprediksi",
                        yaxis_title="Total biogas harian (m³)", legend=dict(orientation="h", y=1.2), margin=dict(t=70))
    st.plotly_chart(chart, use_container_width=True)
    st.caption("Hijau adalah produksi simulasi yang menjadi pembanding. Ungu adalah prediksi ML. Oranye putus-putus adalah tebakan sederhana. Semakin dekat ke garis hijau, semakin kecil kesalahannya.")
    errors = scores.sort_values("MAE (m³)")
    bar = go.Figure(go.Bar(x=errors["MAE (m³)"], y=errors.Metode, orientation="h",
                           marker_color=["#d58b24" if n == "Tebakan sederhana" else "#7956c7" for n in errors.Metode],
                           text=errors["MAE (m³)"].map(lambda v: f"{v:.3f} m³"), textposition="auto"))
    bar.update_layout(template="plotly_white", height=260, xaxis_title="Rata-rata kesalahan (m³) · lebih kecil lebih baik",
                      yaxis=dict(autorange="reversed"), margin=dict(t=10, b=40))
    st.subheader("Metode mana yang paling sedikit meleset?")
    st.plotly_chart(bar, use_container_width=True)
    st.caption("MAE 0,30 m³ berarti prediksi rata-rata meleset sebesar 0,30 m³. Ini bukan batas pasti kesalahan untuk setiap hari.")
    with st.expander("Lihat angka evaluasi dan cara pengujian"):
        st.dataframe(scores.round(3), hide_index=True, use_container_width=True)
        st.write("MAE: rata-rata besar kesalahan. RMSE: lebih peka terhadap kesalahan besar. Keduanya lebih baik jika kecil. R² dapat negatif: model lebih buruk daripada pembanding konstan berupa rata-rata target pengujian. R² bukan persentase akurasi.")
        st.write("Sekitar 80% awal untuk latihan dan 20% akhir untuk pengujian. Prediksi dibuat satu hari ke depan dengan menganggap catatan hari ini sudah lengkap. Model tetap selama pengujian; riwayat harian diperbarui dari catatan yang sudah tersedia. Ini bukan prediksi sekaligus untuk seluruh periode mendatang.")
        st.write("Model dan pengaturan ditetapkan di awal. Perbandingan ini bersifat eksplorasi; memilih atau menyetel model berulang kali berdasarkan hasil uji membutuhkan periode validasi terpisah sebelum mengklaim performa akhir.")
    st.download_button("Unduh hasil latihan (CSV)", results.assign(sumber="simulasi").to_csv(index=False).encode("utf-8-sig"),
                       file_name="hasil_latihan_ml_SIMULASI.csv", mime="text/csv")
    st.caption("Latihan ini tidak mengganti model pada dashboard. Hasil tersimpan sementara selama sesi aplikasi.")
