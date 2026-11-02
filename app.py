from pathlib import Path
from html import escape

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from csv_data import data_panel

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Smart Biodigester", page_icon="🌱", layout="wide")
st.markdown("""
<style>
.stApp, [data-testid="stHeader"] {background:#fff;color:#182b36;}
.block-container {max-width:1440px;padding-top:2.5rem;}
h1 {letter-spacing:-1.5px;}
[data-testid="stMetric"] {border:1px solid #e4e9ed;border-radius:14px;padding:18px;background:#fff;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
.eyebrow {color:#147d70;font-size:12px;letter-spacing:2px;font-weight:700;}
.status {border:1px solid #e4e9ed;border-radius:14px;padding:18px 22px;margin:16px 0 24px;}
.status small {display:block;color:#637580;margin-top:7px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px;}
.headbar {display:flex;justify-content:space-between;align-items:center;gap:18px;padding:19px 26px;border-radius:16px;margin-bottom:26px;box-shadow:0 8px 24px #182b3610;}
.headbar strong {font-size:20px;letter-spacing:1px;}
.headbar span {font-size:13px;}
.hero {display:flex;justify-content:space-between;align-items:center;gap:24px;margin-bottom:24px;}
.hero h1 {font-size:40px;padding:0;margin:6px 0 10px;}
.hero p {color:#64748b;margin:0;}
.hero-icon {font-size:48px;background:#ecfdf5;border-radius:24px;padding:16px 24px;}
.kpi {border:1px solid var(--edge);background:var(--tint);border-radius:20px;padding:24px;position:relative;overflow:hidden;height:220px;margin:12px 0;}
.kpi .label {font-size:13px;font-weight:650;color:#405563;}
.kpi .value {font-size:38px;font-weight:700;letter-spacing:-1.5px;color:var(--ink);margin:12px 0 3px;}
.kpi .description {font-size:12px;color:#536675;}
.kpi svg {width:100%;height:54px;margin-top:14px;}
.kpi .icon {float:right;font-size:21px;}
.sensor {height:214px;padding:20px;}
.sensor .value {font-size:30px;}
[data-baseweb="tab-list"] {gap:12px;border-bottom:1px solid #e8edf2;padding-bottom:10px;}
[data-baseweb="tab"] {padding:10px 18px;border-radius:10px;}
[data-baseweb="tab"][aria-selected="true"] {background:#e9f7f3;color:#087c65;font-weight:700;}
[data-testid="stPlotlyChart"] {border:1px solid #edf1f5;border-radius:18px;padding:8px;box-shadow:0 5px 20px #182b3605;}
@media(max-width:700px) {.hero h1{font-size:30px}.hero-icon{display:none}.headbar{align-items:flex-start;flex-direction:column}.kpi{height:auto;min-height:200px}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data/biogas_data.csv", parse_dates=["timestamp"]).sort_values("timestamp")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models/biogas_model.joblib")


def process_status(row):
    checks = [(row.ph < 6.7, "pH di bawah ambang demo 6,7"),
              (row.temperature < 33.5, "Suhu di bawah ambang demo 33,5 °C"),
              (row.methane < 48, "Metana di bawah ambang demo 48%"),
              (row.gas_volume < 9, "Produksi di bawah ambang demo 9 m³")]
    reasons = [text for condition, text in checks if condition]
    if len(reasons) >= 2:
        return "KRITIS", "#dc3545", reasons
    if reasons:
        return "WASPADA", "#f5c842", reasons
    return "NORMAL", "#13966a", ["Tidak ada parameter yang melewati ambang demo."]


def number(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def visual_card(label, value, description, series, ink, tint, icon, small=False):
    values = list(series)[-20:]
    low, high = min(values), max(values)
    points = " ".join(f"{i*300/max(len(values)-1,1):.1f},{45-(v-low)/max(high-low,0.001)*35:.1f}" for i,v in enumerate(values))
    st.markdown(f'''<div class="kpi {'sensor' if small else ''}" style="--ink:{ink};--tint:{tint};--edge:{ink}25">
    <span class="icon">{icon}</span><div class="label">{escape(label)}</div>
    <div class="value">{escape(value)}</div><div class="description">{escape(description)}</div>
    <svg viewBox="0 0 300 54" preserveAspectRatio="none" role="img" aria-label="Tren relatif nilai pada kartu">
    <polygon points="0,54 {points} 300,54" fill="{ink}" opacity="0.09"/>
    <polyline points="{points}" fill="none" stroke="{ink}" stroke-width="2.5" stroke-linejoin="round"/></svg></div>''', unsafe_allow_html=True)


def line_chart(frame, column, label, unit, color, forecast=None):
    chart = go.Figure(go.Scatter(
        x=frame.timestamp, y=frame[column], name="Data unggahan" if uploaded_mode else "Data simulasi",
        mode="lines+markers", line=dict(color=color, width=3), marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(" + ",".join(str(int(color[i:i+2],16)) for i in (1,3,5)) + ",0.07)",
        hovertemplate=f"%{{x|%d/%m/%Y}}<br>{label}: %{{y:.2f}} {unit}<extra></extra>"))
    if forecast is not None:
        chart.add_trace(go.Scatter(
            x=[frame.timestamp.iloc[-1], frame.timestamp.iloc[-1] + pd.Timedelta(days=1)],
            y=[frame[column].iloc[-1], forecast], name="Prediksi besok",
            mode="lines+markers", line=dict(color="#ae6b20", dash="dash", width=2.5),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="%{x|%d/%m/%Y}<br>Prediksi: %{y:.2f} m³<extra></extra>"))
    chart.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                        font=dict(color="#405563"), height=320,
                        margin=dict(l=12,r=12,t=12,b=12),
                        legend=dict(orientation="h", y=1.15), showlegend=forecast is not None,
                        xaxis=dict(title="Tanggal", tickformat="%d/%m", showgrid=False),
                        yaxis=dict(title=f"{label} ({unit})" if unit else label,
                                   gridcolor="#eef1f4", rangemode="tozero"))
    return chart


page = st.sidebar.selectbox("Halaman", ["Dashboard monitoring", "Latihan ML"])
if page == "Latihan ML":
    from ml_lab import render_lab
    render_lab()
    st.stop()

source = st.sidebar.radio("Sumber data", ["Demo bawaan", "Unggah CSV"])
uploaded_mode = source == "Unggah CSV"
if uploaded_mode:
    with st.expander("Data · unggah CSV dan pemeriksaan", expanded=True):
        data = data_panel()
    if data is None:
        st.stop()
else:
    scenario = st.sidebar.selectbox("Skenario demo", ["Kondisi stabil", "pH rendah", "Beberapa parameter bermasalah"], key="demo_scenario")
    descriptions = {
        "Kondisi stabil": "Memakai data simulasi bawaan. Parameter terakhir berada dalam rentang aturan demo.",
        "pH rendah": "pH diturunkan bertahap hingga 6,3 pada 7 hari terakhir. Parameter lain tetap. Status akhir: Waspada.",
        "Beberapa parameter bermasalah": "Dalam 7 hari terakhir, pH turun ke 6,2; suhu ke 31 °C; metana ke 44%; dan produksi ke 8 m³. Status akhir: Kritis.",
    }
    st.sidebar.caption(descriptions[scenario])
    st.sidebar.info("Simulasi untuk demonstrasi. Warna status berasal dari aturan contoh, bukan hasil prediksi ML.")
    data = load_data().copy(deep=True)
    changes = {"Kondisi stabil": {}, "pH rendah": {"ph": 6.3},
               "Beberapa parameter bermasalah": {"ph": 6.2, "temperature": 31.0, "methane": 44.0, "gas_volume": 8.0}}
    count = min(7, len(data))
    for field, end_value in changes[scenario].items():
        start_value = float(data[field].iloc[-count])
        data.loc[data.index[-count:], field] = [start_value + (end_value-start_value)*i/max(count-1,1) for i in range(count)]
    # Rebuild history features from the edited copy before predicting.
    data["gas_volume_lag_1d"] = data.gas_volume.shift(1)
    data["gas_volume_lag_7d"] = data.gas_volume.shift(7)
    data["gas_volume_rolling_3d"] = data.gas_volume.rolling(3).mean()
    data["temperature_rolling_3d"] = data.temperature.rolling(3).mean()
bundle = load_model()
latest = data.iloc[-1]
prediction = None if uploaded_mode else float(bundle["model"].predict(pd.DataFrame([latest[bundle["features"]].to_dict()]))[0])
status, color, reasons = process_status(latest)
date = latest.timestamp.strftime("%d/%m/%Y")

bar_text = "#44370b" if status == "WASPADA" else "#ffffff"
st.markdown(f'<div class="headbar" style="background:{color};color:{bar_text}"><strong>● &nbsp; {status}</strong><span>Status aturan demo &nbsp; / &nbsp; Catatan {date}</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div><div class="eyebrow">BIOGAS INTELLIGENCE / OVERVIEW</div><h1>Smart Biodigester</h1><p>Semua kondisi proses, dalam satu pandangan.</p></div><div class="hero-icon">🌿</div></div>', unsafe_allow_html=True)
st.caption(f"{'DATA UNGGAHAN' if uploaded_mode else 'DATA SIMULASI'} · Catatan terakhir: {date} · Interval: harian · Belum terhubung langsung ke sensor")
st.caption("Warna status mengikuti aturan contoh; belum menjadi penilaian keselamatan atau klasifikasi ML.")
if not uploaded_mode:
    st.info(f"Skenario aktif: {scenario}. {descriptions[scenario]} Perubahan hanya pada salinan data sesi ini. Prediksi model bersifat ilustratif, bukan bukti dampak perubahan parameter.")

overview, monitoring, detail = st.tabs(["Ringkasan produksi", "Monitoring sensor", "Evaluasi model & data"])
with overview:
    cols = st.columns(2)
    with cols[0]:
        visual_card("PRODUKSI TERAKHIR", f"{number(latest.gas_volume)} m³", "Volume pada catatan harian terakhir", data.gas_volume, "#087f65", "#edf9f4", "◈")
    with cols[1]:
        if prediction is None:
            st.info("Prediksi belum tersedia untuk data unggahan. Monitoring tetap dapat digunakan.")
        else:
            change = prediction-latest.gas_volume
            visual_card("PREDIKSI HARI BERIKUTNYA", f"{number(prediction)} m³", f"Selisih perkiraan: {number(change,2)} m³ dari catatan terakhir", [latest.gas_volume,prediction], "#6951c6", "#f3f0ff", "✦")
    st.caption("Model bawaan: prediksi masih demonstrasi. Nilai R² saat ini negatif; lihat penjelasan di tab Evaluasi model & data.")
    st.subheader("Kondisi pada catatan terakhir")
    cols = st.columns(4)
    for col, label, value, explanation, field, ink, tint, icon in zip(cols,
        ["Suhu", "Keasaman (pH)", "Kadar metana", "Tekanan"],
        [f"{number(latest.temperature)} °C", number(latest.ph, 2), f"{number(latest.methane)}%", f"{number(latest.pressure)} kPa"],
        ["Temperatur pada catatan terakhir.", "Semakin kecil pH, semakin asam.",
         "Bagian metana dalam campuran gas.", "Nilai tekanan pada catatan terakhir."],
        ["temperature","ph","methane","pressure"],
        ["#bc691d","#2879c0","#168a73","#8261b6"],
        ["#fff6eb","#edf6ff","#eefaF5","#f5f0fd"],
        ["☀","◉","◌","↗"]):
        with col:
            visual_card(label,value,explanation,data[field],ink,tint,icon,small=True)
    st.caption("Mini-grafik kartu sensor menunjukkan tren relatif 20 catatan terakhir; skala berbeda antar kartu. Gunakan grafik detail untuk membaca nilai.")

    st.subheader("Perubahan dari hari ke hari")
    period = st.radio("Periode semua grafik monitoring", [7, 14, 30, 90], index=2, horizontal=True,
                      format_func=lambda n: f"{n} hari terakhir")
    recent = data[data.timestamp >= data.timestamp.max() - pd.Timedelta(days=period-1)]
    st.caption(f"{recent.timestamp.iloc[0]:%d/%m/%Y} sampai {recent.timestamp.iloc[-1]:%d/%m/%Y} · Arahkan kursor ke grafik untuk membaca nilai tepatnya.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Produksi biogas**")
        st.caption("Garis hijau = data unggahan." if uploaded_mode else "Garis hijau = data simulasi. Garis putus-putus = perkiraan besok.")
        st.plotly_chart(line_chart(recent, "gas_volume", "Volume biogas", "m³", "#147d70", prediction), width="stretch")
        st.caption(f"Rentang periode ini: {number(recent.gas_volume.min())}–{number(recent.gas_volume.max())} m³ per catatan harian. Sumbu dimulai dari nol agar perubahan tidak tampak berlebihan.")
    with right:
        st.markdown("**Keasaman bahan (pH)**")
        st.caption("Garis biru menunjukkan pH. Nilai lebih rendah berarti lebih asam.")
        chart = line_chart(recent, "ph", "pH", "", "#557caa")
        chart.update_yaxes(range=[0,14], dtick=2)
        st.plotly_chart(chart, width="stretch")
        st.caption(f"Rentang pH periode ini: {number(recent.ph.min(),2)}–{number(recent.ph.max(),2)}. pH tidak memiliki satuan.")

    left, right = st.columns([1.6, 1])
    with left, st.container(border=True):
        st.markdown("**Kalender produksi**")
        st.caption("Setiap kotak = satu hari. Hijau lebih pekat = volume lebih besar; kotak kosong = di luar periode atau tidak ada data.")
        calendar = recent.assign(day=recent.timestamp.dt.dayofweek,
                                 week=recent.timestamp - pd.to_timedelta(recent.timestamp.dt.dayofweek, unit="D"))
        values = calendar.pivot(index="day", columns="week", values="gas_volume").reindex(range(7))
        dates = calendar.assign(label=calendar.timestamp.dt.strftime("%d/%m/%Y")).pivot(index="day", columns="week", values="label").reindex(range(7)).fillna("")
        heat = go.Figure(go.Heatmap(z=values.values, x=[v.strftime("%d/%m") for v in values.columns],
                                  y=["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"],
                                  customdata=dates.values, xgap=5, ygap=5,
                                  colorscale=[[0,"#e6f3ef"],[1,"#147d70"]], colorbar=dict(title="m³"),
                                  hoverongaps=False, hovertemplate="%{customdata}<br>Produksi: %{z:.2f} m³<extra></extra>"))
        heat.update_layout(template="plotly_white", height=310, margin=dict(l=10,r=10,t=10,b=10),
                           xaxis_title="Minggu dimulai", yaxis=dict(autorange="reversed"), paper_bgcolor="white")
        st.plotly_chart(heat, width="stretch")
    with right, st.container(border=True):
        st.markdown("**Hari terakhir vs perkiraan besok**")
        st.caption("Dua batang ini membandingkan volume, bukan akurasi model.")
        if prediction is None:
            st.info("Belum ada prediksi tervalidasi untuk dibandingkan dengan data unggahan.")
        else:
            bars = go.Figure(go.Bar(x=["Catatan terakhir", "Perkiraan besok"], y=[latest.gas_volume, prediction],
                                    marker_color=["#147d70", "#d3a365"],
                                    text=[number(latest.gas_volume)+" m³", number(prediction)+" m³"],
                                    textposition="outside", hovertemplate="%{x}<br>%{y:.2f} m³<extra></extra>"))
            bars.update_layout(template="plotly_white", height=310, margin=dict(l=10,r=10,t=20,b=10),
                               yaxis=dict(title="Volume (m³)", range=[0,max(latest.gas_volume,prediction)*1.25]), paper_bgcolor="white")
            st.plotly_chart(bars, width="stretch")

    with st.container(border=True):
        st.markdown("**Catatan pemeriksaan**")
        for reason in reasons:
            st.write(f"• {reason}")
        st.write("Langkah berikutnya: periksa kualitas data dan cocokkan ambang dengan karakteristik biodigester bersama tim teknis.")
        st.caption("Catatan ini dihasilkan oleh aturan demo, bukan SHAP. Rekomendasi perubahan operasi belum divalidasi.")

with monitoring:
    st.subheader("Pantau setiap parameter")
    st.caption(f"Mengikuti pilihan {period} hari di tab Ringkasan produksi. Grafik dipisah karena satuannya berbeda.")
    sensor_specs = [("temperature", "Suhu", "°C", "#c18339", 33.5),
                    ("ph", "Keasaman (pH)", "", "#557caa", 6.7),
                    ("methane", "Kadar metana", "%", "#147d70", 48),
                    ("pressure", "Tekanan", "kPa", "#8370ac", None)]
    for offset in (0, 2):
        columns = st.columns(2)
        for column_ui, (field, title, unit, hue, threshold) in zip(columns, sensor_specs[offset:offset+2]):
            with column_ui, st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(f"Terakhir: {number(latest[field],2)} {unit} · Min: {number(recent[field].min(),2)} · Maks: {number(recent[field].max(),2)}")
                fig = line_chart(recent, field, title, unit, hue)
                # A closer view reveals changes; the full units and scale remain visible.
                lo, hi = float(recent[field].min()), float(recent[field].max())
                if threshold is not None:
                    lo, hi = min(lo,threshold), max(hi,threshold)
                    fig.add_hline(y=threshold, line_dash="dot", line_color="#b18b58",
                                  annotation_text=f"Ambang demo {threshold}", annotation_position="top left")
                padding = max((hi-lo)*0.3, 0.15)
                fig.update_yaxes(range=[lo-padding,hi+padding], rangemode="normal")
                st.plotly_chart(fig, width="stretch")
                st.caption("Sumbu diperbesar agar perubahan kecil terlihat. " +
                           ("Garis titik-titik adalah aturan demo, bukan batas aman universal." if threshold is not None else "Belum ada ambang tekanan yang ditetapkan."))

    with st.container(border=True):
        st.markdown("**Riwayat status per hari**")
        st.caption("Hijau = Normal · Jingga = Waspada · Merah = Kritis. Status berasal dari jumlah parameter yang melewati ambang demo.")
        statuses = recent.apply(process_status, axis=1)
        timeline = go.Figure(go.Bar(x=recent.timestamp, y=[1]*len(recent),
                                   marker_color=[s[1] for s in statuses],
                                   customdata=[s[0] for s in statuses],
                                   hovertemplate="%{x|%d/%m/%Y}<br>Status demo: %{customdata}<extra></extra>"))
        timeline.update_layout(template="plotly_white", height=150, margin=dict(l=10,r=10,t=10,b=10),
                               yaxis=dict(visible=False), xaxis=dict(tickformat="%d/%m", title="Tanggal"), bargap=0.15)
        st.plotly_chart(timeline, width="stretch")
        counts = pd.Series([s[0] for s in statuses]).value_counts()
        st.caption(" · ".join(f"{name}: {counts.get(name,0)} hari" for name in ["NORMAL","WASPADA","KRITIS"]))

with detail:
    st.subheader("Evaluasi model bawaan · data simulasi")
    if uploaded_mode:
        st.info("Metrik dan grafik evaluasi ini tetap milik model demo, bukan hasil pengujian data unggahan Anda.")
    st.write("Evaluasi memakai bagian akhir dataset simulasi sebagai data uji. Angka ini belum menunjukkan performa pada biodigester nyata.")
    metrics = bundle["metrics"]
    cols = st.columns(3)
    cols[0].metric("R² · kecocokan prediksi", number(metrics["r2"],2))
    cols[0].caption("Nilai negatif berarti prediksi lebih buruk daripada memakai rata-rata nilai aktual data uji. Ini bukan persentase akurasi.")
    cols[1].metric("MAE · rata-rata kesalahan", f"{number(metrics['mae'],2)} m³")
    cols[1].caption("Rata-rata selisih mutlak prediksi dan nilai sebenarnya. Semakin kecil, semakin baik.")
    cols[2].metric("RMSE · kesalahan berbobot", f"{number(metrics['rmse'],2)} m³")
    cols[2].caption("Lebih peka terhadap kesalahan prediksi besar. Semakin kecil, semakin baik.")
    st.markdown("**Sudah digunakan:** Random Forest untuk prediksi dan aturan contoh untuk status.")
    st.caption("AutoML, SHAP, dan optimasi parameter belum diimplementasikan.")
    st.markdown("**Prediksi dibanding nilai aktual pada data uji**")
    st.caption("Hijau = nilai aktual simulasi, jingga putus-putus = prediksi. Semakin dekat kedua garis, semakin kecil kesalahannya.")
    evaluation_data = load_data()
    test = evaluation_data.iloc[int(len(evaluation_data)*0.8):]
    test_prediction = bundle["model"].predict(test[bundle["features"]])
    comparison = go.Figure()
    for name, values, hue, dash in [("Aktual simulasi",test.target_gas_volume_24h,"#147d70","solid"),
                                    ("Prediksi model",test_prediction,"#be843d","dash")]:
        comparison.add_trace(go.Scatter(x=test.timestamp+pd.Timedelta(days=1), y=values, name=name,
                                        mode="lines+markers", line=dict(color=hue,dash=dash),
                                        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f} m³<extra>%{fullData.name}</extra>"))
    comparison.update_layout(template="plotly_white",height=340,margin=dict(l=10,r=10,t=30,b=10),
                             legend=dict(orientation="h",y=1.15),xaxis=dict(title="Tanggal yang diprediksi",tickformat="%d/%m"),
                             yaxis=dict(title="Volume (m³)",rangemode="tozero"))
    st.plotly_chart(comparison,width="stretch")
    with st.expander("Lihat data yang digunakan"):
        st.caption("Kolom suhu dalam °C; metana dalam %; tekanan dalam kPa; produksi dalam m³ per catatan harian. Satuan jumlah bahan perlu dikonfirmasi sebelum integrasi sensor.")
        st.dataframe(data[["timestamp","temperature","ph","methane","pressure","gas_volume","feed_amount"]], hide_index=True, width="stretch")
