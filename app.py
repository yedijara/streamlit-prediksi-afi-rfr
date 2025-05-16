

import streamlit as st
import pandas as pd
import joblib
from streamlit.components.v1 import html

# Load model dari file .pkl
model = joblib.load("model_afi.pkl")

# Load data
df = pd.read_excel("train_inklusi_rf2.xlsx")

# Preprocessing
df['Potential Underreported'] = df['Potential Underreported'].map({'Yes': 1, 'No': 0})
selected_columns = [
    'Tahun', 'Kabupaten / Kota', 'Rek. Simpanan / Penduduk', 'Rek. Pinjaman / Penduduk',
    'Kantor Bank / km2', 'ATM / km2', 'Agen / km2', 'Kantor Non Bank / km2',
    'UFI - Kredit UMKM / PDRB', 'UFI - DPK Indv / PDRB', 'Jumlah penduduk',
    'PDRB Per kapita\n(Rp)', 'IPM', 'Tingkat Kemiskinan', 'Luas Terhuni\n(km2)',
    'Potential Underreported', 'AFI - Inklusi Keuangan'
]
df = df[selected_columns].dropna()

# Filter hanya tahun 2024
df = df[df['Tahun'] == 2024]

# Sidebar: user input
st.title("Simulasi Prediksi Skor AFI per Kabupaten/Kota (Baseline: Tahun 2024)")
city = st.selectbox("Pilih Kabupaten/Kota", sorted(df['Kabupaten / Kota'].unique()))
city_data = df[df['Kabupaten / Kota'] == city].iloc[0]

# Tampilkan data baseline
st.subheader("Data Baseline (2024)")
st.markdown(f"📍 Kabupaten/Kota: **{city}**")
st.markdown(f"Rekening Simpanan per 1000 penduduk: **{city_data['Rek. Simpanan / Penduduk']:.4f}**")
st.markdown(f"Rekening Pinjaman per 1000 penduduk: **{city_data['Rek. Pinjaman / Penduduk']:.4f}**")
st.markdown(f"Kantor Bank per km²: **{city_data['Kantor Bank / km2']:.4f}**")
st.markdown(f"Kantor Non Bank per km²: **{city_data['Kantor Non Bank / km2']:.4f}**")
st.markdown(f"Agen Laku Pandai per km²: **{city_data['Agen / km2']:.4f}**")
st.markdown(f"Skor AFI Saat Ini (2024): **{city_data['AFI - Inklusi Keuangan']:.4f}**")

st.markdown("---")
st.subheader("Simulasi Perubahan")

rek_simpanan = st.slider("Rek. Simpanan / Penduduk", -20.0, 100.0, float(city_data['Rek. Simpanan / Penduduk']))
rek_pinjaman = st.slider("Rek. Pinjaman / Penduduk", -20.0, 100.0, float(city_data['Rek. Pinjaman / Penduduk']))
kantor_bank = st.slider("Kantor Bank / km²", -20.0, 100.0, float(city_data['Kantor Bank / km2']))
kantor_nonbank = st.slider("Kantor Non Bank / km²", -20.0, 100.0, float(city_data['Kantor Non Bank / km2']))
agen = st.slider("Agen / km²", -200.0, 1000.0, float(city_data['Agen / km2']))

# Siapkan input prediksi
input_data = pd.DataFrame([{
    'Rek. Simpanan / Penduduk': rek_simpanan,
    'Rek. Pinjaman / Penduduk': rek_pinjaman,
    'Kantor Bank / km2': kantor_bank,
    'ATM / km2': city_data['ATM / km2'],
    'Agen / km2': agen,
    'Kantor Non Bank / km2': kantor_nonbank,
    'UFI - Kredit UMKM / PDRB': city_data['UFI - Kredit UMKM / PDRB'],
    'UFI - DPK Indv / PDRB': city_data['UFI - DPK Indv / PDRB'],
    'Jumlah penduduk': city_data['Jumlah penduduk'],
    'PDRB Per kapita\n(Rp)': city_data['PDRB Per kapita\n(Rp)'],
    'IPM': city_data['IPM'],
    'Tingkat Kemiskinan': city_data['Tingkat Kemiskinan'],
    'Luas Terhuni\n(km2)': city_data['Luas Terhuni\n(km2)'],
    'Potential Underreported': city_data['Potential Underreported']
}])

# Prediksi dan tampilkan hasil
prediksi_afi = model.predict(input_data)[0]
selisih = prediksi_afi - city_data['AFI - Inklusi Keuangan']

st.markdown("---")
st.subheader("Hasil Prediksi")
st.markdown(f"📊 Prediksi Skor AFI Baru: **{prediksi_afi:.4f}**")
st.markdown(f"🧮 Selisih dengan skor awal: **{selisih:+.4f}**")

st.markdown("---")
st.subheader("Peta AFI dan UFI")
st.markdown("[🗺️ Klik disini untuk membuka peta](https://yedijara.github.io/peta-afi-ufi/)", unsafe_allow_html=True)