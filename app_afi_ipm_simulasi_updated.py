
import streamlit as st
import pandas as pd
import joblib

# Load model dari file .pkl
model_afi = joblib.load("model_afi.pkl")
model_ipm = joblib.load("model_ipm.pkl")

# Load data
df = pd.read_excel("train_inklusi_rf v3.xlsx")

# Drop kolom yang sudah tidak ada
df = df.dropna()

# Filter hanya tahun 2024
df = df[df['Tahun'] == 2024]

# Sidebar: user input
st.title("Simulasi Prediksi Skor AFI dan IPM per Kabupaten/Kota (Baseline: Tahun 2024)")
city = st.selectbox("Pilih Kabupaten/Kota", sorted(df['Kabupaten / Kota'].unique()))
city_data = df[df['Kabupaten / Kota'] == city].iloc[0]

# Tampilkan data baseline
st.subheader("Data Baseline (2024)")
st.markdown(f"📍 Kabupaten/Kota: **{city}**")
st.markdown(f"AFI Saat Ini: **{city_data['AFI']:.4f}**")
st.markdown(f"IPM Saat Ini: **{city_data['IPM']:.2f}**")

st.markdown("---")
st.subheader("Simulasi Perubahan")

# Input dari pengguna
jumlah_kantor_bu = st.slider("Jumlah Kantor BU", 0, 300, int(city_data['Jumlah Kantor BU']))
agen = st.slider("Agen Laku Pandai", 0, 2000, int(city_data['Agen Laku Pandai']))
atm = st.slider("Jumlah ATM", 0, 300, int(city_data['Jumlah ATM']))
bpr = st.slider("Jumlah Kantor BPR/S", 0, 100, int(city_data['Jumlah Kantor BPR/S']))
pegadaian = st.slider("Jumlah Kantor Pegadaian", 0, 100, int(city_data['Jumlah Kantor Pegadaian']))
pmv = st.slider("Jumlah Kantor PMV", 0, 50, int(city_data['Jumlah Kantor PMV']))
pnm = st.slider("Jumlah Kantor PNM", 0, 50, int(city_data['Jumlah Kantor PNM']))
rek_tab_bu = st.number_input("Rek. Tabungan BU", value=float(city_data['Rekening Tabungan Perorangan BU']))
rek_kredit_bu = st.number_input("Rek. Kredit BU", value=float(city_data['Rekening Kredit Perorangan BU']))
rek_tab_bprs = st.number_input("Rek. Tabungan BPR/S", value=float(city_data['Rekening Tabungan Perorangan BPR/S']))
rek_kredit_bprs = st.number_input("Rek. Kredit BPR/S", value=float(city_data['Rekening Kredit Perorangan BPR/S']))
nom_tab_bu = st.number_input("Nom. Tabungan BU (Rp)", value=float(city_data['Nominal Tabungan Perorangan BU (Rp)']))
nom_kredit_bu = st.number_input("Nom. Kredit BU (Rp)", value=float(city_data['Nominal Kredit Perorangan BU (Rp)']))
luas = st.number_input("Luas Terhuni", value=float(city_data['Luas Terhuni']))
penduduk = st.number_input("Jumlah Penduduk", value=float(city_data['Jumlah penduduk']))
pdrb_perkapita = st.number_input("PDRB Per Kapita", value=float(city_data['PDRB Per kapita']))
pdrb = st.number_input("PDRB", value=float(city_data['PDRB']))

# Siapkan data untuk prediksi AFI
input_afi = pd.DataFrame([{
    'Jumlah Kantor BU': jumlah_kantor_bu,
    'Agen Laku Pandai': agen,
    'Jumlah ATM': atm,
    'Jumlah Kantor BPR/S': bpr,
    'Jumlah Kantor Pegadaian': pegadaian,
    'Jumlah Kantor PMV': pmv,
    'Jumlah Kantor PNM': pnm,
    'Rekening Tabungan Perorangan BU': rek_tab_bu,
    'Rekening Kredit Perorangan BU': rek_kredit_bu,
    'Rekening Tabungan Perorangan BPR/S': rek_tab_bprs,
    'Rekening Kredit Perorangan BPR/S': rek_kredit_bprs,
    'Nominal Tabungan Perorangan BU (Rp)': nom_tab_bu,
    'Nominal Kredit Perorangan BU (Rp)': nom_kredit_bu,
    'Luas Terhuni': luas,
    'Jumlah penduduk': penduduk,
    'PDRB Per kapita': pdrb_perkapita,
    'PDRB': pdrb
}])

# Prediksi AFI
prediksi_afi = model_afi.predict(input_afi)[0]

# Siapkan data untuk prediksi IPM
input_ipm = input_afi.copy()
input_ipm.insert(0, 'AFI', prediksi_afi)

# Prediksi IPM
prediksi_ipm = model_ipm.predict(input_ipm)[0]

st.markdown("---")
st.subheader("Hasil Prediksi")
st.markdown(f"📊 Prediksi Skor AFI Baru: **{prediksi_afi:.4f}**")
st.markdown(f"📈 Prediksi IPM Baru: **{prediksi_ipm:.2f}**")
