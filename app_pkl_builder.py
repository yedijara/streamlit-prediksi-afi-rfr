# Ini builder .pkl dari preprocessing sampai training model

import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_excel("train_inklusi_rf2.xlsx")

# Preprocessing
df['Potential Underreported'] = df['Potential Underreported'].map({'Yes': 1, 'No': 0})
selected_columns = [
    'Kabupaten / Kota', 'Rek. Simpanan / Penduduk', 'Rek. Pinjaman / Penduduk',
    'Kantor Bank / km2', 'ATM / km2', 'Agen / km2', 'Kantor Non Bank / km2',
    'UFI - Kredit UMKM / PDRB', 'UFI - DPK Indv / PDRB', 'Jumlah penduduk',
    'PDRB Per kapita\n(Rp)', 'IPM', 'Tingkat Kemiskinan', 'Luas Terhuni\n(km2)',
    'Potential Underreported', 'AFI - Inklusi Keuangan'
]
df = df[selected_columns].dropna()

# Train model
X = df.drop(columns=['Kabupaten / Kota', 'AFI - Inklusi Keuangan'])
y = df['AFI - Inklusi Keuangan']
model = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])
model.fit(X, y)

joblib.dump(model, "model_afi_dengan_nilai_yang_saya_suka.pkl")