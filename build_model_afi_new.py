
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_excel("train_inklusi_rf v3.xlsx")

# Fitur dan target model AFI
X = df[[
    'Jumlah Kantor BU', 'Agen Laku Pandai', 'Jumlah ATM', 'Jumlah Kantor BPR/S',
    'Jumlah Kantor Pegadaian', 'Jumlah Kantor PMV', 'Jumlah Kantor PNM',
    'Rekening Tabungan Perorangan BU', 'Rekening Kredit Perorangan BU',
    'Rekening Tabungan Perorangan BPR/S', 'Rekening Kredit Perorangan BPR/S',
    'Nominal Tabungan Perorangan BU (Rp)', 'Nominal Kredit Perorangan BU (Rp)',
    'Luas Terhuni', 'Jumlah penduduk', 'PDRB Per kapita', 'PDRB'
]]
y = df['AFI']

# Bangun model dan latih
model = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])
model.fit(X, y)

# Simpan model
joblib.dump(model, "model_afi.pkl")
print("Model AFI (baru) berhasil disimpan.")
