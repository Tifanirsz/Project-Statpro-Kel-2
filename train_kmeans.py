import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

RANDOM_STATE = 42
FITUR = ["YearsAtCompany", "YearsInCurrentRole", "TotalWorkingYears"]
K_FINAL = 4   # Jumlah cluster yang dipilih

# 1. Load Data
df = pd.read_csv("data/data_bersih.csv")

# 2. Scaling
scaler = StandardScaler()
X = scaler.fit_transform(df[FITUR])

# 3. Evaluasi K (Elbow, Silhouette, Davies-Bouldin)
hasil = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    hasil.append({
        "k": k, 
        "inertia": km.inertia_,
        "silhouette": silhouette_score(X, km.labels_),
        "davies_bouldin": davies_bouldin_score(X, km.labels_)
    })

evaluasi_k = pd.DataFrame(hasil).round(3)
print("=== Hasil Evaluasi K ===")
print(evaluasi_k)
evaluasi_k.to_csv("evaluasi_k.csv", index=False)

# 4. Model Final
km_final = KMeans(n_clusters=K_FINAL, n_init=10, random_state=RANDOM_STATE).fit(X)
df["cluster_raw"] = km_final.labels_

# Urutkan cluster dari masa kerja terkecil -> terbesar
urutan = df.groupby("cluster_raw")["YearsAtCompany"].mean().sort_values().index
peta = {int(lama): baru for baru, lama in enumerate(urutan)}
df["cluster"] = df["cluster_raw"].map(peta)

profil = df.groupby("cluster")[FITUR].mean().round(1)
profil["jumlah"] = df.groupby("cluster").size()
profil["persen"] = (profil["jumlah"] / len(df) * 100).round(1)

print("\n=== Profil Cluster ===")
print(profil)

# Penamaan Segmen Senioritas
NAMA_SEGMEN = {
    3: {0: "Junior", 1: "Menengah", 2: "Senior"},
    4: {0: "Junior", 1: "Baru Bergabung - Berpengalaman", 2: "Menengah", 3: "Senior"},
}[K_FINAL]

df["segmen"] = df["cluster"].map(NAMA_SEGMEN)
profil["segmen"] = profil.index.map(NAMA_SEGMEN)

# 5. Reduksi Dimensi (PCA) untuk plotting grafik di Streamlit
pca = PCA(n_components=2, random_state=RANDOM_STATE)
komp = pca.fit_transform(X)
df["PC1"], df["PC2"] = komp[:, 0], komp[:, 1]

# 6. Simpan Artefak untuk Teman UI
joblib.dump(km_final, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump({"fitur": FITUR, "peta": peta, "nama_segmen": NAMA_SEGMEN}, "model_info.pkl")
df.to_csv("hasil_cluster.csv", index=False)
profil.to_csv("profil_cluster.csv")

print("\nSukses! Semua artefak (.pkl dan .csv) berhasil disimpan.")