import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------- Config ----------
INPUT_PATH = r"C:\Users\tshab\Downloads\airpoll spreadsheet.xlsx"
SHEET_NAME = 0   # or use sheet name like "Sheet1"
RANDOM_STATE = 42
EXPORT_PATH = r"C:\Users\tshab\Downloads\airpoll_kmeans_results.xlsx"

# ---------- Load data ----------
df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
print("Loaded dataframe shape:", df.shape)
print(df.head())

# ---------- Basic EDA ----------
print("\n--- Info ---")
print(df.info())
print("\n--- Descriptive statistics ---")
print(df.describe(include='all').T)

# Adjust if you have non-feature columns (like IDs, dates)
non_feature_cols = []
features = [c for c in df.columns if c not in non_feature_cols]
X_raw = df[features].copy()

# ---------- Handle missing data ----------
num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_raw.columns if c not in num_cols]

if num_cols:
    X_raw[num_cols] = SimpleImputer(strategy="median").fit_transform(X_raw[num_cols])

if cat_cols:
    X_raw[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X_raw[cat_cols])

# ---------- Distribution plots ----------
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(X_raw[col], kde=True)
    plt.title(f"Distribution: {col}")
    plt.tight_layout()
    plt.show()   # <--- You can now right-click or use the save icon to download

# ---------- Correlation heatmap ----------
plt.figure(figsize=(10,8))
sns.heatmap(X_raw[num_cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.show()

# ---------- Prepare features ----------
if cat_cols:
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
else:
    X_enc = X_raw.copy()

scaler = StandardScaler()
X = scaler.fit_transform(X_enc)

# ---------- PCA for visualization ----------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2 = pca.fit_transform(X)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca2[:,0], y=X_pca2[:,1])
plt.title("PCA (2D projection)")
plt.show()

# ---------- Elbow & silhouette ----------
inertia, sil_scores = [], []
K_range = range(2,11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

plt.figure()
plt.plot(list(K_range), inertia, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow method")
plt.show()

plt.figure()
plt.plot(list(K_range), sil_scores, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.title("Silhouette scores")
plt.show()

best_k = K_range[int(np.argmax(sil_scores))]
print("Best k (by silhouette):", best_k)

# ---------- Final KMeans ----------
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
labels = kmeans.fit_predict(X)
df["kmeans_cluster"] = labels

centers_scaled = kmeans.cluster_centers_
centers_unscaled = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_unscaled, columns=X_enc.columns)
centers_df.index.name = "cluster"

# ---------- Profiling ----------
profile = df.groupby("kmeans_cluster")[features].agg(["count","mean","std"])

# PCA scatter by cluster
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca2[:,0], y=X_pca2[:,1], hue=labels, palette="tab10", legend="full")
plt.title(f"PCA by cluster (k={best_k})")
plt.legend(title="Cluster")
plt.show()

# ---------- Export Excel ----------
with pd.ExcelWriter(EXPORT_PATH, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="data_with_clusters", index=False)
    centers_df.to_excel(writer, sheet_name="cluster_centers")
    profile.to_excel(writer, sheet_name="cluster_profile")

print(f"\nExcel results saved to: {EXPORT_PATH}")


