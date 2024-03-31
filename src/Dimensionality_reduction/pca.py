# dimensionality_reduction/pca.py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca(df, target_column, n_components=2):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio for each Principal Component: {explained_variance_ratio}")

    return X_pca, explained_variance_ratio
