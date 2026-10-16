
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor


def treinar_knn_pca(X_train, y_train, k=3, componentes=0.95):
    """Treina KNN com redução de dimensionalidade por PCA."""

    modelo = Pipeline([
        ("padronizacao", StandardScaler()),
        ("pca", PCA(n_components=componentes)),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])

    modelo.fit(X_train, y_train)

    return modelo
