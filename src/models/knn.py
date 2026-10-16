
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score


def treinar_knn(X_train, y_train, k=5):
    """Treina um KNN para regressão."""

    modelo = Pipeline([
        ("padronizacao", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])

    modelo.fit(X_train, y_train)

    return modelo

def avaliar_valores_k(X_train, y_train, grupos_train):
    """Compara diferentes valores de K por validação cruzada."""

    valores_k = [3, 5, 7, 9, 11, 15]
    cv = GroupKFold(n_splits=5)

    resultados = {}

    for k in valores_k:
        modelo = Pipeline([
            ("padronizacao", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=k))
        ])

        scores = cross_val_score(
            modelo,
            X_train,
            y_train,
            groups=grupos_train,
            cv=cv,
            scoring="neg_root_mean_squared_error"
        )

        resultados[k] = -scores.mean()

    return resultados

