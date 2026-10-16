
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def criar_knn_classificador(k=5):
    """Cria um pipeline KNN para classificação."""

    modelo = Pipeline([
        ("ausentes", SimpleImputer(
            strategy="constant",
            fill_value="missing"
        )),
        ("codificacao", OneHotEncoder(
            handle_unknown="ignore"
        )),
        ("classificador", KNeighborsClassifier(
            n_neighbors=k
        ))
    ])

    return modelo
