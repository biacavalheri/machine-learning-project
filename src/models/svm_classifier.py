
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


def criar_svm():
    """Cria um pipeline SVM para classificação."""

    modelo = Pipeline([
        ("ausentes", SimpleImputer(
            strategy="constant",
            fill_value="missing"
        )),
        ("codificacao", OneHotEncoder(
            handle_unknown="ignore"
        )),
        ("classificador", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        ))
    ])

    return modelo
