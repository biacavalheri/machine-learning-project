
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def criar_bagging():
    """Cria um classificador Bagging com árvores de decisão."""

    arvore_base = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42
    )

    modelo = Pipeline([
        ("ausentes", SimpleImputer(
            strategy="constant",
            fill_value="missing"
        )),
        ("codificacao", OneHotEncoder(
            handle_unknown="ignore"
        )),
        ("classificador", BaggingClassifier(
            estimator=arvore_base,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    return modelo
