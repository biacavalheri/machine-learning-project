
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def criar_arvore_classificacao():
    """Cria uma árvore de decisão usando entropia."""

    modelo = Pipeline([
        ("ausentes", SimpleImputer(
            strategy="constant",
            fill_value="missing"
        )),
        ("codificacao", OneHotEncoder(
            handle_unknown="ignore"
        )),
        ("classificador", DecisionTreeClassifier(
            criterion="entropy",
            random_state=42
        ))
    ])

    return modelo
