
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def criar_adaboost():
    """Cria um classificador AdaBoost com árvores de decisão."""

    arvore_base = DecisionTreeClassifier(
        max_depth=1,
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
        ("classificador", AdaBoostClassifier(
            estimator=arvore_base,
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        ))
    ])

    return modelo
