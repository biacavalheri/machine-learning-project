
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def criar_random_forest():
    """Cria um Random Forest para classificação."""

    modelo = Pipeline([
        ("ausentes", SimpleImputer(
            strategy="constant",
            fill_value="missing"
        )),
        ("codificacao", OneHotEncoder(
            handle_unknown="ignore"
        )),
        ("classificador", RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            random_state=42,
            n_jobs=-1
        ))
    ])

    return modelo
