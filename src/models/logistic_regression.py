
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def criar_regressao_logistica(X_train):
    """Cria o pipeline de regressão logística."""

    atributos = X_train.columns.tolist()

    preprocessamento = ColumnTransformer([
        (
            "categoricos",
            Pipeline([
                ("ausentes", SimpleImputer(
                    strategy="constant",
                    fill_value="missing"
                )),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]),
            atributos
        )
    ])

    modelo = Pipeline([
        ("preprocessamento", preprocessamento),
        ("classificador", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    return modelo
