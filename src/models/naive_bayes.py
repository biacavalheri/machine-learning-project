
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer


def criar_naive_bayes():
    """Cria um pipeline Naive Bayes para dados categóricos."""

    modelo = Pipeline([
        ("ausentes", SimpleImputer(
            strategy="constant",
            fill_value="missing"
        )),
        ("codificacao", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
        ("ajuste_codigos", FunctionTransformer(
            lambda X: X + 1
        )),
        ("classificador", CategoricalNB())
    ])

    return modelo
