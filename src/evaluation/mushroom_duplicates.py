
import pandas as pd


def verificar_duplicatas(X_train, X_test, y_train, y_test):
    """Investiga duplicatas e sobreposição entre treino e teste."""

    print("\n=== ANÁLISE DE DUPLICATAS ===")

    print(f"Duplicatas no treinamento: {X_train.duplicated().sum()}")
    print(f"Duplicatas no teste: {X_test.duplicated().sum()}")

    # Normaliza valores ausentes para permitir a comparação
    treino = X_train.fillna("missing")
    teste = X_test.fillna("missing")

    # Identifica combinações de atributos presentes nos dois conjuntos
    combinacoes_treino = pd.MultiIndex.from_frame(treino)
    combinacoes_teste = pd.MultiIndex.from_frame(teste)

    sobreposicao = combinacoes_teste.isin(combinacoes_treino)

    print(
        f"Observações do teste com atributos idênticos "
        f"aos do treinamento: {sobreposicao.sum()}"
    )

    print(
        f"Percentual do teste com sobreposição: "
        f"{sobreposicao.mean() * 100:.2f}%"
    )

    # Verifica se existem atributos idênticos com classes diferentes
    dados = pd.concat([
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]).rename("classe")
    ], axis=1)

    conflitos = (
        dados.fillna("missing")
        .groupby(list(X_train.columns))["classe"]
        .nunique()
    )

    print(
        "Combinações de atributos com classes conflitantes:",
        (conflitos > 1).sum()
    )
