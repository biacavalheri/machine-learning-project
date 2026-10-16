
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree


def visualizar_arvore(modelo, X_train, y_train):
    """Treina a árvore e gera gráficos para interpretação."""

    Path("results").mkdir(exist_ok=True)

    modelo.fit(X_train, y_train)

    codificador = modelo.named_steps["codificacao"]
    arvore = modelo.named_steps["classificador"]

    nomes_atributos = codificador.get_feature_names_out(
        X_train.columns
    )

    # Visualização dos primeiros níveis
    plt.figure(figsize=(18, 9))

    plot_tree(
        arvore,
        feature_names=nomes_atributos,
        class_names=["Comestível", "Venenoso"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=9
    )

    plt.title("Árvore de Decisão — Primeiros Níveis")
    plt.tight_layout()
    plt.savefig(
        "results/mushroom_arvore.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Importância dos atributos codificados
    importancias = pd.Series(
        arvore.feature_importances_,
        index=nomes_atributos
    )

    # Agrupa as categorias por atributo original
    importancias_agrupadas = pd.Series({
        coluna: importancias[
            [nome for nome in nomes_atributos
             if nome.startswith(f"{coluna}_")]
        ].sum()
        for coluna in X_train.columns
    }).sort_values(ascending=False)

    print("\n=== IMPORTÂNCIA DOS ATRIBUTOS ===")
    print(importancias_agrupadas.head(10).round(4))

    plt.figure(figsize=(10, 6))

    importancias_agrupadas.head(10).sort_values().plot(
        kind="barh"
    )

    plt.title("Dez Atributos Mais Importantes — Mushroom")
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.savefig(
        "results/mushroom_importancias.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
