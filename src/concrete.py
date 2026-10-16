
from pathlib import Path

import pandas as pd

from datasets.concrete_data import carregar_concrete, explorar_concrete
from preprocessing.concrete_preprocessing import dividir_concrete

from models.linear_regression import treinar_regressao_linear
from models.knn import treinar_knn, avaliar_valores_k
from models.knn_pca import treinar_knn_pca
from models.decision_tree import treinar_arvore_decisao
from models.random_forest import treinar_random_forest
from models.bagging import treinar_bagging
from models.adaboost import treinar_adaboost

from evaluation.regression_metrics import avaliar_regressao
from evaluation.cross_validation import comparar_modelos_cv

from visualization.concrete_plots import gerar_graficos_concrete
from visualization.concrete_evaluation import grafico_reais_previstos


def main():
    # 1. Carregamento e exploração
    X, y = carregar_concrete()
    explorar_concrete(X, y)
    gerar_graficos_concrete(X, y)

    # 2. Divisão dos dados
    X_train, X_test, y_train, y_test, grupos_train = dividir_concrete(
        X, y
    )

    # 3. Seleção do melhor K por validação cruzada
    resultados_k = avaliar_valores_k(
        X_train, y_train, grupos_train
    )

    melhor_k = min(resultados_k, key=resultados_k.get)

    print("\n=== COMPARAÇÃO DOS VALORES DE K ===")
    for k, rmse in resultados_k.items():
        print(f"K={k:2d} | RMSE médio: {rmse:.4f} MPa")

    print(f"\nMelhor K: {melhor_k}")

    # 4. Treinamento dos modelos
    modelos = {
        "Regressão Linear": treinar_regressao_linear(
            X_train, y_train
        ),
        "KNN": treinar_knn(
            X_train, y_train, k=melhor_k
        ),
        "KNN + PCA": treinar_knn_pca(
            X_train, y_train, k=melhor_k
        ),
        "Árvore de Decisão": treinar_arvore_decisao(
            X_train, y_train
        ),
        "Random Forest": treinar_random_forest(
            X_train, y_train
        ),
        "Bagging": treinar_bagging(
            X_train, y_train
        ),
        "AdaBoost": treinar_adaboost(
            X_train, y_train
        )
    }

    # 5. Validação cruzada
    resultados_cv = comparar_modelos_cv(
        modelos,
        X_train,
        y_train,
        grupos_train
    )

    print("\n=== VALIDAÇÃO CRUZADA ===")
    print(resultados_cv.round(4).to_string(index=False))

    # 6. Avaliação no conjunto de teste
    resultados_teste = []

    for nome, modelo in modelos.items():
        metricas = avaliar_regressao(
            modelo,
            X_test,
            y_test
        )

        resultados_teste.append({
            "Modelo": nome,
            "Teste MAE": metricas["MAE"],
            "Teste RMSE": metricas["RMSE"],
            "Teste R²": metricas["R²"]
        })

    tabela_teste = pd.DataFrame(resultados_teste)

    print("\n=== RESULTADOS FINAIS — TESTE ===")
    print(tabela_teste.round(4).to_string(index=False))

    # 7. Consolidação dos resultados
    resultados_cv = resultados_cv.rename(columns={
        "MAE": "CV MAE",
        "RMSE": "CV RMSE",
        "R²": "CV R²",
        "Desvio RMSE": "CV Desvio RMSE"
    })

    comparacao = pd.merge(
        resultados_cv,
        tabela_teste,
        on="Modelo"
    )

    comparacao = comparacao.sort_values("CV RMSE")

    Path("results").mkdir(exist_ok=True)

    comparacao.to_csv(
        "results/concrete_comparacao_modelos.csv",
        index=False
    )

    print("\n=== COMPARAÇÃO CONSOLIDADA ===")
    print(comparacao.round(4).to_string(index=False))

    # 8. Interpretação dos modelos
    modelo_arvore = modelos["Árvore de Decisão"]

    importancias = sorted(
        zip(X_train.columns, modelo_arvore.feature_importances_),
        key=lambda item: item[1],
        reverse=True
    )

    print("\n=== IMPORTÂNCIA DOS ATRIBUTOS — ÁRVORE ===")
    for atributo, importancia in importancias:
        print(f"{atributo}: {importancia:.4f}")

    modelo_pca = modelos["KNN + PCA"]
    componentes = modelo_pca.named_steps["pca"].n_components_

    print(f"\nComponentes principais do PCA: {componentes}")

    # 9. Visualização dos resultados
    grafico_reais_previstos(
        {
            "Regressão Linear": modelos["Regressão Linear"],
            "Bagging": modelos["Bagging"]
        },
        X_test,
        y_test
    )

    print("\nResultados salvos em results/")


if __name__ == "__main__":
    main()
