
from pathlib import Path

import pandas as pd

from datasets.mushroom_data import carregar_mushroom, explorar_mushroom
from preprocessing.mushroom_preprocessing import dividir_mushroom

from models.logistic_regression import criar_regressao_logistica
from models.naive_bayes import criar_naive_bayes
from models.knn_classifier import criar_knn_classificador
from models.svm_classifier import criar_svm
from models.decision_tree_classifier import criar_arvore_classificacao
from models.random_forest_classifier import criar_random_forest
from models.bagging_classifier import criar_bagging
from models.adaboost_classifier import criar_adaboost

from evaluation.classification_metrics import avaliar_classificacao_cv
from evaluation.mushroom_duplicates import verificar_duplicatas
from evaluation.mushroom_final import avaliar_modelos_finais

from visualization.mushroom_tree import visualizar_arvore
from visualization.mushroom_evaluation import plotar_matrizes_confusao


def main():
    # 1. Carregamento e exploração
    X, y = carregar_mushroom()
    explorar_mushroom(X, y)

    # 2. Divisão dos dados
    X_train, X_test, y_train, y_test = dividir_mushroom(X, y)

    # 3. Verificação de duplicatas
    verificar_duplicatas(X_train, X_test, y_train, y_test)

    # 4. Definição dos classificadores
    modelos = {
        "Regressão Logística": criar_regressao_logistica(X_train),
        "Naive Bayes": criar_naive_bayes(),
        "KNN": criar_knn_classificador(k=5),
        "SVM": criar_svm(),
        "Árvore de Decisão": criar_arvore_classificacao(),
        "Random Forest": criar_random_forest(),
        "Bagging": criar_bagging(),
        "AdaBoost": criar_adaboost()
    }

    # 5. Validação cruzada
    resultados_cv = []

    for nome, modelo in modelos.items():
        resultado = avaliar_classificacao_cv(
            modelo,
            X_train,
            y_train
        )

        medias = resultado.set_index("Métrica")["Média"]
        desvios = resultado.set_index("Métrica")["Desvio-padrão"]

        resultados_cv.append({
            "Modelo": nome,
            "CV Acurácia": medias["Acurácia"],
            "CV Precisão": medias["Precisão"],
            "CV Recall": medias["Recall"],
            "CV F1": medias["F1"],
            "CV Desvio Acurácia": desvios["Acurácia"],
            "CV Desvio Precisão": desvios["Precisão"],
            "CV Desvio Recall": desvios["Recall"],
            "CV Desvio F1": desvios["F1"]
        })

    tabela_cv = pd.DataFrame(resultados_cv)

    print("\n=== VALIDAÇÃO CRUZADA ===")
    print(tabela_cv.round(4).to_string(index=False))

    # 6. Avaliação final no teste
    resultados_teste, matrizes = avaliar_modelos_finais(
        modelos,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n=== RESULTADOS FINAIS — TESTE ===")
    print(resultados_teste.round(4).to_string(index=False))

    # 7. Consolidar resultados
    resultados_teste = resultados_teste.rename(columns={
        "Acurácia": "Teste Acurácia",
        "Precisão": "Teste Precisão",
        "Recall": "Teste Recall",
        "F1": "Teste F1"
    })

    comparacao = pd.merge(
        tabela_cv,
        resultados_teste,
        on="Modelo"
    )

    comparacao = comparacao.sort_values(
        ["CV Recall", "CV F1"],
        ascending=False
    )

    Path("results").mkdir(exist_ok=True)

    comparacao.to_csv(
        "results/mushroom_comparacao_modelos.csv",
        index=False
    )

    print("\n=== COMPARAÇÃO CONSOLIDADA ===")
    print(comparacao.round(4).to_string(index=False))

    # 8. Visualizações
    visualizar_arvore(
        modelos["Árvore de Decisão"],
        X_train,
        y_train
    )

    plotar_matrizes_confusao(matrizes)

    print("\nResultados salvos em results/")


if __name__ == "__main__":
    main()
