
# Projeto de Machine Learning

Projeto desenvolvido para a disciplina de Mineração de Dados e Aprendizado de Máquina, do curso de Ciência da Computação da UNESP - Rio Claro.

O objetivo é aplicar e comparar diferentes algoritmos de aprendizado supervisionado em dois problemas: regressão e classificação. O desenvolvimento segue as etapas do processo KDD (Knowledge Discovery in Databases), incluindo exploração dos dados, pré-processamento, modelagem e avaliação.

## 1. Datasets

Foram utilizados dois conjuntos de dados disponíveis no UCI Machine Learning Repository:

**Concrete Compressive Strength**
- 1.030 amostras e 8 atributos numéricos.
- Objetivo: prever a resistência à compressão do concreto (MPa).
- Tipo de problema: regressão.

**Mushroom**
- 8.124 amostras e 22 atributos categóricos.
- Objetivo: classificar cogumelos como comestíveis ou venenosos.
- Tipo de problema: classificação binária.

## 2. Algoritmos utilizados

### Regressão — Concrete
- Regressão Linear
- K-Nearest Neighbors (KNN)
- KNN com PCA
- Árvore de Decisão
- Random Forest
- Bagging
- AdaBoost

### Classificação — Mushroom
- Regressão Logística
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Árvore de Decisão
- Random Forest
- Bagging
- AdaBoost

## 3. Metodologia

Os experimentos foram implementados em Python, utilizando principalmente as bibliotecas Pandas, NumPy, Scikit-learn e Matplotlib.

Os dados foram divididos em 80% para treinamento e 20% para teste. Os modelos foram avaliados por validação cruzada com 5 divisões, além da avaliação final no conjunto de teste.

No Concrete, foi utilizada divisão agrupada para evitar que composições idênticas aparecessem simultaneamente nos conjuntos de treinamento e teste.

No Mushroom, foi utilizada divisão estratificada para preservar a proporção das classes.

As métricas utilizadas foram:
- **Regressão:** MAE, RMSE e R².
- **Classificação:** acurácia, precisão, recall, F1-score e matriz de confusão.

## 4. Principais resultados

### Concrete — Regressão

| Algoritmo | RMSE (teste) | R² (teste) |
|---|---:|---:|
| Regressão Linear | 10,5336 | 0,5162 |
| KNN | 9,2846 | 0,6242 |
| KNN + PCA | 9,1370 | 0,6360 |
| Árvore de Decisão | 7,9138 | 0,7269 |
| Random Forest | 4,7469 | 0,9018 |
| Bagging | 4,7299 | 0,9025 |
| AdaBoost | 7,6488 | 0,7449 |

Os melhores resultados foram obtidos pelos algoritmos **Bagging** e **Random Forest**, ambos com R² superior a 0,90 no conjunto de teste.

### Mushroom — Classificação

| Algoritmo | Acurácia (teste) | Recall (teste) |
|---|---:|---:|
| Regressão Logística | 99,94% | 99,87% |
| Naive Bayes | 94,58% | 89,66% |
| KNN | 100,00% | 100,00% |
| SVM | 100,00% | 100,00% |
| Árvore de Decisão | 100,00% | 100,00% |
| Random Forest | 100,00% | 100,00% |
| Bagging | 100,00% | 100,00% |
| AdaBoost | 99,88% | 99,74% |

Cinco algoritmos alcançaram 100% de acurácia no conjunto de teste: **KNN, SVM, Árvore de Decisão, Random Forest e Bagging**.

O Naive Bayes apresentou o menor desempenho, especialmente em relação à identificação de cogumelos venenosos.

## 5. Execução

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute os experimentos a partir da pasta principal:

```bash
python src/concrete.py
python src/mushroom.py
```

Os resultados, incluindo tabelas CSV e gráficos, são armazenados na pasta `results/`.

## 6. Organização do projeto

- `src/datasets/`: carregamento e exploração dos datasets.
- `src/evaluation/`: avaliação e validação cruzada.
- `src/models/`: implementação dos algoritmos.
- `src/preprocessing/`: preparação e divisão dos dados.
- `src/visualization/`: geração dos gráficos.
- `results/`: resultados dos experimentos.
- `requirements.txt`: dependências do projeto.

## 7. Fontes dos dados

- [Concrete Compressive Strength — UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- [Mushroom — UCI](https://archive.ics.uci.edu/dataset/73/mushroom)
