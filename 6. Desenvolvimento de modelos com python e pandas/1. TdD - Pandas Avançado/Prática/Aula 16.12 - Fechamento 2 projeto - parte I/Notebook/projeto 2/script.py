# Bibliotecas Padrão
import os

# Manipulação de Dados
import pandas as pd
import numpy as np

# Visualização de Dados
import seaborn as sns
import matplotlib.pyplot as plt

# Modelagem Estatística
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Análise Exploratória
from ydata_profiling import ProfileReport

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# Carregamento dos Dados
previsao_renda = pd.read_csv("./input/previsao_de_renda.csv")
previsao_renda.info()

# Criar Diretório de Saída
os.makedirs("./output", exist_ok=True)

# Gera Relatório Exploratório
prof = ProfileReport(previsao_renda, explorative=True, minimal=True)
prof.to_file("./output/previsao_de_renda.html")

# Processamento de Dados
previsao_renda.data_ref = pd.to_datetime(previsao_renda.data_ref)
previsao_renda.drop(columns=["Unnamed: 0", "id_cliente"], inplace=True)
previsao_renda.dropna(inplace=True)

# Criar Novas Variáveis
df_transformed = (
    previsao_renda.assign(
        numero_de_dependentes=previsao_renda.qtd_filhos
        + previsao_renda.qt_pessoas_residencia
        - 1
    )
    .assign(
        idade_por_tempo_de_emprego=previsao_renda.idade / previsao_renda.tempo_emprego
    )
    .assign(
        posse_bens=previsao_renda.posse_de_veiculo.astype(str)
        + "_"
        + previsao_renda.posse_de_imovel.astype(str)
    )
)

# Separar Features e Target
X = previsao_renda.drop(columns=["renda", "data_ref"])
y = previsao_renda["renda"]

# Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=100
)

# Transformar Variáveis Categóricas
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train.columns = X_train.columns.str.replace(" ", "_", regex=True)
X_test.columns = X_test.columns.str.replace(" ", "_", regex=True)

# Converter para Float
X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)


def stepwise_selection(
    X, y, initial_list=[], threshold_in=0.05, threshold_out=0.05, verbose=True
):
    """Seleção de Variáveis Usando Stepwise Regression"""
    included = list(initial_list)
    while True:
        changed = False

        # Forward Step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=np.float64)
        for new_column in excluded:
            model = sm.OLS(
                y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))
            ).fit()
            new_pval[new_column] = model.pvalues[new_column]

        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f"Add {best_feature:30} with p-value {best_pval:.6f}")

        # Backward Step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]

        if not pvalues.empty:
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f"Drop {worst_feature:30} with p-value {worst_pval:.6f}")

        if not changed:
            break
    return included


# Selecionar Variáveis Importantes
variaveis = stepwise_selection(X_train, y_train)
print("Variáveis Selecionadas:", variaveis)

# Treinamento do Modelo
regr = DecisionTreeRegressor()
regr.fit(X_train, y_train)
score = regr.score(X_test, y_test)
print(f"Acurácia do Modelo: {score:.4f}")
