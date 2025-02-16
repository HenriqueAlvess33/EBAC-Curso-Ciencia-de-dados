from datetime import datetime

##### Manipulação de dados #####
import numpy as np  # Manipulação de arrays e operações matemáticas
import pandas as pd  # Manipulação de dados em DataFrames

##### Visualização de dados #####
import matplotlib.pyplot as plt  # Para criação de gráficos
import seaborn as sns  # Para visualizações mais avançadas com gráficos estatísticos

##### Modelagem estatística #####
import statsmodels.api as sm  # Para modelagem estatística (como regressões)
import statsmodels.formula.api as smf  # Para fórmulas em regressões e modelos estatísticos

##### Análise exploratória #####
from ydata_profiling import (
    ProfileReport,
)  # Para gerar relatórios exploratórios automáticos

##### Machine Learning #####
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)  # Para avaliação de modelos de regressão
from sklearn.model_selection import (
    train_test_split,
)  # Para dividir dados em treino e teste
from sklearn.tree import (
    DecisionTreeRegressor,
)  # Para modelo de regressão de árvore de decisão

##### Interface gráfica #####
import streamlit as st  # Biblioteca para criar interfaces interativas em Python
from streamlit_pandas_profiling import (
    st_profile_report,
)

from st_aggrid import AgGrid, GridOptionsBuilder

# Configuração da página no Streamlit (layout largo)
st.set_page_config(layout="wide")

# Adicionando um título estilizado no topo da página
st.markdown(
    """
    <style>
        .container1 {
            text-align: center;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            margin-top: 50px;
        }
        h1 {
            color: #333;
            font-size: 22px;
        }
    </style>
    <div class="container1">
        <h1>Construção de Modelo para Previsão de Renda</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Descrição da primeira etapa do processo CRISP-DM (Entendimento do negócio)
st.markdown(
    """
    <div class="container2">
        <h2>1ª Etapa CRISP-DM - Entendimento do negócio</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Objetivo principal: Compreender como a variável de renda deve ser utilizada na concessão de crédito aos clientes e "
    "de que forma a relação desta variável dependente, em conjunto com as explicativas, pode contribuir para definir "
    "o perfil 'default' do cliente."
    "\n\nÁrea de atuação: Nossa empresa atua no ramo financeiro, e é imprescindível contar com bons modelos de "
    "análise de crédito, já que este é o alicerce para balizar todas as operações. O modelo será utilizado por "
    "equipes de crédito, risco e análise financeira, com o objetivo de minimizar os riscos de inadimplência, "
    "aprimorar a experiência do usuário e maximizar a base de clientes elegíveis."
)

# Descrição da segunda etapa do processo CRISP-DM (Entendimento dos dados)
st.markdown(
    """
    <div class="container3">
        <h2>2ª Etapa CRISP-DM - Entendimento dos dados</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Criação de um DataFrame com informações sobre as variáveis do dataset
data = pd.DataFrame(
    {
        "Variável": [
            "data_ref",
            "id_cliente",
            "sexo",
            "posse_de_veiculo",
            "posse_de_imovel",
            "qtd_filhos",
            "tipo_renda",
            "educacao",
            "estado_civil",
            "tipo_residencia",
            "idade",
            "tempo_emprego",
            "qt_pessoas_residencia",
            "renda",
        ],
        "Descrição": [
            "Data de registro das informações do cliente",
            "Código de identificação para cada cliente",
            "Sexo do cliente",
            "Indicação da posse de veículo do cliente",
            "Indicação da posse de imóvel do cliente",
            "Quantidade de filhos que o cliente possui",
            "Tipo de renda do cliente",
            "Nível de educação acadêmica do cliente",
            "Status conjugal do cliente",
            "Modelo de moradia do cliente",
            "Idade do cliente",
            "Tempo de trabalho contínuo do cliente",
            "Quantidade de pessoas que dividem moradia com o cliente",
            "Número absoluto de ganho mensal do cliente",
        ],
        "Tipo": [
            "datetime64[ns]",
            "int64",
            "object",
            "bool",
            "bool",
            "int64",
            "object",
            "object",
            "object",
            "object",
            "int64",
            "object",
            "float64",
            "int64",
        ],
    }
)

# Exibir o DataFrame contendo a descrição das variáveis
st.dataframe(data, use_container_width=True)

# Carregar os dados do arquivo CSV (arquivo que contém os dados para análise)
previsao_renda = pd.read_csv("./input/previsao_de_renda.csv")

# Gerar um relatório exploratório com o ydata_profiling (opcional)
if st.checkbox("Profile Report"):
    relatorio_da_base = ProfileReport(previsao_renda, explorative=True, minimal=True)
    st_profile_report(relatorio_da_base)

# Criando uma lista com os nomes das colunas do dataset (ignorando as 3 primeiras)
lista_de_colunas = previsao_renda.columns.to_list()
lista_de_colunas = lista_de_colunas[3:]

previsao_renda = previsao_renda.dropna(subset=["tempo_emprego"])

previsao_renda = (
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


# Criando uma lista de DataFrames contendo as contagens de valores únicos por coluna
resultados = [
    previsao_renda[coluna]
    .value_counts()
    .reset_index()
    .rename(columns={"index": coluna})
    for coluna in lista_de_colunas
]

# Concatenando os resultados em um único DataFrame e adicionando o nome da variável correspondente
resultado_final = pd.concat(
    [df.assign(variavel=coluna) for df, coluna in zip(resultados, lista_de_colunas)],
    ignore_index=True,
)

# Criando um dicionário contendo a distribuição de valores para cada variável do dataset
dicionario_de_distribuicao = {
    coluna: resultado_final.loc[resultado_final["variavel"] == coluna].dropna(axis=1)
    for coluna in lista_de_colunas
}

# Exibindo a distribuição da variável 'sexo' no formato percentual
(
    dicionario_de_distribuicao["sexo"]
    .drop("variavel", axis=1)
    .assign(percentual=lambda x: (x["count"] / x["count"].sum()) * 100)
    .set_index("count")
)

# Adicionando uma opção de checagem da distribuição das variáveis no sidebar do Streamlit
if st.sidebar.checkbox("Checagem da distribuição de variáveis:"):
    chave_selecionada = st.sidebar.selectbox("Selecione a variável: ", lista_de_colunas)

    # Exibir a tabela e o gráfico de distribuição da variável selecionada
    if chave_selecionada in dicionario_de_distribuicao:
        df_transformado = (
            dicionario_de_distribuicao[chave_selecionada]
            .drop("variavel", axis=1)
            .assign(percentual=lambda x: (x["count"] / x["count"].sum()) * 100)
            .set_index("count")
        )

        # Exibir a tabela formatada no Streamlit
        st.sidebar.dataframe(df_transformado)

        # Exibindo o gráfico de distribuição da variável
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        plt.tight_layout(pad=6.0)
        plt.subplots_adjust()
        axes.flatten()
        if (previsao_renda[chave_selecionada].dtypes == "object") & (
            previsao_renda[chave_selecionada].dtypes == "bool"
        ):
            sns.countplot(data=previsao_renda, x=chave_selecionada, ax=axes[0])
            sns.lineplot(
                data=previsao_renda, x=chave_selecionada, y="renda", ax=axes[1]
            )
            axes[0].set_title(
                f"Distribuição da variável {chave_selecionada} pela base de dados",
                fontsize=14,
            )
            axes[1].set_title(
                f"Análise bivariada da variável {chave_selecionada}", fontsize=14
            )

            st.pyplot(fig)

        else:
            sns.histplot(data=previsao_renda, x=chave_selecionada, ax=axes[0])
            sns.lineplot(
                data=previsao_renda, x=chave_selecionada, y="renda", ax=axes[1]
            )
            axes[0].set_title(
                f"Distribuição da variável {chave_selecionada} pela base de dados",
                fontsize=14,
            )
            axes[1].set_title(
                f"Análise bivariada da variável {chave_selecionada}", fontsize=14
            )
            st.pyplot(fig)

st.markdown(
    """
    <div class="container2">
        <h2>3ª Etapa CRISP-DM - Preparação dos dados </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Já foram removidas as linhas com dados faltantes. A ausência de dados somente foi constatada na variável tipo_emprego, antes da remoção a base de dados possuia 15.000 linhas e passou a ter 12.427 após o tratamento"
)
st.write(
    "Também foram criadas duas novas variáveis. Sendo elas numero_de_dependentes, relacionada a qt_pessoas_residencia, e idade_por_tempo_emprego, apresentando a relação entre a variável idade e tempo_emprego"
)


st.markdown(
    """
    <div class="container2">
        <h2>4ª Etapa CRISP-DM - Modelagem </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

x = previsao_renda.drop(columns=["renda", "data_ref"])
y = previsao_renda.renda

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=100
)

X_train = pd.get_dummies(x_train)
X_test = pd.get_dummies(x_test)


X_train.columns = X_train.columns.str.replace(" ", "_", regex=True)
X_test.columns = X_test.columns.str.replace(" ", "_", regex=True)
X_test.reindex(columns=X_train.columns, fill_value=0)

y = y_train

X_train = X_train.astype(float)
X_test = X_test.astype(float)

y = y.astype(float)


string_variaveis = "tempo_emprego + sexo_F + sexo_M + idade + tipo_renda_Empresário + educacao_Superior_completo + estado_civil_Casado"
string_variaveis_power = "np.power(tempo_emprego,2) + sexo_F + sexo_M + np.power(idade,2) + tipo_renda_Empresário + educacao_Superior_completo + estado_civil_Casado"
string_variaveis_log = "np.log(tempo_emprego) + sexo_F + sexo_M + np.log(idade) + tipo_renda_Empresário + educacao_Superior_completo + estado_civil_Casado"
colunas_ridge = ["Ridge Normal", "Ridge Power", "Ridge Logaritmo"]

lista_de_strings = [string_variaveis, string_variaveis_power, string_variaveis_log]
lista_de_strings_para_nomear = [
    "string_variaveis",
    "string_variaveis_power",
    "string_variaveis_log",
]
registro_de_residuos = {}
dicionario_indicadores = {}
valores_preditos = {}

md_treino = smf.ols(f"renda ~ {string_variaveis}", data=X_train.join(y_train))
reg = md_treino.fit_regularized(method="elastic_net", refit=True, L1_wt=1, alpha=0.001)
reg.summary()


r2_lasso = reg.rsquared
r2_lasso_ajustado = reg.rsquared_adj
aic_lasso = reg.aic

for i, string in enumerate(lista_de_strings):
    md_treino = smf.ols(f"renda ~ {string}", data=X_train.join(y_train))
    reg = md_treino.fit_regularized(
        method="elastic_net", refit=True, L1_wt=0, alpha=0.001
    )

    y_pred_test = reg.predict(X_test)
    valores_preditos[lista_de_strings_para_nomear[i]] = y_pred_test

    residuos_test = y_test - y_pred_test

    registro_de_residuos[lista_de_strings_para_nomear[i]] = residuos_test

    r_quadrado = r2_score(y_test, y_pred_test)
    r_quadrado_ajustado = 1 - ((1 - r_quadrado) * (len(y_test) - 1)) / (
        len(y_test) - len(X_test.columns) - 1
    )

    rss = np.power(y_test - y_pred_test, 2).sum()
    log_vero_test = (
        -len(y_train) / 2 * (np.log(2 * np.pi) + np.log(rss / len(y_test)) + 1)
    )

    aic = 2 * len(reg.params) - 2 * log_vero_test

    dicionario_indicadores[colunas_ridge[i]] = {
        "R-Quadrado": round(r_quadrado, 3),
        "R-Quadrado Ajustado": round(r_quadrado_ajustado, 3),
        "AIC": aic,
        "Coef": reg.params,
    }


# @st.cache_data
# def treinamento_de_arvore_de_decisao(X_train, y_train, X_test, y_test):
#     regr = DecisionTreeRegressor(random_state=100)
#     regr.fit(X_train, y_train)

#     r2 = regr.score(X_test, y_test)

#     n = X_test.shape[0]
#     p = X_test.shape[1]

#     r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

#     y_pred = regr.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)

#     aic = n * np.log(mse) + 2 * p if mse > 0 else float("inf")

#     return r2, r2_adj, aic, regr


# r2, r2_adj, aic, model = treinamento_de_arvore_de_decisao(
#     X_train, y_train, X_test, y_test
# )

regr = DecisionTreeRegressor(ccp_alpha=33260.49826610106, random_state=2360873)
regr.fit(X_train, y_train)

r2 = regr.score(X_test, y_test)

n = X_test.shape[0]
p = X_test.shape[1]

r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

aic = n * np.log(mse) + 2 * p if mse > 0 else float("inf")

info_modelos = pd.DataFrame(
    {
        "Regressao Lasso Normal": {
            "R-Quadrado": r2_lasso,
            "R-Quadrado Ajustado": r2_lasso_ajustado,
            "AIC": aic_lasso,
        },
        "Regressao Ridge Normal": {
            "R-Quadrado": dicionario_indicadores["Ridge Normal"]["R-Quadrado"],
            "R-Quadrado Ajustado": dicionario_indicadores["Ridge Normal"][
                "R-Quadrado Ajustado"
            ],
            "AIC": dicionario_indicadores["Ridge Normal"]["AIC"],
        },
        "Regressao Ridge Logaritmica": {
            "R-Quadrado": dicionario_indicadores["Ridge Logaritmo"]["R-Quadrado"],
            "R-Quadrado Ajustado": dicionario_indicadores["Ridge Logaritmo"][
                "R-Quadrado Ajustado"
            ],
            "AIC": dicionario_indicadores["Ridge Logaritmo"]["AIC"],
        },
        "Regressao Ridge na Potencia": {
            "R-Quadrado": dicionario_indicadores["Ridge Power"]["R-Quadrado"],
            "R-Quadrado Ajustado": dicionario_indicadores["Ridge Power"][
                "R-Quadrado Ajustado"
            ],
            "AIC": dicionario_indicadores["Ridge Power"]["AIC"],
        },
        "Árvore de Regressão": {
            "R-Quadrado": r2,
            "R-Quadrado Ajustado": r2_adj,
            "AIC": aic,
        },
    }
).T


chave_de_selecao_de_modelo = st.selectbox(
    "Escolha o modelo para obter os dados de performance:", info_modelos.index
)


st.dataframe(info_modelos.loc[[chave_de_selecao_de_modelo]])

st.markdown("## Modelo árvore de regressão")

# Entradas do usuário
idade = st.number_input("Sua idade", min_value=18, max_value=100, value=30)

tempo_de_emprego = st.number_input(
    "Tempo de emprego (anos)", min_value=0, max_value=100, value=5
)
tipo_de_renda = st.selectbox(
    "Modelo de renda",
    ["Empresário", "Assalariado", "Servidor público", "Pensionista", "Bolsista"],
)

posse_de_veiculo = st.selectbox("Possui veiculo ?", ["Sim", "Não"])

posse_de_imovel = st.selectbox("Possui imóvel ?", ["Sim", "Não"])

qtd_filhos = st.slider("Quantos filhos você tem ?", 0, 40, 0)

qt_pessoas_residencia = st.slider("Quantas pessoas moram com você ?", 0, 15, 0)

estado_civil = st.selectbox(
    "Estado Cívil",
    ["Solteiro", "União", "Viúvo", "Separado"],
)

tipo_residencia = st.selectbox(
    "Tipo de moradia",
    ["Aluguel", "Casa", "Com os pais", "Comunitário", "Estúdio", "Governamental"],
)

sexo = st.selectbox(
    "Qual o seu sexo ?",
    ["Masculino", "Feminino"],
)

if st.button("Prever Renda"):
    # Criar um DataFrame com as entradas do usuário
    user_input = pd.DataFrame(
        {
            "idade": [float(idade)],
            "tempo_emprego": [float(tempo_de_emprego)],
            "tipo_renda": [tipo_de_renda],
            "posse_de_veiculo": [posse_de_veiculo],
            "qtd_filhos": [qtd_filhos],
            "tipo_residencia": [tipo_residencia],
            "sexo": [sexo],
            "qt_pessoas_residencia": [qt_pessoas_residencia],
        }
    )

    # Aplicar o mesmo processamento das variáveis dummies
    user_input = pd.get_dummies(user_input)

    # Reindexar para garantir as mesmas colunas do treinamento
    user_input = user_input.reindex(columns=X_train.columns, fill_value=0)

    # Fazer a previsão usando o modelo treinado
    renda_prevista = regr.predict(user_input)

    st.write(
        f"A previsão da renda para as entradas fornecidas é: R${renda_prevista[0]:.2f}"
    )
