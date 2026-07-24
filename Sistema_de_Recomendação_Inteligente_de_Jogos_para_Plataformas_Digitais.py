#importações
import pandas as pd # manipulação e análise de dados tabulares
import numpy as np # operações matemáticas e manipulação de arrays
import matplotlib.pyplot as plt  # criação de gráficos e visualizações
import seaborn as sns # visualização estatística baseada no matplotlib
from surprise import Dataset, Reader, SVD, KNNBaseline # estruturas de dados e algoritmos de recomendação
from surprise.model_selection import cross_validate, train_test_split  # validação cruzada e divisão de dados
from surprise import accuracy # métricas de avaliação de erro preditivo
from collections import defaultdict # estrutura auxiliar para agrupamento de recomendações

# leitura do arquivo csv atribuindo nomes descritivos às colunas manualmente
# o dataset não possui cabeçalho nativo, por isso header= None é necessário
df = pd.read_csv('steam-200k.csv', header=None, names=['user_id', 'game_title', 'behavior', 'hours_played', 'resto'])

# verificação dimensional do dataframe número de linhas e colunas carregadas
print(f"Dimensões do dataset bruto: {df.shape}")

# inspeção visual das primeiras linhas para confirmar integridade da leitura
print(df.head(10))

# registros de compra têm hours_played=1.0 por padrão e não representam engajamento real
df_play = df[df['behavior'] == 'play'].copy()

# remoção de registros com tempo de jogo nulo ou zerado, que indicam contas inativas
df_play = df_play[df_play['hours_played'] > 0]

# filtragem de esparsidade mantém apenas usuários com histórico mínimo de 5 jogos
# usuários com menos interações produzem vetores muito esparsos, prejudicando o SVD
contagem_usuario = df_play.groupby('user_id')['game_title'].count()
usuarios_validos = contagem_usuario[contagem_usuario >= 5].index
df_play = df_play[df_play['user_id'].isin(usuarios_validos)]

# filtragem de esparsidade mantém apenas jogos com pelo menos 10 interações registradas
# títulos com poucas avaliações não oferecem sinal suficiente para o modelo aprender
contagem_jogo = df_play.groupby('game_title')['user_id'].count()
jogos_validos = contagem_jogo[contagem_jogo >= 10].index
df_filtrado = df_play[df_play['game_title'].isin(jogos_validos)].copy()

print(f"Dimensões após filtragem: {df_filtrado.shape}")
print(f"Usuários únicos: {df_filtrado['user_id'].nunique()}")
print(f"Jogos únicos: {df_filtrado['game_title'].nunique()}")

# a abordagem de limiares fixos (ex: < 1h = nota 1) ignora diferenças entre gêneros de jogos
# um RPG com 20h pode representar engajamento médio, enquanto 20h em um jogo casual é altíssimo

# solução normalizar as horas por percentil dentro de cada título individualmente
# dessa forma a nota reflete o engajamento relativo do usuário em comparação a outros jogadores do mesmo jogo

# separo o índice dos registros em 80% treino e 20% teste antes de qualquer transformação
indices_treino = df_filtrado.sample(frac=0.8, random_state=42).index
indices_teste  = df_filtrado.index.difference(indices_treino)

df_treino_raw = df_filtrado.loc[indices_treino].copy()
df_teste_raw = df_filtrado.loc[indices_teste].copy()


# converte as horas jogadas em uma escala de engajamento de 1 a 5
# utilizando os percentis da distribuição de horas dentro de cada jogo.
# isso garante que a nota seja relativa ao comportamento dos demais jogadores do mesmo título.


def horas_para_engajamento_percentil(grupo):

    horas = grupo['hours_played']
    p20 = horas.quantile(0.20)
    p40 = horas.quantile(0.40)
    p60 = horas.quantile(0.60)
    p80 = horas.quantile(0.80)

    def classificar(h):
        if h <= p20:
            return 1
        elif h <= p40:
            return 2
        elif h <= p60:
            return 3
        elif h <= p80:
            return 4
        else:
            return 5

    grupo = grupo.copy()
    grupo['engajamento'] = horas.apply(classificar)
    return grupo


# aplica ao subset (teste) os limiares de percentil calculados exclusivamente no treino.
# isso evita que o conjunto de teste influencie os limiares de classificação,
# garantindo que a transformação seja idêntica ao que aconteceria

def aplicar_limiares_do_treino(df_subset, limiares):

    def classificar_com_limiar(row):
        jogo = row['game_title']
        h    = row['hours_played']
        # se o jogo não existia no treino, atribuo nota média como fallback
        if jogo not in limiares:
            return 3
        p20, p40, p60, p80 = limiares[jogo]
        if h <= p20: return 1
        elif h <= p40: return 2
        elif h <= p60: return 3
        elif h <= p80: return 4
        else: return 5

    df_out = df_subset.copy()
    df_out['engajamento'] = df_out.apply(classificar_com_limiar, axis=1)
    return df_out

# calculo os percentis usando apenas os dados de treino para evitar data leakage
# o teste vai receber esses mesmos limiares, sem recalcular nada com seus próprios dados
limiares_treino = {}
for jogo, grupo in df_treino_raw.groupby('game_title'):
    horas = grupo['hours_played']
    limiares_treino[jogo] = (
        horas.quantile(0.20),
        horas.quantile(0.40),
        horas.quantile(0.60),
        horas.quantile(0.80),
    )

# aplico os percentis do treino no próprio treino
df_treino = df_treino_raw.groupby('game_title', group_keys=False).apply(horas_para_engajamento_percentil)
df_treino = df_treino.reset_index(drop=True)

# aplico os limiares do treino no teste sem recalcular, sem contaminar
df_teste = aplicar_limiares_do_treino(df_teste_raw, limiares_treino)

# junto treino e teste para manter os gráficos exploratórios com a base completa
df_filtrado = pd.concat([df_treino, df_teste], ignore_index=True)

print("Distribuição da escala de engajamento após normalização por percentil:\n")
print(df_filtrado['engajamento'].value_counts().sort_index())

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análise Exploratória do Dataset Steam', fontsize=16, fontweight='bold', y=1.01)

# gráfico 1: distribuição da escala de engajamento
# mostra se a conversão por percentil gerou uma distribuição equilibrada entre as notas
contagem_eng = df_filtrado['engajamento'].value_counts().sort_index()
axes[0, 0].bar(contagem_eng.index, contagem_eng.values, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Distribuição da Escala de Engajamento')
axes[0, 0].set_xlabel('Nota de Engajamento (1 a 5)')
axes[0, 0].set_ylabel('Quantidade de Registros')
for i, v in enumerate(contagem_eng.values): axes[0, 0].text(i + 1, v + 50, str(v), ha='center', fontsize=9)

# gráfico 2: top 15 jogos mais jogados na base filtrada
# identifica quais títulos concentram mais interações e potencial influência no modelo
top_jogos = df_filtrado.groupby('game_title')['game_title'].count().sort_values(ascending=False).head(15)
top_jogos.index = top_jogos.index.astype(str)
axes[0, 1].barh(top_jogos.index[::-1], top_jogos.values[::-1], color='coral', edgecolor='black')
axes[0, 1].set_title('Top 15 Jogos com Mais Interações')
axes[0, 1].set_xlabel('Número de Jogadores')

# gráfico 3: distribuição de jogos por usuário
# evidencia a natureza esparsa dos dados a maioria dos usuários jogou poucos títulos
jogos_por_usuario = df_filtrado.groupby('user_id')['game_title'].count()
axes[1, 0].hist(jogos_por_usuario, bins=50, color='mediumseagreen', edgecolor='black', log=True)
axes[1, 0].set_title('Distribuição de Jogos por Usuário (escala log)')
axes[1, 0].set_xlabel('Quantidade de Jogos')
axes[1, 0].set_ylabel('Número de Usuários (log)')

# gráfico 4: estimativa visual da densidade da matriz usuário jogo
# calcula o percentual de preenchimento da matriz para quantificar a esparsidade
n_usuarios = df_filtrado['user_id'].nunique()
n_jogos = df_filtrado['game_title'].nunique()
n_interacoes = len(df_filtrado)
densidade = n_interacoes / (n_usuarios * n_jogos) * 100
esparsidade = 100 - densidade

axes[1, 1].pie(
    [densidade, esparsidade],
    labels=[f'Preenchido\n{densidade:.2f}%', f'Vazio\n{esparsidade:.2f}%'],
    colors=['steelblue', 'lightgrey'],
    startangle=90,
    wedgeprops={'edgecolor': 'black'}
)
axes[1, 1].set_title('Esparsidade da Matriz Usuário-Jogo')

plt.tight_layout()
plt.show()

print(f"\nEstatísticas da matriz usuário-jogo:")
print(f"Usuários:{n_usuarios}")
print(f"Jogos:{n_jogos}")
print(f"Interações:{n_interacoes}")
print(f"Densidade:{densidade:.4f}%")
print(f"Esparsidade:{esparsidade:.4f}%")

# define a escala de avaliação para que a biblioteca Surprise interprete as notas de engajamento corretamente
reader = Reader(rating_scale=(1, 5))

# carrego os dados de treino e teste separados no formato exigido pela biblioteca Surprise
# uso os DataFrames já separados antes para manter a consistência com o fix de data leakage
data_treino = Dataset.load_from_df(df_treino[['user_id', 'game_title', 'engajamento']], reader)
data_full = Dataset.load_from_df(df_filtrado[['user_id', 'game_title', 'engajamento']], reader)

# construo o trainset a partir do df_treino e o testset a partir do df_teste
# dessa forma a divisão respeita os limiares calculados separadamente em cada parte
trainset = data_treino.build_full_trainset()
testset  = list(zip(df_teste['user_id'], df_teste['game_title'], df_teste['engajamento']))

# a validação cruzada com k=5 folds fornece uma estimativa mais robusta do desempenho do modelo
# ao contrário de um único split, o CV avalia o modelo em 5 partições distintas dos dados
# reduzindo o risco de overfitting ao conjunto de teste específico escolhido
svd_cv = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)

cv_results = cross_validate(svd_cv, data_full, measures=['RMSE', 'MAE'], cv=5, verbose=False)

print(" Validação Cruzada SVD")
print(f"RMSE médio: {cv_results['test_rmse'].mean():.4f}  ({cv_results['test_rmse'].std():.4f})")
print(f"MAE  médio: {cv_results['test_mae'].mean():.4f}  ({cv_results['test_mae'].std():.4f})")

# o KNNBaseline é utilizado como modelo de referência para comparação com o SVD
# ele combina efeitos de baseline com similaridade entre usuários via correlação de Pearson
knn = KNNBaseline(k=20, sim_options={'name': 'pearson_baseline', 'user_based': True})

knn.fit(trainset)
preds_knn = knn.test(testset)

rmse_knn = accuracy.rmse(preds_knn, verbose=False)
mae_knn  = accuracy.mae(preds_knn,  verbose=False)

print(" KNN Baseline Avaliação no Conjunto de Teste")
print(f"RMSE: {rmse_knn:.4f}")
print(f"MAE: {mae_knn:.4f}")

# o SVD decompõe a matriz usuário jogo em fatores latentes
# esses fatores capturam padrões ocultos de preferência que não estão explícitos nos dados
svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)

svd.fit(trainset)
preds_svd = svd.test(testset)

rmse_svd = accuracy.rmse(preds_svd, verbose=False)
mae_svd  = accuracy.mae(preds_svd,  verbose=False)

print("SVD Avaliação no Conjunto de Teste")
print(f"RMSE: {rmse_svd:.4f}")
print(f"MAE:  {mae_svd:.4f}")

# consolida os resultados dos dois modelos em uma tabela para facilitar a análise comparativa
# permite identificar objetivamente qual algoritmo apresentou menor erro de predição
resultados = pd.DataFrame({
    'Modelo': ['KNN Baseline', 'SVD'],
    'RMSE':[round(rmse_knn, 4), round(rmse_svd, 4)],
    'MAE':[round(mae_knn,  4), round(mae_svd,  4)],
})

resultados['Melhor RMSE'] = resultados['RMSE'] == resultados['RMSE'].min()
resultados['Melhor MAE']  = resultados['MAE']  == resultados['MAE'].min()

print("Comparativo de Desempenho: SVD vs KNN Baseline\n")
print(resultados.to_string(index=False))

modelo_vencedor = resultados.loc[resultados['RMSE'].idxmin(), 'Modelo']
print(f"\nModelo selecionado para recomendação final: {modelo_vencedor} menor RMSE")


# calcula o Hit Rate@K no catálogo completo
# para cada usuário no conjunto de teste, pego os jogos que ele realmente gostou
# (engajamento >= threshold), escondo esses jogos e peço ao modelo para ranquear
# todos os jogos do catálogo que ele ainda não jogou se algum dos jogos escondidos
# aparecer no top-k, conto como acerto.

def hit_rate_full_catalog(modelo, df_teste, df_todos, k=10, threshold=3):

    todos_jogos = df_todos['game_title'].unique()
    hits  = 0
    total = 0

    usuarios_teste = df_teste['user_id'].unique()

    for uid in usuarios_teste:
        # pego os jogos relevantes desse usuário no conjunto de teste
        jogos_relevantes = set(
            df_teste[(df_teste['user_id'] == uid) & (df_teste['engajamento'] >= threshold)]['game_title']
        )
        if not jogos_relevantes:
            continue

        # pego todos os jogos que o usuário ja jogou no treino para excluir das previsões
        jogos_conhecidos = set(df_treino[df_treino['user_id'] == uid]['game_title'])

        # o catálogo de candidatos é todos os jogos menos os que ele já conhece
        candidatos = [j for j in todos_jogos if j not in jogos_conhecidos]

        # gero previsões para todos os candidatos e ranqueio pelo engajamento estimado
        previsoes = [modelo.predict(uid, jogo) for jogo in candidatos]
        previsoes.sort(key=lambda x: x.est, reverse=True)
        top_k_jogos = {pred.iid for pred in previsoes[:k]}

        # verifico se algum jogo relevante do teste apareceu no top-k do catálogo completo
        if jogos_relevantes & top_k_jogos:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0


# calcula o Hit Rate@K apenas dentro do conjunto de teste fechado.
# mantido aqui para comparação direta com o método correto de full-catalog.


hr_real = hit_rate_full_catalog(svd, df_teste, df_filtrado, k=10, threshold=3)
print(f"Hit Rate@10 catálogo completo: {hr_real:.4f}  ({hr_real * 100:.1f}%)")

# gera uma lista personalizada com os N jogos mais recomendados para um usuário,
# excluindo títulos que ele já possui em seu histórico de interações.
# a ordenação é feita pela nota de engajamento estimada pelo modelo.


def recomendar_jogos(usuario_id, modelo, df, n=10):

    todos_jogos = df['game_title'].unique()
    jogos_do_usuario  = df[df['user_id'] == usuario_id]['game_title'].tolist()
    jogos_nao_jogados = [j for j in todos_jogos if j not in jogos_do_usuario]

    # gera previsões de engajamento para todos os jogos ainda não conhecidos pelo usuário
    previsoes = [modelo.predict(usuario_id, jogo) for jogo in jogos_nao_jogados]
    previsoes.sort(key=lambda x: x.est, reverse=True)

    print(f"Top {n} recomendações para o usuário {usuario_id}:\n")
    for i, pred in enumerate(previsoes[:n], 1):
        print(f"  {i:>2}. {pred.iid:<45} engajamento previsto: {pred.est:.2f}")

# executa a função para o primeiro usuário da base filtrada como demonstração
usuario_teste = df_filtrado['user_id'].iloc[0]
recomendar_jogos(usuario_teste, svd, df_filtrado)
