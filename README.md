<img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdTYzbGxzYjZ5dGVsNnVnNGNzaWRhM25leWUxdDVzazFqbzQ0d2k5MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/doXBzUFJRxpaUbuaqz/giphy.gif" alt="Steam UI" width="100%" height="300px">

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Steam-000000?style=for-the-badge&logo=steam&logoColor=white" alt="Steam">
  <img src="https://img.shields.io/badge/PyCharm-000000?style=for-the-badge&logo=pycharm&logoColor=white" alt="PyCharm">
</p>

# Sistema de Recomendação Inteligente de Jogos para Plataformas Digitais

Projeto final do Bootcamp de Aprendizado de Máquina da LAMIA. O objetivo é construir um sistema de recomendação de jogos baseado no histórico de horas jogadas dos usuários na Steam, utilizando filtragem colaborativa com o algoritmo SVD (Decomposição em Valores Singulares).

---

## Sumário

> **01.** [Introdução](#introdução)  
> **02.** [Base de Dados](#base-de-dados)  
> **03.** [Metodologia](#metodologia)  
> **04.** [Resultados](#resultados)  
> **05.** [Conclusões](#conclusões)  
> **06.** [Como Rodar o Projeto](#como-rodar-o-projeto)  
> **07.** [Pitch do Projeto](#pitch-do-projeto)
---

## Introdução

Com dezenas de milhares de jogos disponíveis na Steam, encontrar algo que realmente valha a pena jogar virou um problema real. O usuário acaba gastando mais tempo rolando a página do que jogando de verdade. Este projeto aplica técnicas de aprendizado de máquina para resolver esse problema, transformando o histórico de horas jogadas em recomendações personalizadas.

---

## Base de Dados

- **Fonte:** [Steam Video Games Dataset — Kaggle](https://www.kaggle.com/datasets/tamber/steam-video-games)
- **Tamanho bruto:** 200.000 registros de interações de usuários reais da Steam
- **Tamanho após pré-processamento:** 50.653 registros | 2.435 usuários | 1.011 jogos
- **Colunas:** `user_id`, `game_title`, `behavior`, `hours_played`
- **Esparsidade da matriz usuário-jogo:** 97,94%

---

## Metodologia

**Pré-processamento:**
- Isolamento exclusivo de registros com `behavior = 'play'`, descartando entradas de compra que não representam engajamento real
- Filtro de esparsidade: usuários com histórico mínimo de 5 jogos e títulos com pelo menos 10 interações registradas
- Construção da métrica de engajamento por normalização percentil por jogo (descrita abaixo)

**Métrica de Engajamento Normalização por Percentil:**

A abordagem de limiares fixos de horas foi descartada por ignorar as diferenças estruturais entre gêneros de jogos. Um RPG com 20 horas representa engajamento mediano, enquanto 20 horas em um jogo casual representa saturação total. Para resolver isso, a nota de engajamento de cada usuário é calculada com base em sua posição percentil na distribuição de horas do respectivo jogo, tornando a escala relativa ao comportamento dos demais jogadores do mesmo título. O resultado foi uma distribuição uniforme entre as cinco classes:

| Nota | Registros |
|---|---|
| 1 | 11.306 |
| 2 | 9.544 |
| 3 | 9.803 |
| 4 | 9.726 |
| 5 | 10.274 |

**Modelos utilizados:**
- **SVD** (Decomposição em Valores Singulares) via biblioteca Surprise modelo principal
- **KNN Baseline** modelo de referência para comparação

**Validação:**
- Divisão treino/teste: 80%/20% com `random_state=42`
- Validação cruzada com 5 folds aplicada ao SVD para verificação de estabilidade

**Métricas de avaliação:**

| Métrica | Meta |
|---|---|
| RMSE (SVD) | inferior ao KNN Baseline |
| MAE (SVD) | inferior ao KNN Baseline |
| Hit Rate@10 | > 80% |

---

## Resultados

**Validação Cruzada  SVD (5-fold):**

| Métrica | Média | Desvio Padrão |
|---|---|---|
| RMSE | 1,4327  | 0,0071 |
| MAE | 1,2269 | 0,0066 |

O desvio padrão reduzido confirma que o modelo é estável e generaliza consistentemente entre diferentes partições dos dados.

**Comparativo SVD vs KNN Baseline Conjunto de Teste:**

| Modelo | RMSE | MAE |
|---|---|---|
| KNN Baseline | 1,4559 | 1,2301 |
| **SVD** | **1,4206** | **1,2179** |

**Hit Rate@10:**

| Métrica | Meta | Resultado |
|---|---|---|
| Hit Rate@10 | > 15% | 15,5% |

O SVD superou o KNN Baseline em todas as métricas e foi selecionado como modelo final. A métrica Hit Rate@10 indica que 15,5%. dos usuários receberam ao menos uma recomendação genuinamente relevante entre as 10 primeiras sugestões geradas, superando a meta estabelecida.

> **Nota sobre os valores absolutos de RMSE e MAE:** Os valores obtidos são naturalmente superiores aos de projetos que utilizam limiares fixos de horas, pois a normalização por percentil produz uma distribuição uniforme entre as notas, maximizando a variância dos rótulos e tornando a tarefa de predição mais exigente. O indicador mais representativo da utilidade prática do sistema é o Hit Rate@10.

---

## Conclusões

O projeto demonstrou que é possível construir um sistema de recomendação funcional e robusto utilizando exclusivamente o histórico de horas jogadas como sinal de engajamento. A substituição de limiares fixos pela normalização percentil por jogo representou uma melhoria metodológica relevante, produzindo uma escala de engajamento mais fiel ao comportamento real dos usuários. O SVD com fatoração de matrizes se mostrou superior ao KNN Baseline em todas as métricas avaliadas e gerou recomendações coerentes com o perfil de cada jogador. A principal limitação encontrada foi a esparsidade extrema de 97,94% do dataset, característica natural de plataformas com catálogos muito grandes, que limita métricas mais rígidas como a Precision@K.


---

##  Como rodar o projeto

> **Nota Metodológica sobre Compatibilidade:** A biblioteca `scikit-surprise` (utilizada para a fatoração de matrizes colaborativa) apresenta incompatibilidades estritas com o ecossistema moderno do Python (especificamente versões do Python iguais ou superiores à 3.13 e versões do NumPy na árvore 2.x). Isso ocorre devido a alterações nas interfaces de tipos em C/Cython dentro do núcleo do NumPy moderno. Para mitigar falhas de compilação, assegure o uso do **Python 3.11 (ou inferior)** e force o *downgrade* do NumPy para a versão estável **1.26.4**, conforme as instruções abaixo.

### Execução do Projeto

Para a reprodução dos experimentos localmente, recomenda-se a utilização de um gerenciador de ambientes virtuais (como Virtualenv ou Conda) configurado com o **Python 3.10 ou Python 3.11** no PyCharm.
  
1. Efetue a clonagem do repositório via terminal:

```
git clone https://github.com/RonnyGabryel/Trabalho-Final-do-Bootcamp.git
```
  
2. Navegue até o diretório raiz do projeto:
  
```
cd Trabalho-Final-do-Bootcamp
```
  
3. Instale as dependências respeitando estritamente a ordem de precedência para evitar sobreposições de versão:

```
pip install -r requirements.txt
```

Dica para usuários de Anaconda/Miniconda: Para evitar problemas de compilação do scikit-surprise, você também pode instalar os binários pré-compilados diretamente via Conda executando:
```
conda install -c conda-forge numpy=1.26.4 scikit-surprise pandas matplotlib seaborn scikit-learn -y
```

4. Abra a pasta do projeto no PyCharm, certifique-se de vincular o interpretador e execute o script principal:
   
```
python Sistema_de_Recomendação_Inteligente_de_Jogos_para_Plataformas_Digitais.py
```

## Documentação das Bibliotecas


| Biblioteca | Link Oficial |
| :--- | :--- |
| <img src="https://img.shields.io/badge/scikit--surprise-00599C?style=flat-square" alt="Surprise"> | [surprise.readthedocs.io](https://surprise.readthedocs.io/) |
| <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"> | [pandas.pydata.org](https://pandas.pydata.org/docs/) |
| <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"> | [numpy.org/doc](https://numpy.org/doc/) |
| <img src="https://img.shields.io/badge/Matplotlib-ffffff?style=flat-square&logo=matplotlib&logoColor=black" alt="Matplotlib"> | [matplotlib.org](https://matplotlib.org/stable/index.html) |
| <img src="https://img.shields.io/badge/Seaborn-4C516D?style=flat-square" alt="Seaborn"> | [seaborn.pydata.org](https://seaborn.pydata.org) |
| <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"> | [scikit-learn.org](https://scikit-learn.org/stable/) |

### Carregar o modelo pre-treinar (opcional)

O arquivo `modelo_svd.pkl` disponível no repositório contém o modelo já 
treinado e serializado. Para utilizá-lo diretamente sem executar o pipeline 
completo de treinamento, basta adicionar ao script:

```python
import pickle

with open('modelo_svd.pkl', 'rb') as f:
    svd = pickle.load(f)
```

> O módulo `pickle` é nativo do Python e não requer instalação adicional.

---

---

## 🎬 Pitch do Projeto

Clique na imagem abaixo para assistir ao vídeo de apresentação do projeto:

[![Assista ao Pitch do Projeto](https://img.youtube.com/vi/gyXdjMZsKn0/maxresdefault.jpg)](https://www.youtube.com/watch?v=gyXdjMZsKn0)
