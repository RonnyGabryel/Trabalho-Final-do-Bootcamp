# Sistema de Recomendação Inteligente de Jogos para Plataformas Digitais

Projeto final do Bootcamp de Aprendizado de Máquina da LAMIA. O objetivo é construir um sistema de recomendação de jogos baseado no histórico de horas jogadas dos usuários na Steam, utilizando filtragem colaborativa com o algoritmo SVD.

---

##  Sumário
- [Introdução](#introdução)
- [Problema](#problema)
- [Base de Dados](#base-de-dados)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Conclusões](#conclusões)
- [Como rodar o projeto](#como-rodar-o-projeto)

---

## Introdução

Com dezenas de milhares de jogos disponíveis na Steam, encontrar algo que realmente valha a pena jogar virou um problema real. O usuário acaba gastando mais tempo rolando a página do que jogando de verdade. Este projeto aplica técnicas de aprendizado de máquina para resolver esse problema, transformando o histórico de horas jogadas em recomendações personalizadas.

---

## Problema

A sobrecarga de opções em plataformas digitais de jogos prejudica a experiência do usuário e a retenção dentro da plataforma. O desafio técnico está em transformar dados implícitos  o tempo que o usuário passa jogando em uma métrica de engajamento capaz de alimentar um modelo de recomendação eficiente.

---

## Base de Dados

- **Fonte:** [Steam Video Games Dataset — Kaggle](https://www.kaggle.com/datasets/tamber/steam-video-games)
- **Tamanho:** ~200.000 registros de interações de usuários reais da Steam
- **Colunas:** `user_id`, `game_title`, `behavior`, `hours_played`

---

## Metodologia

**Pré-processamento:**
- Remoção de contas suspeitas
- Filtro de esparsidade: usuários com pelo menos 5 jogos e títulos com mais de 10 interações
- Conversão das horas jogadas em uma escala de engajamento de 1 a 5

**Modelos utilizados:**
- SVD (Singular Value Decomposition) via biblioteca Surprise modelo principal
- KNN Baseline modelo de comparação

**Métricas de avaliação:**

| Métrica | Meta |
|---|---|
| RMSE | < 1,10 |
| MAE | < 0,90 |
| Hit Rate@10 | > 75% |

---

## Resultados

| Métrica | Meta | Resultado |
|---|---|---|
| RMSE | < 1,10 | 1,0327 ✅ |
| MAE | < 0,90 | 0,8385 ✅ |
| Hit Rate@10 | > 75% | 79,3% ✅ |

O modelo SVD superou o KNN Baseline em todas as métricas. A métrica Hit Rate@10 foi escolhida por ser mais adequada para datasets com alta esparsidade como o da Steam, onde a maioria dos usuários interagiu com menos de 1% do catálogo disponível. Ela mede se pelo menos um dos 10 jogos recomendados é relevante para o usuário, o que reflete melhor a experiência real de recomendação.

---

## Conclusões

O projeto mostrou que é possível construir um sistema de recomendação funcional usando apenas o histórico de horas jogadas como sinal de engajamento. O SVD com fatoração de matrizes se mostrou superior ao KNN Baseline e conseguiu gerar recomendações coerentes com o perfil de cada jogador. A principal limitação encontrada foi a esparsidade extrema do dataset, característica natural de plataformas com catálogos muito grandes, que limita métricas mais rígidas como a Precision@K.

---

## Como rodar o projeto

> **Atenção:** a biblioteca scikit-surprise funciona melhor com Python 3.12.x ou inferior. Recomendamos usar o Google Colab para evitar problemas de compatibilidade.

**Instalando as dependências:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise
```

**Rodando no Google Colab:**
1. Acesse [Google Colab](https://colab.research.google.com)
2. Faça o upload do notebook ou abra direto pelo GitHub
3. Execute as células em ordem

**Rodando localmente:**
1. Clone o repositório
```bash
git clone https://github.com/RonnyGabryel/Trabalho-Final-do-Bootcamp.git
```
2. Acesse a pasta
```bash
cd Trabalho-Final-do-Bootcamp
```
3. Abra o notebook
```bash
jupyter notebook
```
4. Execute o arquivo `Sistema de Recomendação Inteligente de Jogos para Plataformas Digitais.ipynb`

---

## Documentação das bibliotecas

- **scikit-surprise:** [surprise.readthedocs.io](https://surprise.readthedocs.io/)
- **pandas:** [pandas.pydata.org](https://pandas.pydata.org/docs/)
- **numpy:** [numpy.org/doc](https://numpy.org/doc/)
- **matplotlib:** [matplotlib.org](https://matplotlib.org/stable/index.html)
- **seaborn:** [seaborn.pydata.org](https://seaborn.pydata.org)
- **scikit-learn:** [scikit-learn.org](https://scikit-learn.org/stable/)
