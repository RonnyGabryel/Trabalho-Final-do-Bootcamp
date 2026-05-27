![Steam UI](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGpzd3E1N211ZXlmMmFjcGx0bWIzbm91YXU2dGNudGNheDdwYXRzbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/f3iwJFOVOwuy7K6FFw/giphy.gif)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Steam-000000?style=for-the-badge&logo=steam&logoColor=white" alt="Steam">
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab">
</p>

# Sistema de Recomendação Inteligente de Jogos para Plataformas Digitais

Projeto final do Bootcamp de Aprendizado de Máquina da LAMIA. O objetivo é construir um sistema de recomendação de jogos baseado no histórico de horas jogadas dos usuários na Steam, utilizando filtragem colaborativa com o algoritmo SVD ou Decomposição em Valores Singulares.

---

## Sumário

> **01.** [Introdução](#introdução)  
> **02.** [O Problema](#problema)  
> **03.** [Base de Dados](#base-de-dados)  
> **04.** [Metodologia](#metodologia)  
> **05.** [Resultados](#resultados)  
> **06.** [Conclusões](#conclusões)  
> **07.** [Como Rodar o Projeto](#como-rodar-o-projeto)

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
- SVD (Decomposição em Valores Singulares) usando biblioteca Surprise modelo principal
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

##  Como rodar o projeto

>  **Atenção:** o scikit-surprise tem conflito com versões recentes do NumPy.

**Rodando no Google Colab (recomendado):**

1. Acesse [Google Colab](https://colab.research.google.com)
2. Faça o upload do notebook ou abra direto pelo GitHub
3. Na primeira célula, execute exatamente isso antes de qualquer import:
```bash
!pip uninstall -y numpy scikit-surprise
!pip install numpy==1.26.4
!pip install scikit-surprise
```
4. Reinicie o ambiente após a instalação 
5. Execute as demais células em ordem

**Rodando localmente:**
1. Clone o repositório
```bash
git clone https://github.com/RonnyGabryel/Trabalho-Final-do-Bootcamp.git
```
2. Acesse a pasta
```bash
cd Trabalho-Final-do-Bootcamp
```
3. Instale as dependências na ordem correta
```bash
pip install numpy==1.26.4
pip install scikit-surprise
pip install pandas matplotlib seaborn scikit-learn
```
4. Abra o notebook
```bash
jupyter notebook
```
5. Execute o arquivo `Sistema_de_Recomendação_Inteligente_de_Jogos_para_Plataformas_Digitais.ipynb.ipynb`

## Documentação das Bibliotecas

| Biblioteca | Link Oficial |
| :--- | :--- |
| <img src="https://img.shields.io/badge/scikit--surprise-00599C?style=flat-square" alt="Surprise"> | [surprise.readthedocs.io](https://surprise.readthedocs.io/) |
| <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"> | [pandas.pydata.org](https://pandas.pydata.org/docs/) |
| <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"> | [numpy.org/doc](https://numpy.org/doc/) |
| <img src="https://img.shields.io/badge/Matplotlib-ffffff?style=flat-square&logo=matplotlib&logoColor=black" alt="Matplotlib"> | [matplotlib.org](https://matplotlib.org/stable/index.html) |
| <img src="https://img.shields.io/badge/Seaborn-4C516D?style=flat-square" alt="Seaborn"> | [seaborn.pydata.org](https://seaborn.pydata.org) |
| <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"> | [scikit-learn.org](https://scikit-learn.org/stable/) |
