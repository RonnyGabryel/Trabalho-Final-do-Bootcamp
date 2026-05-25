# Sistema de Recomendação Inteligente de Jogos para Plataformas Digitais

Projeto final do Bootcamp de Aprendizado de Máquina da LAMIA. O objetivo é construir um sistema de recomendação de jogos baseado no histórico de horas jogadas dos usuários na Steam, utilizando filtragem colaborativa com o algoritmo SVD.

---

## 📑 Sumário
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

A sobrecarga de opções em plataformas digitais de jogos prejudica a experiência do usuário e a retenção dentro da plataforma. O desafio técnico está em transformar dados implícitos — o tempo que o usuário passa jogando — em uma métrica de engajamento capaz de alimentar um modelo de recomendação eficiente.

---

## Base de Dados

- **Fonte:** [Steam Video Games Dataset — Kaggle](https://www.kaggle.com/datasets/tamber/steam-video-games)
- **Tamanho:** ~200.000 registros de interações de usuários reais da Steam
- **Colunas:** `user_id`, `game_title`, `behavior`, `hours_played`

---

## Metodologia

**Pré-processamento:**
- Remoção de contas suspeitas (bots e perfis com tempos de jogo impossíveis)
- Filtro de esparsidade: usuários com pelo menos 5 jogos e títulos com mais de 10 interações
- Conversão das horas jogadas em uma escala de engajamento de 1 a 5

**Modelos utilizados:**
- SVD (Singular Value Decomposition) via biblioteca Surprise — modelo principal
- KNN Baseline — modelo de comparação

**Métricas de avaliação:**

| Métrica | Meta |
|---|---|
| RMSE | < 0,85 |
| MAE | < 0,50 |
| Precision@K | > 75% |

---

## Resultados

> *Seção a ser preenchida após o treinamento do modelo.*

---

## Conclusões

> *Seção a ser preenchida após a análise dos resultados.*

---

## Como rodar o projeto

**Pré-requisitos:**
- Python 3.8+
- Jupyter Notebook

**Instalando as dependências:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise
```

**Rodando:**
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
4. Execute o arquivo `recomendacao_jogos.ipynb`

---

## 📁 Estrutura do repositório
