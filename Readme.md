# Portuguese Banking Marketing — Predictive Machine Learning & Lakehouse Architecture

Projeto de Machine Learning desenvolvido para prever a conversão de clientes em depósitos a prazo durante abordagens de marketing bancário. A solução abrange desde o pré-processamento estrito para evitar vazamento de dados (*data leakage*) até a interpretação preditiva com SHAP Values e o desenho da arquitetura de implantação em Azure Databricks.

---

## Sumário

- [Portuguese Banking Marketing — Predictive Machine Learning \& Lakehouse Architecture](#portuguese-banking-marketing--predictive-machine-learning--lakehouse-architecture)
  - [Sumário](#sumário)
  - [1. Visão Geral do Problema](#1-visão-geral-do-problema)
  - [2. Análise e Qualidade dos Dados](#2-análise-e-qualidade-dos-dados)
    - [Prevenção de Vazamento de Dados (Data Leakage)](#prevenção-de-vazamento-de-dados-data-leakage)
    - [Tratamento de Indeterminações e Regra de Negócio](#tratamento-de-indeterminações-e-regra-de-negócio)
  - [3. Engenharia de Pré-Processamento](#3-engenharia-de-pré-processamento)
  - [4. Modelagem e Resultados](#4-modelagem-e-resultados)
    - [Tabela Comparativa de Performance](#tabela-comparativa-de-performance)
    - [Justificativa da Escolha](#justificativa-da-escolha)
  - [5. Interpretação do Modelo e Insights (SHAP)](#5-interpretação-do-modelo-e-insights-shap)
    - [Principais Variáveis Preditivas](#principais-variáveis-preditivas)
  - [6. Recomendações Estratégicas](#6-recomendações-estratégicas)
  - [7. Arquitetura de Produção (Azure Databricks)](#7-arquitetura-de-produção-azure-databricks)
  - [8. Google Colab](#8-google-colab)

---

## 1. Visão Geral do Problema

O objetivo técnico é mensurar a propensão de clientes para contratação de depósitos a prazo, permitindo a priorização de abordagens comerciais e a otimização de custos operacionais do call center.

- **Dataset:** Bank Marketing Dataset (Kaggle) — 45.211 registros e 17 atributos históricos.
- **Variável Alvo ($y$):** Binária (`1` = Converteu, `0` = Não converteu).
- **Distribuição de Classe:** Apenas 11,7% da base histórica representa conversões positivas, exigindo métricas de validação focadas em ordenação de risco e separabilidade (ROC-AUC).

---

## 2. Análise e Qualidade dos Dados

### Prevenção de Vazamento de Dados (Data Leakage)

> **Decisão Metodológica:** A variável `duration` (duração da chamada) foi removida da matriz de atributos de entrada.
>
> *Racional:* A duração de uma ligação é uma variável futura, disponível apenas após o término do contato comercial. Mantê-la no treinamento causaria um vazamento de informação, resultando em métricas irrealistas em fase de testes e falha completa do modelo em produção pré-campanha.

### Tratamento de Indeterminações e Regra de Negócio

- **Integridade Física:** O dataset não contém valores nulos do tipo `NaN`.
- **Semântica do Atributo `unknown`:** As categorias registradas como `unknown` em `poutcome` (81,75%) e `contact` (28,80%) não foram removidas ou imputadas.
- **Validação Cruzada:** A taxa de `unknown` em `poutcome` coincide com o indicador `pdays = -1` (81,74%), confirmando que o dado ausente representa clientes sem histórico de campanhas anteriores (primeiro contato no funil).

---

## 3. Engenharia de Pré-Processamento

As transformações foram estruturadas em um `ColumnTransformer` do Scikit-Learn e aplicadas estritamente após a separação de treino e teste para isolar a base de validação.

```text
 Base Bruta (45.211 registros)
       │
       ├── Divisão Estratificada (70% Treino / 30% Teste)
       │
       └── ColumnTransformer
             ├── Atributos Numéricos (6)   ──> StandardScaler (Z-Score)
             └── Atributos Categóricos (9) ──> OneHotEncoder (drop='first')
                                                     │
                                           Matriz Processada (41 Features)
```

1. **Divisão Estratificada:** Preserva a proporção de 11,7% de conversão entre treino (31.647 amostras) e teste (13.564 amostras).
2. **Padronização Numérica ($Z = \frac{x - \mu}{\sigma}$):** Aplicação de `StandardScaler` sobre atributos contínuos (`balance`, `age`, `campaign`, `pdays`, `previous`, `day`).
3. **Vetorização Categórica:** Conversão de variáveis qualitativas via One-Hot Encoding com `drop='first'` para prevenir multicolinearidade perfeita (*Dummy Variable Trap*).
4. **Tratamento do Desbalanceamento:** O rebalanceamento foi controlado na função de perda dos algoritmos via hiperparâmetros (`scale_pos_weight` e `class_weight='balanced'`), evitando reamostragens sintéticas na base de teste.

---

## 4. Modelagem e Resultados

Avaliação comparativa na base de teste não vista (13.564 amostras):

### Tabela Comparativa de Performance

| Modelo | ROC-AUC | Precisão (y=1) | Recall (y=1) | F1-Score (y=1) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Classifier** | **0,7993** | **0,35** | **0,63** | **0,45** | **Modelo Selecionado** |
| **Random Forest** | 0,7878 | 0,35 | 0,76 | 0,48 | Baseline Ensembles |
| **Regressão Logística** | 0,7700 | 0,28 | 0,94 | 0,43 | Baseline Linear |

### Justificativa da Escolha

O **XGBoost** obteve o maior valor de ROC-AUC (0,7993), apresentando a melhor capacidade global de ordenação de probabilidade. O modelo manteve uma taxa de Recall de 63% sobre a classe minoritária com o menor índice de falsos positivos entre os algoritmos avaliados.

---

## 5. Interpretação do Modelo e Insights (SHAP)

A análise de importância de atributos foi realizada com **SHAP (SHapley Additive exPlanations)** no modelo XGBoost.

### Principais Variáveis Preditivas

1. **Histórico Interativo (`poutcome_success`):** Apresenta o maior impacto positivo individual na propensão de contratação.
2. **Validação de Canal (`contact_unknown`):** Atua como o principal fator de penalização da pontuação de conversão.
3. **Efeito Sazonal (`month_oct`, `month_mar`, `month_dec` vs `month_may`):** Meses com menor volume total de abordagens mostram eficiência proporcionalmente maior do que meses de pico operacional (Maio).
4. **Comprometimento Financeiro (`housing_yes` e `balance`):** A existência de crédito imobiliário reduz a propensão de adesão, enquanto o volume do saldo bancário atua como preditor positivo.

---

## 6. Recomendações Estratégicas

1. **Higienização do Mailing de Abordagem:** Interromper contatos telefônicos diretos para registros com canal marcado como `unknown`. Direcionar este segmento para atualização de dados em meios digitais.
2. **Readequação do Calendário de Ofertas:** Reduzir a concentração de chamadas no mês de Maio e redistribuir a força de vendas entre os meses de Março, Outubro e Dezembro.
3. **Roteamento de Leads em CRM:** Priorizar o atendimento telefônico para o grupo de clientes com registro `poutcome == success`.

---

## 7. Arquitetura de Produção (Azure Databricks)

Proposta de arquitetura corporativa para servimento do modelo em ambiente de nuvem:

```text
[Fontes de Dados] ──> [Bronze Layer] ──> [Silver Layer] ──> [Gold Layer / Feature Store]
 (CRM / Logs)         (Delta Lake)        (Limpeza/OHE)          (Score / Prep)
                                                                       │
                                                                 [Unity Catalog]
                                                                       │
                                                               [MLflow Registry]
                                                                       │
                                           ┌───────────────────────────┴───────────────────────────┐
                                           ▼                                                       ▼
                             [Batch Scoring Workflow]                           [Real-time Model Serving]
                            (Pontuação Diária no CRM)                                (API RESTful)
```

- **Camada Lakehouse (Medallion):** Ingestão via Databricks Auto Loader (Bronze), limpeza e transformação no padrão Delta Lake (Silver) e persistência de atributos no Databricks Feature Store (Gold).
- **Governança (MLflow + Unity Catalog):** Rastreamento de experimentos, versionamento de código/artefatos e controle de acesso baseado em funções (RBAC).
- **Estratégia de Servimento:** Inferência diária em lote para atualização de listas no CRM e disponibilização de endpoint REST via Databricks Model Serving para consulta em tempo real.

---

## 8. Google Colab

- [Executar Notebook no Google Colab](https://colab.research.google.com/drive/1FNz1s8H1EJ2VqhGsVm51dSNziVCAK8OK?usp=sharing)                    