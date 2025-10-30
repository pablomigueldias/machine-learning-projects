Projeto 01 — Previsão de Notas dos Estudantes

Este projeto tem como objetivo prever a **nota de matemática (math score)** de estudantes com base em informações socioeducacionais e desempenho em outras disciplinas.  
O projeto faz parte da minha trilha de aprendizado em **Machine Learning**, cobrindo desde a **análise exploratória de dados (EDA)** até o **deploy do modelo treinado**.

---

## Objetivo

Prever a **nota de matemática** a partir de variáveis como:
- gênero do estudante  
- grupo étnico  
- nível educacional dos pais  
- tipo de almoço (standard ou free/reduced)  
- participação em curso preparatório  
- notas de leitura e escrita

---

## Estrutura do Projeto

machine-learning-projects/
│
├── projeto_01_previsao_notas/
│ ├── data/
│ │ └── StudentsPerformance.csv
│ │
│ ├── imagens/
│ │ ├── comparacao.png
│ │ ├── correlacao-notas.png
│ │ ├── dispersao-residuos.png
│ │ ├── distribuicao-de-notas.png
│ │ ├── distribuicao-residuos.png
│ │ ├── influencia-curso.png
│ │ ├── media-por-genero.png
│ │ └── regressao-linear.png
│ │
│ ├── models/
│ │ └── melhor_modelo.joblib
│ │
│ ├── notebooks/
│ │ └── previsao_notas.ipynb
│ │
│ ├── src/
│ │ ├── train.py
│ │ └── predict.py
│ │
│ └── README.md



---

## 🔍 Etapas do Projeto

### **1️⃣ Análise Exploratória de Dados (EDA)**

Foram analisadas as distribuições das notas e suas correlações:

| Gráfico | Descrição |
|----------|------------|
| ![Distribuição das Notas](reports/distribuicao-de-notas.png) | Mostra a frequência das notas de matemática, leitura e escrita. |
| ![Correlação entre Notas](reports/correlacao-notas.png) | Correlação forte entre leitura e escrita (0.95), e moderada entre leitura e matemática (0.82). |

---

### **2️⃣ Análises Categóricas**

| Gráfico | Interpretação |
|----------|---------------|
| ![Influência do Curso Preparatório](reports/influencia-curso.png) | Estudantes que completaram o curso preparatório tiveram notas de matemática mais altas. |
| ![Média por Gênero](reports/media-por-genero.png) | Homens apresentaram leve vantagem nas notas de matemática em relação às mulheres. |

---

### **3️⃣ Treinamento e Avaliação dos Modelos**

Modelos testados:
- Linear Regression  
- Decision Tree  
- Random Forest  
- KNN  
- SVR  

| Gráfico | Interpretação |
|----------|---------------|
| ![Comparação de Modelos](reports/comparacao.png) | A **Regressão Linear** apresentou o melhor desempenho (R² ≈ 0.87). |

---

### **4️⃣ Avaliação do Modelo Final**

| Gráfico | Interpretação |
|----------|---------------|
| ![Regressão Linear — Real vs Predito](reports/regressao-linear.png) | Forte relação linear entre valores reais e previstos. |
| ![Dispersão dos Resíduos](reports/dispersao-residuos.png) | Resíduos distribuídos aleatoriamente, indicando um bom ajuste. |
| ![Distribuição dos Resíduos](reports/distribuicao-residuos.png) | Distribuição aproximadamente normal, reforçando a consistência do modelo. |

---

## Métricas do Modelo Final

| Métrica | Valor |
|----------|-------|
| R²       | 0.87 |
| MAE      | 3.42 |
| RMSE     | 4.91 |

O modelo explica **87% da variação das notas de matemática** com erro médio de aproximadamente **3,4 pontos**.

---

## Deploy e Inferência

O modelo final foi salvo e carregado para predição de novos alunos.

### Exemplo de uso — `src/predict.py`
```python
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(BASE_DIR, "..", "models", "melhor_modelo.joblib")

modelo_path = os.path.normpath(modelo_path)

print(" Carregando modelo de:", modelo_path)

model = joblib.load(modelo_path)

novos_dados = pd.DataFrame([
    {
        "gender": "female",
        "race/ethnicity": "group C",
        "parental level of education": "bachelor's degree",
        "lunch": "standard",
        "test preparation course": "completed",
        "reading score": 78,
        "writing score": 74
    },
    {
        "gender": "male",
        "race/ethnicity": "group D",
        "parental level of education": "some college",
        "lunch": "free/reduced",
        "test preparation course": "none",
        "reading score": 60,
        "writing score": 58
    }
])

predicoes = model.predict(novos_dados)

for i, p in enumerate(predicoes, 1):
    print(f"Aluno {i}: nota prevista de matemática = {round(p, 2)}")
```

## 🧩 Tecnologias Utilizadas

Python 3.11
Pandas
NumPy
Scikit-Learn
Seaborn / Matplotlib
Joblib



