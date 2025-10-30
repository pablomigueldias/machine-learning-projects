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

