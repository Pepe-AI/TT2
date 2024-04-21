from Modelo import ModeloClasificador
from limpieza_unique import LimpiadorCV
import pandas as pd
import numpy as np
import PyPDF2

path = "diccionarios"
categoria = "Data Science"


# Ruta al archivo PDF en Google Drive
file_path = 'Pruebas con PDFs/CV6.pdf'

#  Abre el archivo PDF en modo lectura binaria
with open(file_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()


print(len(text))

# Instancia de la clase de limpieza
limpiador = LimpiadorCV(path)
Clean_Resume = limpiador.clean_resume(text)
print(len(Clean_Resume))

Clean_Resume = limpiador.clean_useless_words(Clean_Resume, categoria)
print(len(Clean_Resume))

# Crear un objeto ModeloClasificador con el modelo entrenado
clasificador = ModeloClasificador('xgb_model.joblib', 'tfidf_vectorizer.joblib', 'selected_features.joblib', 'pca.joblib', 'label_encoder.joblib')
predicted_category = clasificador.predecir(Clean_Resume)
print(predicted_category)