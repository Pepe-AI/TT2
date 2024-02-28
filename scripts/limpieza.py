#------------------Extra----------------------#
import numpy as np
import pandas as pd
import seaborn as sns
import os
import xgboost as xgb
import warnings
import gc
from bs4 import BeautifulSoup
import re
import time
from joblib import dump, load
from collections import Counter
from gensim.models import Word2Vec
#------------------Extra----------------------#


#------------------imblearn----------------------#
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
#------------------imblearn----------------------#


#------------------Sklearn----------------------#
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
#------------------Sklearn----------------------#


#------------------Matplotlib----------------------#
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#------------------Matplotlib----------------------#


#------------------Hyperopt----------------------#
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
#------------------Hyperopt----------------------#


#------------------Spacy----------------------#
import spacy 
from scipy import sparse
from spacy.lang.en.examples import sentences
#------------------Spacy----------------------#


#------------------NLTK----------------------#
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nlp = spacy.load("en_core_web_lg")
#------------------NLTK----------------------#


#from gensim.models import Word2Vec
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, Concatenate
#def EDA():

# Definimos una función para preprocesar el texto
def preprocess_text(x):
    # Pasar a minúsculas
    x = str(x).lower()

    # Reemplazar números con abreviaturas de letras
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    # Eliminar URL
    x = re.sub(r"http\S+", "", x)
    # Eliminar caracteres no alfanuméricos y símbolos de puntuación
    x = re.sub(r"\W+", " ", x)
    # Reemplazar acentos por su forma base
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n"
    }
    for key, value in replacements.items():
        x = x.replace(key, value)
    # Eliminar guiones y guiones bajos
    x = x.replace("-", " ").replace("_", " ")
    # Reemplazar caracteres especiales
    replacements = {
        "%": " percent ",
        "&": " and ",
        "$": " dollar ",
        "€": " euro "
    }
    for key, value in replacements.items():
        x = x.replace(key, value)

    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    words = x.split()
    filtered_words = [word for word in words if word not in stop_words]
    x = " ".join(filtered_words)

    # Lematizar las palabras utilizando spaCy (pasar a forma base las palabras)
    doc = nlp(x)
    x = " ".join(token.lemma_ for token in doc)
    # Eliminar etiquetas HTML utilizando BeautifulSoup
    bfs = BeautifulSoup(x, "html.parser")
    x = bfs.get_text()
    # Eliminar espacios en blanco adicionales
    x = x.strip()

    # Eliminar caracteres diferentes de letras y espacios
    x = re.sub(r'[^a-zA-Z\s]', '', x)

    return x

# Definimos una función para eliminar las palabras no relacionadas con cada etiqueta
def remove_unrelated_words(text, words_to_remove):
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in words_to_remove]
    return ' '.join(cleaned_words)

#Identificamos las palabras más comunes por etiqueta
def useless_words():
    # Paso 1: Agrupa tus datos por etiqueta
    grouped = df.groupby('Category')

    # Paso 2-5: Para cada grupo de etiqueta, combina todos los textos, tokeniza y cuenta la frecuencia de las palabras
    word_counts_by_label = {}
    for label, group in grouped:
        # Combina todos los textos en un solo texto
        combined_text = ' '.join(group['Resume'])
        # Tokeniza el texto
        words = combined_text.split()
        # Cuenta la frecuencia de cada palabra
        word_counts = Counter(words)
        # Encuentra las palabras más comunes por etiqueta (puedes ajustar el número)
        top_words = word_counts.most_common(10)  # Cambia 10 al número de palabras que desees obtener
        # Guarda los resultados
        word_counts_by_label[label] = top_words

    # Convierte el diccionario en un DataFrame para una mejor visualización
    word_counts_df = pd.DataFrame(word_counts_by_label)

    return word_counts_df

# Función para obtener embeddings de texto
def get_embeddings(text):
    embeddings = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(word2vec_model.vector_size)

# Funcion para creaer datos sinteticos y balancear el dataset
def creation_synthetic_data(X, y):
    # Definir el número deseado máximo de muestras para las clases
    max_samples_per_class = 550

    # Calcular el conteo de cada etiqueta
    label_counts = Counter(y)

    # Inicializar listas para almacenar las características y etiquetas después del corte
    X_cut = []
    y_cut = []

    # Iterar sobre las características y las etiquetas y mantener solo las muestras que no exceden el número máximo deseado de muestras por clase
    for i, label in enumerate(y):
        if label_counts[label] <= max_samples_per_class or len([1 for l in y_cut if l == label]) < max_samples_per_class:
            X_cut.append(X[i])
            y_cut.append(label)

    # Convertir las listas a arrays numpy
    X_cut = np.array(X_cut)
    y_cut = np.array(y_cut)

    # Imprimir el número de elementos nuevo por etiqueta después del corte
    label_counts_cut = Counter(y_cut)


    #Aplicar SMOTE
    smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
    X_res, y_res = smote.fit_resample(X_cut, y_cut)

    return X_res, y_res


#Cargar dataset
data = pd.read_csv('Curriculum Vitae.csv', encoding="utf-8")


#eliminar valores
valores_a_eliminar = ['HR','Advocate','Arts','Sales','Mechanical Engineer', 'Health and fitness','Civil Engineer', 'Business Analyst', 'Electrical Engineering', 'Operations Manager', 'PMO',"SAP Developer", "Automation Testing"]
df = data[~data['Category'].isin(valores_a_eliminar)]

df = df.copy()
df['Resume'] = df['Resume'].apply(preprocess_text)

#print(df.head())

# Dataframe con las palabras más comunes por etiqueta
word_counts_df = useless_words()

# Visualiza el DataFrame con las palabras más comunes por etiqueta




# Tokenización
tokenized_text = df['Resume'].apply(lambda x: x.split())
#eliminamos palabras en comun de todas las etiquetas que no aportan nada y puede generar sesgo
print(tokenized_text)

# Definimos las palabras que consideramos que no están relacionadas con cada etiqueta que no aporte, se busca encontrar mas para mejor entrenamiento
words_to_remove = {
    'Blockchain': ['description', 'exprience', 'month', 'detail', 'january'],
    'Data Science': ['exprience', 'month', 'year'],
    'Database': ['exprience', 'work', 'use', 'description', 'company'],
    'DevOps Engineer': ['project', 'exprience', 'month', 'team', 'company'],
    'DotNet Developer': ['project', 'exprience', 'month', 'team', 'detail', 'detail'],
    'ETL Developer': ['use', 'project', 'exprience', 'job', 'detail'],
    'Hadoop': ['use', 'description', 'project', 'exprience'],
    'Java Developer': ['exprience', 'month', 'company', 'year', 'use'],
    'Network Security Engineer': ['exprience', 'month', 'year'],
    'Python Developer': ['exprience', 'month', 'year'],
    'Testing': ['detail', 'exprience', 'month', 'company', 'project'],
    'Web Designing': ['project', 'description', 'use', 'exprience']
}



# Aplicamos la función a cada fila del DataFrame
for category, words_to_remove in words_to_remove.items():
    df.loc[df['Category'] == category, 'Resume'] = df[df['Category'] == category]['Resume'].apply(lambda x: remove_unrelated_words(x, words_to_remove))



# Entrenamiento del modelo Word2Vec
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=150, window=5, min_count=1, workers=4)

# Aplicando la función para obtener embeddings a la columna 'text'
df['text_embeddings'] = tokenized_text.apply(get_embeddings)


#obtenemos X xomo los datos de embedings
X = np.vstack(df['text_embeddings'].to_numpy())

# Codificación de etiquetas y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Category'])

X_res, y_res = creation_synthetic_data(X, y)

# Convertir las etiquetas codificadas de vuelta a la forma original
y_res = label_encoder.inverse_transform(y_res)



# Convertir y_res a DataFrame si aún no lo es
y_res_df = pd.DataFrame(y_res, columns=['Etiqueta'])

# Obtener el recuento de cada etiqueta
conteo_etiquetas = y_res_df['Etiqueta'].value_counts()



# Codificación de etiquetas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_res)
X = X_res



X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
# Unir los DataFrames
combined_df = pd.concat([y_res_df, X_df], axis=1)

#crear .csv con datos numericos y datos categoricos
combined_df.to_csv('ML_curriculum_vitae.csv', index=False)