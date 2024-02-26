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
#------------------Extra----------------------#


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


#def EDA():





#Cargar dataset
data = pd.read_csv('Curriculum Vitae.csv', encoding="utf-8")
print(data.shape)

#eliminar valores
valores_a_eliminar = ['HR','Advocate','Arts','Sales','Mechanical Engineer', 'Health and fitness','Civil Engineer', 'Business Analyst', 'Electrical Engineering', 'Operations Manager', 'PMO',"SAP Developer", "Automation Testing"]
df = data[~data['Category'].isin(valores_a_eliminar)]

from collections import Counter

# Codificación de etiquetas y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Category'])

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
print("Número de elementos por etiqueta después del corte:")
for label, count in label_counts_cut.items():
    print(f"Etiqueta: {label}, Elementos: {count}")