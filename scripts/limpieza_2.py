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
from wordcloud import WordCloud
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


from sklearn.metrics import confusion_matrix, log_loss
from sklearn.calibration import CalibratedClassifierCV

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
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

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


#Cargar dataset
data = pd.read_csv('archivo_filtrado.csv', encoding="utf-8")

resumeDataSet = data.copy()
resumeDataSet['cleaned_resume'] = ''

def visualize_cloud(label, data):
    # Filtrar los resúmenes por la etiqueta 'Web Developer'
    web_dev_resumes = data[data['Job Title'] == 'Web Developer']['cleaned_Job_Description']

    # Combinar todos los resúmenes en una única cadena de texto
    combined_resumes = ' '.join(web_dev_resumes)

    # Crear y visualizar la nube de palabras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_resumes)

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # No mostrar los ejes para una visualización más limpia
    plt.show()



def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace

    # Eliminar stopwords
    #stopwords = set(stopwords.words('english'))
    #words = resumeText.split()
    #filtered_words = [word for word in words if word not in stop_words]
    #resumeText = " ".join(filtered_words)
    # Lematizar las palabras utilizando spaCy (pasar a forma base las palabras)
    #doc = nlp(resumeText)
    #resumeText = " ".join(token.lemma for token in doc)
    # Eliminar etiquetas HTML utilizando BeautifulSoup
    #bfs = BeautifulSoup(resumeText, "html.parser")
    #resumeText = bfs.get_text()
    # Eliminar espacios en blanco adicionales
    #resumeText = resumeText.strip()

    return resumeText


resumeDataSet['cleaned_Job_Description'] = resumeDataSet["Job Description"].map(cleanResume)
resumeDataSet['cleaned_Responsibilities'] = resumeDataSet["Responsibilities"].map(cleanResume)
resumeDataSet['cleaned_skills'] = resumeDataSet["skills"].map(cleanResume)

#resumeDataSet['cleaned_resume'] = resumeDataSet['cleaned_Job_Description'] + resumeDataSet['cleaned_Responsibilities'] + resumeDataSet['cleaned_skills']


# Filtrar los resúmenes por la etiqueta 'Web Developer'
web_dev_resumes = resumeDataSet[resumeDataSet['Category'] == 'Testing']['Resume']
print("info",web_dev_resumes.count())
# Combinar todos los resúmenes en una única cadena de texto
combined_resumes = ' '.join(web_dev_resumes)

# Crear y visualizar la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_resumes)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No mostrar los ejes para una visualización más limpia
plt.show()







# Codificación de etiquetas y
label_encoder = LabelEncoder()
resumeDataSet['Job Title'] = label_encoder.fit_transform(resumeDataSet['Job Title'])

# Obtener los valores únicos de la variable "Category"
valores_unicos = resumeDataSet['Job Title'].unique()




