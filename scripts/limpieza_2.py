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
data = pd.read_csv('updated_data_final_cleaned3.csv', encoding="utf-8")

resumeDataSet = data.copy()


def visualize_cloud(label, data):
    # Filtrar los resúmenes por la etiqueta 'Web Developer'
    web_dev_resumes = data[data['Category'] == 'Testing']['Resume']

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
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) # remove non-ascii characters
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    resumeText = resumeText.lower()

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

def clean_ussles_words(resumen, categoria):
    # Convertimos la categoría a minúsculas para asegurar la correspondencia con las claves del diccionario
    palabras_irrelevantes = diccionarios.get(categoria, set())
    palabras_limpias = [palabra for palabra in resumen.split() if palabra not in palabras_irrelevantes]
    return ' '.join(palabras_limpias)



resumeDataSet['cleaned_Resume'] = resumeDataSet["Resume"].map(cleanResume)

resumeDataSet['cleaned_Resume'] = resumeDataSet['cleaned_Resume'].str.replace('Â', '')
resumeDataSet['cleaned_Resume'] = resumeDataSet['cleaned_Resume'].str.replace('â', '')




#resumeDataSet['cleaned_resume'] = resumeDataSet['cleaned_Job_Description'] + resumeDataSet['cleaned_Responsibilities'] + resumeDataSet['cleaned_skills']


#despu8es de limpiar los datos, se procede a elimnar las palabras irrelevantes para cada etiqeuta

# Ruta a la carpeta que contiene tus archivos de diccionarios
carpeta_diccionarios = 'diccionarios'

# Cargar los diccionarios de palabras irrelevantes desde los archivos
diccionarios = {}


for archivo in os.listdir(carpeta_diccionarios):
    categoria = archivo.split('.')[0]  # Asume que el nombre del archivo es exactamente la categoría
    with open(os.path.join(carpeta_diccionarios, archivo), 'r', encoding='utf-8') as f:
        # Crea un conjunto de palabras irrelevantes para cada categoría
         # Elimina las comas de cada palabra antes de agregarlas al conjunto
         diccionarios[categoria] = set(palabra.strip().rstrip(',') for palabra in f.read().splitlines())


# Filtrar los resúmenes por la etiqueta 'Web Developer'
web_dev_resumes = resumeDataSet[resumeDataSet['Category'] == 'Web Designing']['cleaned_Resume']

# Combinar todos los resúmenes en una única cadena de texto
combined_resumes = ' '.join(web_dev_resumes)

# Crear y visualizar la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_resumes)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No mostrar los ejes para una visualización más limpia
plt.show()




# Aplicar la función de limpieza a cada fila del DataFrame
resumeDataSet['cleaned_Resume'] = resumeDataSet.apply(lambda fila: clean_ussles_words(fila['cleaned_Resume'], fila['Category']), axis=1)



# Filtrar los resúmenes por la etiqueta 'Web Developer'
web_dev_resumes = resumeDataSet[resumeDataSet['Category'] == 'Web Designing']['cleaned_Resume']

# Combinar todos los resúmenes en una única cadena de texto
combined_resumes = ' '.join(web_dev_resumes)

# Crear y visualizar la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_resumes)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # No mostrar los ejes para una visualización más limpia
plt.show()


skills_dict = {
    'Blockchain': 'Ethereum Smart Contracts, Cryptography, Solidity, DApp Development, Hyperledger Fabric, Consensus Algorithms, Node.js, Corda, Blockchain System Design, Truffle, Ganache, Blockchain API, Ripple, Litecoin, Bitcoin, Blockchain Security, Key Management, DeFi, NFTs',
    'Data Science': 'Python, R, Machine Learning, Deep Learning, Statistical Analysis, pandas, NumPy, Data Visualization, Matplotlib, Seaborn, SQL, NoSQL, Spark, Hadoop, PySpark, Ensemble Models, PCA, t-SNE, Airflow, Keras, TensorFlow, Time Series, Predictive Analysis, A/B Testing',
    'Database': 'SQL, PL/SQL, Database Administration, MySQL, PostgreSQL, Oracle, MongoDB, Cassandra, Database Design, Normalization, Data Recovery, Query Optimization, Distributed Databases, Database Tuning, Replication, Sharding, Cluster Management, ETL, Data Warehousing, Real-time DB, Redis, Database Security, Compliance',
    'DevOps Engineer': 'CI/CD, Jenkins, GitLab CI, Docker, Kubernetes, Ansible, Terraform, Monitoring, Logging, Prometheus, Grafana, ELK, Unix Administration, Bash, Python, CloudFormation, Spinnaker, Secrets Management, Ruby, Perl, Vagrant',
    'Dotnet Developer': 'C#, .NET Framework, ASP.NET MVC, Entity Framework, LINQ, Azure, WPF, Windows Forms, RESTful APIs, .NET Core, Microservices, SignalR, RabbitMQ, CI/CD, TeamCity, Xamarin, Stress Testing',
    'Java Developer': 'Java, Spring Framework, Spring Boot, Hibernate, JPA, Maven, Gradle, SOAP, REST, JUnit, Design Patterns, Jersey, Netflix OSS, Spring Security, Android Development, SQL, NoSQL Integration, JVM Optimization',
    'Network Security Engineer': 'Firewalls, VPNs, IDS/IPS, SSL/TLS, SSH, Vulnerability Assessment, Pentesting, Intrusion Detection, ISO 27001, NIST, Cryptography, SIEMs, Digital Forensics, Firewall Configuration, Network Security Policies, DLP, CISSP, CISM',
    'Python Developer': 'Python, Django, Flask, Scripting, Data Analysis, Pandas, NumPy, Web Scraping, BeautifulSoup, Scrapy, API Development, FastAPI, Django REST, pytest, unittest, Pyramid, Bottle, asyncio, ORM, SQLAlchemy, Packaging, Kafka, RabbitMQ',
    'Testing': 'HTML, CSS, JavaScript, UX, UI, Frameworks, Graphic Design, Photoshop, Illustrator, Logo Design, Prototyping, Accessibility, SEO, Management, Test Animation, Optimization, Security, Responsive Design',
    'Web Designing': 'HTML, CSS, JavaScript, Responsive Design, UX, UI, Bootstrap, Foundation, Graphic Design, Photoshop, Illustrator, Logo Design, Prototyping, Accessibility, SEO, WordPress, Git, Web Animation, Debugging, Performance Optimization, Web Security'
}

# Función para encontrar las habilidades comunes
def find_common_skills(category):
    skills = []
    for key, value in skills_dict.items():
        if key == category:
            skills = value.split(', ')
            break
    return skills

# Agregar la columna de habilidades comunes
resumeDataSet['Common Skills'] = resumeDataSet['Category'].apply(find_common_skills)

resumeDataSet["Common Skills"] = [
    [palabra.lower() for palabra in lista] for lista in resumeDataSet["Common Skills"]
]

resumeDataSet["Common Skills"] = resumeDataSet["Common Skills"].str.join(" ")

print(resumeDataSet["Common Skills"])

resumeDataSet.to_csv("Datase_ML.csv", index=False)


