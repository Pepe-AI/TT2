from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


#Cargar dataset
data = pd.read_csv('ML_Curriculum_Vitae.csv', encoding="utf-8")

# Inicializar una lista para almacenar los pares de filas con al menos un valor en común
pares_con_valores_comunes = []


# Recorrer el DataFrame fila por fila
for i in range(len(data) - 1):  # Ir hasta la penúltima fila
    #print("i:",i)
    for j in range(i + 1, len(data)):  # Comenzar desde la fila siguiente hasta la última
        #print("\nj:",j)
        # Verificar si hay algún valor en común
        if (data.iloc[i,1] == data.iloc[j,1]):

            # Si se encuentra un valor en común, guardar el par de filas
            pares_con_valores_comunes.append((i, j))
            break  # Pasar a la siguiente fila i según la condición dada


# Imprimir los resultados
print("i: ",i,"Valor 1: ",data.iloc[0,1])
print("j: ",j,"valor 2: ",data.iloc[10,1])
#print('Pares de filas con valores comunes:', pares_con_valores_comunes)








y = data['Etiqueta']
X = data.loc[:, 'feature_0':'feature_99']

# Imprimir los resultados
print(f'Se encontraron {len(pares_con_valores_comunes)} pares de filas con al menos un valor en común de un dataset de.{len(y)}')
#print('Pares de filas con valores comunes:', pares_con_valores_comunes)

#print (X.shape, y.shape)
#print (X.head())
#print (y.head())
#codificamos y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)



#Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#print(y_train.value_counts())
#print(y_test.value_counts())
#print(X_train.head())
#print(y_train.head())
#print(X_test.head())
#print(y_test.head())


# Configuración inicial
#model_logistic = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)  # Asegúrate de especificar un solver adecuado para 'multinomial'
# Aplicar validación cruzada
#scores = cross_val_score(model_logistic, X, y, cv=5, scoring='accuracy')  # cv=5 indica que quieres una validación cruzada de 5 iteraciones

# Entrenar el modelo SVM
model_SVC = SVC(kernel='linear', C=1.0)

# Aplicar validación cruzada
scores = cross_val_score(model_SVC, X, y, cv=5, scoring='accuracy')  # cv=5 indica una validación cruzada de 5 pliegues

# Imprimir los resultados
#print(f"Scores de precisión de cada iteración: {scores}")
#print(f"Precisión media: {scores.mean()} con una desviación estándar de {scores.std()}")

