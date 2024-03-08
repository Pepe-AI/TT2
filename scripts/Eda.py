import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Cargar dataset
df = pd.read_csv('job_descriptions.csv', encoding="utf-8")

df = df.drop(['latitude', 'longitude', 'Company Size', 'Job Posting Date', 'Preference', 'Contact Person', 'Contact', 'Job Portal', 'Benefits', 'Company Profile'], axis=1)

# Imprimir el número de columnas
print("Número de columnas:", len(df.columns))


# Contar cuántas veces aparece 'México' en la columna 'country'
conteo_mexico = (df['Country'] == 'Mexico').sum()

# Imprimir el resultado
print("Número de veces que aparece México:", conteo_mexico)


# Seleccionar la columna
columna = df["Job Title"]
# Seleccionar la columna
columna2 = df["Role"]
# Contar valores únicos
jobTitle_valores_unicos = columna.unique().shape[0]

# Imprimir el resultado
print(f"Número de valores únicos en la columna 'Job Title': {jobTitle_valores_unicos}")


# Contar valores únicos
role_valores_unicos = columna2.unique().shape[0]

# Imprimir el resultado
print(f"Número de valores únicos en la columna 'Job Title': {role_valores_unicos}")

valores = []

for valor in columna.unique():
    valores.append(valor)

print(valores)