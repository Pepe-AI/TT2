import pandas as pd
import matplotlib.pyplot as plt

#Cargar dataset
df = pd.read_csv('Curriculum Vitae.csv', encoding="utf-8")
print(df.shape)

df.info()

#vemos las etiquetas
valores_unicos = df['Category'].unique()
print(valores_unicos)

print(df.shape)
print(df.columns)
print(df.Category.value_counts())


#vemos la distribucion de las clases entre sí
label_value_counts = df.Category.value_counts()
print(label_value_counts/label_value_counts.sum())

plt.title('Distribución de clases')
label_value_counts.plot.bar()
plt.show()