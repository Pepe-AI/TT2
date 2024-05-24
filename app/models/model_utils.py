import joblib
import numpy as np

def load_model(model_path):
    """
    Carga un modelo desde el archivo especificado.

    Args:
    model_path (str): Ruta al archivo del modelo.

    Returns:
    object: El modelo cargado.
    """
    return joblib.load(model_path)

def save_model(model, model_path):
    """
    Guarda un modelo en el archivo especificado.

    Args:
    model (object): El modelo a guardar.
    model_path (str): Ruta al archivo donde se guardará el modelo.
    """
    joblib.dump(model, model_path)

def save_components(vectorizer, selector, svd, model, label_encoder):
    """
    Guarda todos los componentes necesarios para la predicción.

    Args:
    vectorizer (object): El vectorizador Tfidf.
    selector (array): Las características seleccionadas.
    svd (object): El modelo SVD.
    model (object): El modelo de clasificación.
    label_encoder (object): El codificador de etiquetas.
    """
    joblib.dump(vectorizer, 'modelos/tfidf_vectorizer.joblib')
    joblib.dump(selector, 'modelos/selected_features.joblib')
    joblib.dump(svd, 'modelos/pca.joblib')
    joblib.dump(model, 'modelos/rf_model.joblib')
    joblib.dump(label_encoder, 'modelos/label_encoder.joblib')

def predict(model, vectorizer, selector, svd, label_encoder, text):
    """
    Realiza una predicción utilizando el modelo cargado y los componentes necesarios.

    Args:
    model (object): El modelo de machine learning.
    vectorizer (object): El vectorizador Tfidf.
    selector (array): Las características seleccionadas.
    svd (object): El modelo SVD.
    label_encoder (object): El codificador de etiquetas.
    text (str): El texto a clasificar.

    Returns:
    str: La categoría predicha.
    """
    text_vectorized = vectorizer.transform([text])
    text_selected = text_vectorized[:, selector]
    text_reduced = svd.transform(text_selected)
    prediction_numeric = model.predict(text_reduced)
    prediction_label = label_encoder.inverse_transform(prediction_numeric)
    return prediction_label[0] if len(prediction_label) == 1 else prediction_label