import os
import PyPDF2
import numpy as np
from flask import current_app as app
from werkzeug.utils import secure_filename
from modelos.Modelo import ModeloClasificador

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clasificar_cv(uploaded_file, categoria, limpiador):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
        uploaded_file.save(file_path)

        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join([page.extract_text() for page in reader.pages if page.extract_text()])

        clean_resume = limpiador.clean_resume(text)
        clean_resume = limpiador.clean_useless_words(clean_resume, categoria)

        clasificador = ModeloClasificador('modelos/svm_model.joblib', 'modelos/tfidf_vectorizer.joblib', 'modelos/selected_features.joblib', 'modelos/pca.joblib', 'modelos/label_encoder.joblib')
        predicted_category = clasificador.predecir(clean_resume)

        if isinstance(predicted_category, np.ndarray):
            predicted_category = predicted_category.tolist()
            if len(predicted_category) == 1:
                predicted_category = predicted_category[0]

        return predicted_category

    except FileNotFoundError:
        return f"Error: Archivo PDF no encontrado."
    except Exception as e:
        return f"Error inesperado: {str(e)}"

def categoria_valida(categoria):
    diccionarios_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'diccionarios')
    archivos_txt = [f[:-4] for f in os.listdir(diccionarios_dir) if f.endswith('.txt')]
    return categoria in archivos_txt