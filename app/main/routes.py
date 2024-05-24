from flask import Blueprint, request, jsonify, current_app as app
from werkzeug.utils import secure_filename
from .utils import allowed_file, clasificar_cv, categoria_valida
from modelos.limpieza_unique import LimpiadorCV

bp = Blueprint('main', __name__)

@bp.route('/clasificar', methods=['POST'])
def clasificar():
    print('Categoría recibida:', request.form['categoria'])

    for file in request.files.getlist('file'):
        print('Archivo recibido:', file.filename)
    print("--------------------------")    

    if 'file' not in request.files:
        return jsonify({'error': 'No se ha enviado ningún archivo.'}), 400

    uploaded_files = request.files.getlist('file')
    categoria = request.form.get('categoria')
    if not categoria:
        return jsonify({'error': 'No se seleccionó ninguna categoría. Favor de seleccionar una categoría.'}), 400

    limpiador = LimpiadorCV("diccionarios")

    if not categoria_valida(categoria):
        return jsonify({'error': f'Categoría "{categoria}" no válida. Favor de seleccionar una categoría existente.'}), 400

    resultados = []
    archivos_con_categoria = []

    for uploaded_file in uploaded_files:
        if uploaded_file and allowed_file(uploaded_file.filename):
            categoria_predicha = clasificar_cv(uploaded_file, categoria, limpiador)
            resultados.append({'nombre_archivo': uploaded_file.filename, 'categoria_predicha': categoria_predicha})
            if categoria_predicha == categoria:
                archivos_con_categoria.append(uploaded_file.filename)
        else:
            resultados.append({'error': 'Tipo de archivo no permitido o no se proporcionó ningún archivo.'})

    resultados.append({'archivos_con_categoria': archivos_con_categoria})

    return jsonify(resultados)