import os
from flask import Flask, request, jsonify
from modelos.Modelo import ModeloClasificador
from modelos.limpieza_unique import LimpiadorCV
import PyPDF2
from app import create_app
from werkzeug.utils import secure_filename
import numpy as np
from flask_cors import CORS

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
