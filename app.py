from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Créer l'application Flask
app = Flask(__name__)

# Configurer le dossier où les images téléchargées seront stockées
app.config['UPLOAD_FOLDER'] = 'static/images/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Charger les modèles pré-entrainés
knn_model = joblib.load('knn_model.joblib')
svm_model = joblib.load('svm_model.joblib')
rf_model = joblib.load('random_forest_model.joblib')
nn_model = load_model('neural_network_model.h5')

# Fonction pour vérifier les extensions d'image autorisées
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fonction pour traiter l'image
def preprocess_image(image):
    image = image.resize((224, 224))  # Redimensionner l'image à la taille attendue par les modèles
    image_array = np.array(image) / 255.0  # Normaliser l'image
    image_array = image_array.reshape(1, -1)  # Aplatir l'image pour les modèles non CNN
    return image_array

# Route pour afficher la page principale
@app.route('/')
def index():
    return render_template('index.html')

# Route API pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Sauvegarder l'image
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Ouvrir et traiter l'image
        image = Image.open(file_path)
        processed_image = preprocess_image(image)
        
        # Choisir le modèle
        model_type = request.form.get('model')
        
        if model_type == 'knn':
            prediction = knn_model.predict(processed_image)
            probability = np.max(knn_model.predict_proba(processed_image))
        elif model_type == 'svm':
            prediction = svm_model.predict(processed_image)
            probability = svm_model.decision_function(processed_image)
            probability = np.max(probability)
        elif model_type == 'random_forest':
            prediction = rf_model.predict(processed_image)
            probability = np.max(rf_model.predict_proba(processed_image))
        elif model_type == 'neural_network':
            prediction = nn_model.predict(processed_image)
            probability = np.max(prediction)
        else:
            return jsonify({'error': 'Invalid model type'})
        
        # Renvoie la catégorie prédite et la probabilité
        category = prediction[0]  # On suppose que prediction est un tableau
        return jsonify({'category': category, 'probability': f'{probability * 100:.2f}%'})
    else:
        return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
