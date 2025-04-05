import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Charger les données depuis le fichier CSV
csv_filename = 'image_embeddings.csv'
df = pd.read_csv(csv_filename)

# Extraire les labels et les embeddings
X = df.drop(columns=['image_path', 'label']).values  # Embeddings
y = df['label'].values  # Labels

# Encoder les labels en entiers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Normalisation des données (il est toujours bon de normaliser les données avant d'entraîner un modèle de NN)
X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

# Créer le modèle de réseau de neurones avec Keras
model = models.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),  # La taille des embeddings
    layers.Dense(128, activation='relu'),  # Couche cachée avec 128 neurones
    layers.Dense(64, activation='relu'),  # Couche cachée avec 64 neurones
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # Couche de sortie (nombre de classes)
])

# Compiler le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Évaluer le modèle sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Précision du modèle Neural Network : {test_accuracy * 100:.2f}%')

# Sauvegarder le modèle
model.save('model/neural_network_model.h5')
print("Modèle Neural Network sauvegardé sous : neural_network_model.h5")
