import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib  # Utilisé pour sauvegarder et charger le modèle

# Charger le fichier CSV
csv_filename = 'image_embeddings.csv'
df = pd.read_csv(csv_filename)

# Extraire les labels et les embeddings
X = df.drop(columns=['image_path', 'label']).values  # Les embeddings
y = df['label'].values  # Les labels (catégories)

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Choisir k = 5 (vous pouvez ajuster ce paramètre)
knn.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = knn.predict(X_test)

# Évaluer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Précision du modèle KNN : {accuracy * 100:.2f}%')

# Sauvegarder le modèle
model_filename = 'model/knn_model.joblib'
joblib.dump(knn, model_filename)
print(f"Modèle KNN sauvegardé sous : {model_filename}")

