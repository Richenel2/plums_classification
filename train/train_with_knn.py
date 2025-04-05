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
model_filename = 'knn_model.joblib'
joblib.dump(knn, model_filename)
print(f"Modèle KNN sauvegardé sous : {model_filename}")

# Charger le modèle depuis le fichier
loaded_knn = joblib.load(model_filename)
print("Modèle chargé avec succès !")

# Exemple de prédiction sur une nouvelle image (en utilisant un embedding exemple)
# Imaginons que nous avons un nouvel embedding à prédire
# Nouveau embedding (ici, un exemple fictif)
new_embedding = np.random.rand(1, X.shape[1])  # Utilisez un vrai embedding ici

# Prédire la catégorie de la nouvelle image avec le modèle chargé
predicted_label = loaded_knn.predict(new_embedding)
print(f'Label prédit pour le nouvel embedding : {predicted_label[0]}')
