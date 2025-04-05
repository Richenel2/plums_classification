from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Charger les données depuis le fichier CSV
csv_filename = 'image_embeddings.csv'
df = pd.read_csv(csv_filename)

# Extraire les labels et les embeddings
X = df.drop(columns=['image_path', 'label']).values  # Embeddings
y = df['label'].values  # Labels

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle SVM
svm_model = SVC(kernel='linear')  # Utiliser un kernel linéaire (vous pouvez essayer d'autres kernels comme 'rbf')

# Entraîner le modèle
svm_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = svm_model.predict(X_test)

# Évaluer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f'Précision du modèle SVM : {accuracy * 100:.2f}%')

# Sauvegarder le modèle
import joblib
joblib.dump(svm_model, 'model/svm_model.joblib')
print("Modèle SVM sauvegardé sous : svm_model.joblib")
