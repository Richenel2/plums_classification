from sklearn.ensemble import RandomForestClassifier
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

# Créer le modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
rf_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = rf_model.predict(X_test)

# Évaluer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f'Précision du modèle Random Forest : {accuracy * 100:.2f}%')

# Sauvegarder le modèle
import joblib
joblib.dump(rf_model, 'model/random_forest_model.joblib')
print("Modèle Random Forest sauvegardé sous : random_forest_model.joblib")
