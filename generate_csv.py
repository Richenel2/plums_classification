import os
import csv
import PIL
from PIL import Image
import voyageai
from dotenv import load_dotenv

load_dotenv()

import kagglehub

# Download latest version
path = kagglehub.dataset_download("arnaudfadja/african-plums-quality-and-defect-assessment-data")

# Initialiser le client VoyageAI
vo = voyageai.Client()

# Chemins vers les dossiers de données
base_path = f'{path}/african_plums_dataset/african_plums/'
categories = ['bruised', 'cracked', 'rotten', 'spotted', 'unripe', 'unaffected']

# Liste pour stocker les résultats
data_for_csv = []

# Fonction pour obtenir les embeddings des images
def get_embeddings(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')  # S'assurer que l'image est en RGB
        inputs = [[img]]
        result = vo.multimodal_embed(inputs, model="voyage-multimodal-3")
        return result.embeddings[0]  # Retourne le premier embedding
    except Exception as e:
        print(f"Erreur pour l'image {image_path}: {e}")
        return None

# Parcourir les sous-dossiers pour obtenir les images et leurs labels
for category in categories:
    folder_path = os.path.join(base_path, category)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            if image_path.endswith(('.jpg', '.jpeg', '.png')):  # Vérifie si c'est une image
                embedding = get_embeddings(image_path)
                if embedding is not None:
                    data_for_csv.append([image_path, category] + embedding)

# Enregistrer les résultats dans un fichier CSV
csv_filename = 'image_embeddings.csv'
header = ['image_path', 'label'] + [f'embedding_{i}' for i in range(len(data_for_csv[0]) - 2)]

# Écrire dans le fichier CSV
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Écrire les en-têtes
    writer.writerows(data_for_csv)  # Écrire les données

print(f"Fichier CSV généré avec succès : {csv_filename}")
