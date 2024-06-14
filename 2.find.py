from deepface import DeepFace
import json
from tqdm import tqdm
import time
import threading
import pandas as pd

# Fonction pour suivre l'avancement de la recherche DeepFace
def find_with_progress(img_path, db_path):
    start_time = time.time()  # Temps de début de l'exécution
    result = None

    def deepface_thread():
        nonlocal result
        result = DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            enforce_detection=False,  # Pour éviter la détection de visage (optionnel)
            model_name="VGG-Face",
            distance_metric="cosine"
        )

    # Lancer le thread
    thread = threading.Thread(target=deepface_thread)
    thread.start()

    # Afficher la progression
    with tqdm(desc="DeepFace Search", unit="image") as progress_bar:
        while thread.is_alive():
            elapsed_time = time.time() - start_time
            progress_bar.set_postfix({"Temps écoulé": f"{elapsed_time:.2f} secondes"})
            time.sleep(1)

    # Attendre que le thread se termine
    thread.join()

    return result

# Chemins de l'image et de la base de données
img_path = "toFind/moi.png"
db_path = "images"

# Appel de la fonction avec la barre de progression
result = find_with_progress(img_path, db_path)

# Imprimer le résultat
print(type(result))
print(result)