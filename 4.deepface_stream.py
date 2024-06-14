from deepface import DeepFace
import json
from tqdm import tqdm
import time
import threading

from deepface import DeepFace
import json
from tqdm import tqdm
import time
import threading


# Fonction pour suivre l'avancement de la vérification DeepFace
def stream_with_progress(images_path):
    start_time = time.time()  # Temps de début de l'exécution
    result = None

    def deepface_thread():
        nonlocal result
        result = DeepFace.stream(db_path = images_path)

    # Lancer le thread
    thread = threading.Thread(target=deepface_thread)
    thread.start()

    # Afficher la progression
    with tqdm(desc="DeepFace Verification", unit="pair", total=100) as progress_bar:
        while thread.is_alive():
            elapsed_time = time.time() - start_time
            progress = min((elapsed_time / 60) / 10, 1.0)
            progress_bar.update(int(progress * 100) - progress_bar.n)
            progress_bar.set_postfix({"Temps écoulé": f"{elapsed_time:.2f} secondes"})
            time.sleep(1)

    # Attendre que le thread se termine
    thread.join()

    return result

# Chemin des images
images_path = "images"

# Appel de la fonction avec la barre de progression
result = stream_with_progress(images_path)