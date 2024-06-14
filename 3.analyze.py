from deepface import DeepFace
import json
from tqdm import tqdm
import time
import threading


# Fonction pour suivre l'avancement de la vérification DeepFace
def analyze_with_progress(img_path):
    start_time = time.time()  # Temps de début de l'exécution
    result = None

    def deepface_thread():
        nonlocal result
        result = DeepFace.analyze(
            img_path = img_path ,
            actions = ['age', 'gender', 'race', 'emotion'],
        )

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

# Chemin de l'image
img_path = "images/IMG_9846.jpg"

# Appel de la fonction avec la barre de progression
result = analyze_with_progress(img_path)

# Imprimer le dictionnaire au format JSON
print(json.dumps(result, indent=4))
