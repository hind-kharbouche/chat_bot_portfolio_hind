"""
Script principal d'indexation des documents Markdown dans Upstash Vector.
Ce script orchestre le chargement, le découpage et l'envoi des données vers Upstash.
"""
import os
from typing import List
from dotenv import load_dotenv

# On charge les variables d'environnement (.env) pour accéder aux clés API
load_dotenv()

# Imports de nos propres modules créés sur mesure
from loader import load_markdown_files
from chunker import chunk_markdown
from upstash_vector import Index, Vector

# Initialisation de la connexion à la base de données Upstash Vector
index = Index(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
)

# Dossier où sont rangés tes fichiers de connaissances (.md)
data_folder = "data"

def run_indexation():
    """
    Fonction principale qui fait tout le travail d'indexation.
    """
    print(f"Lancement de l'indexation des fichiers dans : {data_folder}...")
    
    # ÉTAPE 1 : On charge les fichiers texte depuis le disque dur
    # fruit du travail de 'loader.py'
    files_data = load_markdown_files(data_folder)
    
    vectors = []
    
    # ÉTAPE 2 : On découpe chaque fichier en "chunks" (petits morceaux)
    # fruit du travail de 'chunker.py'
    for filename, content in files_data:
        chunks = chunk_markdown(content)
        for i, chunk in enumerate(chunks):
            # On ne garde que les morceaux qui contiennent du texte
            if chunk.strip():
                # On prépare un objet "Vector" pour Upstash
                vectors.append(
                    Vector(
                        id=f"{filename}-{i}", # ID unique pour chaque morceau
                        data=chunk,           # Le texte réel du morceau
                        metadata={            # Infos bonus pour aider l'IA plus tard
                            "source": filename,
                            "chunk_index": i,
                            "title": chunk.splitlines()[0] if chunk.splitlines() else ""
                        }
                    )
                )
    
    # ÉTAPE 3 : On envoie tout d'un coup à Upstash (plus rapide que un par un)
    if vectors:
        index.upsert(vectors=vectors)
        print(f"Succès : {len(vectors)} morceaux ont été envoyés vers Upstash.")
    else:
        print("Attention : Aucun fichier ou contenu n'a été trouvé à indexer.")

# Si on lance ce fichier directement, on exécute l'indexation
if __name__ == "__main__":
    run_indexation()