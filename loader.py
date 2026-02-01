import os

def load_markdown_files(directory_path: str):
    """
    Lit tous les fichiers .md d'un dossier spécifié.
    Renvoie une liste de tuples : (nom_du_fichier, contenu_texte).
    """
    files_data = []
    
    # On vérifie si le dossier existe pour éviter que le code s'arrête brutalement
    if not os.path.exists(directory_path):
        print(f"Le dossier {directory_path} n'existe pas.")
        return files_data
        
    # On liste tous les fichiers qui se terminent par .md
    md_files = [f for f in os.listdir(directory_path) if f.endswith(".md")]
    
    for filename in md_files:
        filepath = os.path.join(directory_path, filename)
        # On ouvre chaque fichier en précisant l'encodage utf-8 pour bien gérer les accents
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            files_data.append((filename, content))
            
    return files_data
