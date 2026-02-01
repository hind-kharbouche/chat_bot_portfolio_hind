from typing import List

def chunk_markdown(text: str) -> List[str]:
    """
    Cette fonction découpe un grand texte Markdown en plusieurs petits morceaux (chunks).
    On découpe dès qu'on voit un titre de niveau 1 (#) ou 2 (##).
    """
    chunks = []
    current_chunk = ""
    
    # On parcourt le texte ligne par ligne
    for line in text.splitlines():
        # Si la ligne commence par un # (titre)
        if line.lstrip().startswith("#"):
            # Si on avait déjà du texte en cours, on l'ajoute à notre liste
            if current_chunk:
                chunks.append(current_chunk.strip())
            # On recommence un nouveau morceau avec le titre
            current_chunk = line + "\n"
        else:
            # Sinon, on continue d'ajouter la ligne au morceau actuel
            current_chunk += line + "\n"
            
    # Ne pas oublier d'ajouter le tout dernier morceau
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks
