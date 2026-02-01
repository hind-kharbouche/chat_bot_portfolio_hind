"""
Outil permettant à l'agent IA de chercher des informations dans le portfolio (Upstash Vector).
Cet outil est utilisé par l'agent quand il a besoin de faits concrets sur moi.
"""
import logging
from typing import Optional
from agents import function_tool
from upstash_vector import Index
from config import Config
from functools import lru_cache

# Configuration des logs
logger = logging.getLogger(__name__)

# Le décorateur @lru_cache permet de garder en mémoire les résultats des recherches fréquentes
# pour éviter de refaire des appels inutiles à Upstash (gain de temps et d'argent)
@lru_cache(maxsize=Config.CACHE_SIZE)
def _perform_search(query: str, top_k: int) -> str:
    """
    Fonction interne qui effectue la recherche réelle dans la base de données Upstash.
    """
    url = Config.get_upstash_url()
    token = Config.get_upstash_token()
    
    # Vérification que les clés de connexion sont bien présentes
    if not url or not token:
        logger.error("Configuration Upstash manquante (URL ou Token).")
        return "Erreur : La configuration Upstash est manquante dans le fichier .env."

    try:
        # 1. Connexion à l'index vectoriel
        index = Index(url=url, token=token)
        logger.info(f"Recherche dans le portfolio pour : {query[:50]}...")
        
        # 2. On envoie la question de l'utilisateur à Upstash
        # Upstash transforme la question en vecteur et cherche les morceaux de texte proches
        results = index.query(
            data=query, 
            top_k=top_k, 
            include_metadata=True, # On veut les infos sur la source (nom du fichier)
            include_data=True      # On veut le contenu texte réel
        )
        
        if not results:
            logger.warning(f"Aucun résultat trouvé pour : {query[:50]}")
            return "Désolé, je n'ai pas trouvé d'informations pertinentes dans mon dossier data."
            
        # 3. On formate les résultats pour que l'IA puisse les lire facilement
        formatted_results = []
        for res in results:
            content = getattr(res, 'data', '') or ''
            metadata = getattr(res, 'metadata', {}) or {}
            title = metadata.get('title', 'Sans titre')
            source = metadata.get('source', 'Inconnue')
            
            formatted_results.append(
                f"Source: {source} (Section: {title})\nContenu:\n{content}\n"
            )
        
        logger.info(f"Trouvé {len(results)} résultats.")
        return "\n---\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche Upstash : {str(e)}", exc_info=True)
        return f"Une erreur technique est survenue lors de la recherche : {str(e)}"

# Le décorateur @function_tool déclare cette fonction comme un outil utilisable par l'IA
@function_tool
def search_portfolio(query: str) -> str:
    """
    Recherche des informations dans le portfolio sur le parcours de Hind Kharbouche.
    Utilise cet outil dès que l'utilisateur pose une question sur les expériences, 
    les projets, les compétences ou la formation.

    Args:
        query: La question de l'utilisateur ou les mots-clés à chercher.
    """
    return _perform_search(query, Config.TOP_K_RESULTS)
