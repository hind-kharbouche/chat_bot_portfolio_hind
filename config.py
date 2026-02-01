"""
Configuration centralisée pour le chatbot portfolio.
Toutes les constantes et paramètres configurables sont définis ici.
"""
import os
from typing import List
from dotenv import load_dotenv

# Charger les variables d'environnement dès le début
load_dotenv()

class Config:
    """Configuration principale de l'application."""
    
    # Informations de contact
    EMAIL = "hind.kharbouche@etu.univ-poitiers.fr"
    
    # Tag pour les suggestions de boutons
    SUGGESTIONS_TAG = "[SUGGESTIONS]"
    
    # Paramètres du modèle
    MODEL_NAME = "gpt-4.1-nano"
    MODEL_TEMPERATURE = 0.7
    
    # Paramètres de recherche vectorielle
    TOP_K_RESULTS = 5
    
    # Validation des entrées
    MAX_QUESTION_LENGTH = 500
    MIN_QUESTION_LENGTH = 3
    
    # Questions suggérées
    SUGGESTED_QUESTIONS_FR: List[str] = [
        "Quelles sont tes compétences techniques ?",
        "Parle-moi de ton expérience à La Banque Postale",
        "Quels projets as-tu réalisés ?",
        "Quelle est ta formation ?",
        "Quelles sont tes activités personnelles ?",
        "Quels outils de data visualisation maîtrises-tu ?"
    ]
    
    # Cache
    CACHE_SIZE = 100
    
    # Retry logic
    MAX_RETRIES = 3
    RETRY_MIN_WAIT = 1  # secondes
    RETRY_MAX_WAIT = 10  # secondes
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_upstash_url(cls) -> str:
        """Récupère l'URL Upstash depuis les variables d'environnement."""
        return os.getenv("UPSTASH_VECTOR_REST_URL", "")
    
    @classmethod
    def get_upstash_token(cls) -> str:
        """Récupère le token Upstash depuis les variables d'environnement."""
        return os.getenv("UPSTASH_VECTOR_REST_TOKEN", "")

    @classmethod
    def get_redis_url(cls) -> str:
        """Récupère l'URL Upstash Redis."""
        return os.getenv("UPSTASH_REDIS_REST_URL", "")

    @classmethod
    def get_redis_token(cls) -> str:
        """Récupère le token Upstash Redis."""
        return os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
