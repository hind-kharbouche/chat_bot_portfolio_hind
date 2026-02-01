"""
Agent IA pour le portfolio interactif.
Ce fichier contient la logique principale de l'intelligence artificielle.
"""
import logging
import os
import openai
from agents import Agent, Runner, ModelSettings
from agent_tool import search_portfolio
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config

# Configuration des logs pour voir ce qui se passe dans la console
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Tentative de configuration du timeout global pour OpenAI (si supporté par la lib interne)
os.environ["OPENAI_TIMEOUT"] = "60"
try:
    openai.timeout = 60.0
except:
    pass


def get_instructions() -> str:
    """
    Génère les consignes (système) que l'IA doit suivre.
    """
    email = Config.EMAIL
    tag = Config.SUGGESTIONS_TAG
    
    return f"""Tu es Hind Kharbouche. Réponds aux questions à la première personne (je, mon, ma, mes).
    Ton rôle est d'informer les recruteurs et les professionnels sur ton parcours de manière directe et professionnelle.
    
    RÈGLES CRITIQUES :
    1. Tu ne dois t'appuyer QUE sur les informations extraites par l'outil `search_portfolio`.
    2. N'utilise JAMAIS tes connaissances internes pour répondre à une question sur mon parcours, mes compétences ou mes expériences.
    3. Si l'information n'est pas PRÉSENTE dans les résultats de `search_portfolio` :
       - Essaie d'abord de répondre avec ce que tu as trouvé qui est pertinent
       - Si vraiment aucune information pertinente n'est disponible, SEULEMENT dans ce cas, indique que tu ne possèdes pas cette information précise et propose de me contacter par email : {email}
    4. Toute réponse doit être factuelle et basée uniquement sur les documents fournis.
    
    TON ET PERSONNALITÉ :
    - INTERDICTION de commencer tes phrases par des expressions impersonnelles comme "D'après les informations disponibles", "Selon les documents", "D'après mon portfolio" ou "Les résultats de recherche indiquent".
    - Parle directement : au lieu de dire "D'après mon portfolio, j'ai une expérience de...", dis "J'ai une expérience de...".
    - Adopte un ton professionnel, dynamique et rassurant.
    
    Règles de formatage :
    - Utilise TOUJOURS `search_portfolio` pour toute question sur mon parcours.
    - Si on te demande de présenter tes projets, regroupe-les par catégories et termine ton message UNIQUEMENT par cette balise :
    {tag} Programmation, Statistiques, Data Visualization, VBA & Automatisation
    
    Réponds uniquement en français."""

# La décoration @retry permet de relancer l'appel si l'IA rencontre un problème réseau temporaire
@retry(
    stop=stop_after_attempt(Config.MAX_RETRIES),
    wait=wait_exponential(min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT)
)
def run_agent(user_input: str):
    """
    Lance l'agent IA.
    """
    try:
        # On configure l'Agent avec ses outils et ses instructions
        agent = Agent(
            name="PortfolioAssistant",
            instructions=get_instructions(),
            model=Config.MODEL_NAME,
            tools=[search_portfolio], # L'agent a accès à l'outil de recherche dans les données
            model_settings=ModelSettings(temperature=0.4, timeout=60.0) # Température basse pour plus de précision
        )
        
        logger.info("Lancement de l'agent")
        # On exécute l'agent et on retourne le résultat
        return Runner.run_sync(agent, user_input)
        
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise

