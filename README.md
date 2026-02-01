# RAPPORT DE PROJET

**Auteur** : Hind KHARBOUCHE  
**Formation** : BUT 3 Science des Données - IUT de Niort  
**Année universitaire** : 2025-2026  
**Projet** : Chatbot Portfolio basé sur RAG (Retrieval Augmented Generation)

---

## 🚀 Application en ligne

**[▶️ Accéder au chatbot portfolio](https://chatbotportfoliohind-rvggbq3tabgkbgp6kkhtsw.streamlit.app)**

Posez vos questions sur mon parcours, mes compétences et mes projets directement à l'IA !

---  


## SOMMAIRE

1. Introduction et Contexte
2. Analyse et Conception
3. Implémentation Technique
4. Conclusion et Perspectives

# 1. INTRODUCTION ET CONTEXTE

## 1.1 Contexte du projet

Dans le cadre du module LLM du BUT 3 Science des Données, nous avons été amenés à développer un portfolio professionnel innovant. Contrairement aux portfolios statiques traditionnels qui se limitent à présenter des informations de manière passive, ce projet vise à créer une **expérience interactive** permettant aux visiteurs de dialoguer naturellement avec un agent IA qui connaît mon parcours professionnel.

## 1.2 Problématique

Comment se démarquer dans un contexte où la création de portfolios est devenue accessible à tous ? La réponse réside dans l'**interactivité** et l'**intelligence artificielle**. Plutôt que de naviguer manuellement entre différentes sections, les visiteurs peuvent simplement poser des questions en langage naturel comme :
- "Quelles sont tes compétences en Python ?"
- "Parle-moi de ton expérience à La Banque Postale"
- "Quels projets as-tu réalisés en data visualisation ?"

## 1.3 Objectifs du projet

### Objectifs fonctionnels
- Créer un chatbot capable de répondre précisément aux questions sur mon parcours
- Garantir des réponses factuelles basées uniquement sur mes données réelles (pas d'hallucinations)
- Offrir une interface utilisateur simple et intuitive
- Déployer l'application en ligne pour un accès public

### Objectifs techniques
- Implémenter une architecture RAG (Retrieval Augmented Generation)
- Utiliser une base de données vectorielle pour la recherche sémantique
- Respecter les bonnes pratiques de développement (code propre, sécurité, documentation)
- Assurer la modularité et la maintenabilité du code

---

# 2. ANALYSE ET CONCEPTION

## 2.1 Choix technologiques

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| **Modèle LLM** | GPT-4.1-nano (OpenAI) | Modèle léger, rapide et économique, parfaitement adapté pour un chatbot conversationnel |
| **Base vectorielle** | Upstash Vector | Solution cloud gratuite avec recherche hybride (dense + sparse BM25), pas de gestion d'infrastructure |
| **Embedding** | BAAI/bge-m3 | Modèle multilingue performant, optimisé pour le français et l'anglais |
| **Framework Agent** | openai-agents | Bibliothèque officielle OpenAI pour créer des agents avec tools, documentation complète |
| **Interface web** | Streamlit | Framework Python simple permettant un développement rapide d'applications web interactives |
| **Langage** | Python 3.13 | Écosystème riche en IA/ML, compatibilité avec toutes les bibliothèques utilisées |

## 2.2 Architecture RAG

L'architecture RAG (Retrieval Augmented Generation) se décompose en deux phases distinctes :

### Phase 1 : Indexation (offline)

Cette phase est exécutée une seule fois lors de la préparation des données :

```
┌─────────────────────────┐
│ Fichiers Markdown (.md) │
│ - formation.md          │
│ - Experience.md         │
│ - Projet.md             │
│ - Activite.md           │
│ - Competences.md        │
│ - Profil.md             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Découpage en chunks     │
│ (par titres #, ##)      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Vectorisation           │
│ (BAAI/bge-m3)          │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Stockage Upstash Vector │
│ + métadonnées           │
└─────────────────────────┘
```

**Stratégie de chunking** : Le découpage se fait intelligemment à chaque titre Markdown (`#` ou `##`), ce qui permet de conserver la cohérence sémantique de chaque section. Chaque chunk est enrichi de métadonnées (nom du fichier source, index, titre de la section).

### Phase 2 : Récupération et Génération (online)

Cette phase s'exécute à chaque question de l'utilisateur :

```
┌─────────────────────────┐
│ Question utilisateur    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Recherche sémantique    │
│ (similarité cosinus)    │
│ top_k = 5               │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Récupération chunks     │
│ pertinents + métadonnées│
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Construction du prompt  │
│ (contexte + question)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Génération GPT-4.1-nano │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Affichage réponse       │
│ (interface Streamlit)   │
└─────────────────────────┘
```

## 2.3 Structure des données

Les données du portfolio sont organisées en 6 fichiers Markdown thématiques :

1. **formation.md** : Parcours académique (BUT Science des Données, Baccalauréat)
2. **Experience.md** : Expériences professionnelles (alternance La Banque Postale, projets)
3. **Projet.md** : 13 projets réalisés (RGPD, DataViz, BDR, enquêtes, etc.)
4. **Activite.md** : Activités personnelles (sport, calcul mental, colonies)
5. **Competences.md** : Compétences techniques (Python, R, SQL, PowerBI, etc.) et soft skills
6. **Profil.md** : Informations personnelles et présentation générale

Cette organisation permet une maintenance facile et une évolutivité du contenu.

---

# 3. IMPLÉMENTATION TECHNIQUE

## 3.1 Architecture du code

```
projet-iut-potfolio/
├── data/                    # Données du portfolio 
│   ├── formation.md
│   ├── Experience.md
│   ├── Projet.md
│   ├── Activite.md
│   ├── Competences.md
│   └── Profil.md
├── agent.py                 # Configuration de l'agent IA
├── agent_tool.py            # Outil de recherche vectorielle
├── indexation.py            # Script d'indexation
├── chunker.py               # Découpage des documents Markdown
├── loader.py                # Chargement des fichiers Markdown
├── history_manager.py       # Gestion de l'historique avec Redis
├── config.py                # Configuration centralisée
├── app.py                   # Interface Streamlit
├── requirements.txt         # Dépendances
├── .env                     # Variables d'environnement
└── .env.example             # Template de configuration
```

## 3.2 Modules principaux

### Module 1 : `indexation.py`

**Rôle** : Indexer les fichiers Markdown dans la base vectorielle Upstash.

**Fonctionnalités clés** :
- Lecture automatique de tous les fichiers `.md` du dossier `data/`
- Découpage intelligent par titres avec la fonction `chunk_markdown()`
- Création de vecteurs avec métadonnées enrichies
- Envoi en batch vers Upstash Vector

**Extrait de code commenté** :

```python
def chunk_markdown(text: str) -> List[str]:
    """
    Découpe le texte Markdown en chunks à chaque titre (# ou ##).
    
    Args:
        text: Le contenu Markdown à découper
        
    Returns:
        Une liste de chunks (morceaux de texte)
    """
    chunks = []
    current_chunk = ""
    
    for line in text.splitlines():
        # Détection d'un nouveau titre
        if line.lstrip().startswith("#"):
            # Sauvegarde du chunk précédent
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Début d'un nouveau chunk
            current_chunk = line + "\n"
        else:
            # Ajout de la ligne au chunk courant
            current_chunk += line + "\n"
    
    # Ajout du dernier chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
```

**Bonnes pratiques appliquées** :
-  Type hints pour la clarté du code
-  Docstring complète au format Google
-  Gestion des cas limites (chunks vides)
-  Utilisation de variables d'environnement pour les credentials

### Module 2 : `agent_tool.py`

**Rôle** : Fournir une fonction de recherche vectorielle utilisable par l'agent.

**Fonctionnalités clés** :
- Décorateur `@function_tool` pour intégration avec openai-agents
- Connexion sécurisée à Upstash Vector via variables d'environnement
- Recherche sémantique avec `top_k=5` résultats
- Formatage des résultats avec métadonnées (source, titre)

**Extrait de code** :

```python
@function_tool
def search_portfolio(query: str) -> str:
    """
    Search the portfolio for relevant information using semantic search.
    
    Args:
        query: The user's question or search term.
        
    Returns:
        A string containing relevant chunks of information.
    """
    # Récupération sécurisée des credentials
    url = os.getenv("UPSTASH_VECTOR_REST_URL")
    token = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
    
    if not url or not token:
        return "Error: Upstash configuration missing."
    
    try:
        index = Index(url=url, token=token)
        
        # Recherche sémantique
        results = index.query(
            data=query, 
            top_k=5,
            include_metadata=True,
            include_data=True
        )
        
        # Formatage des résultats
        formatted_results = []
        for res in results:
            content = getattr(res, 'data', '') or ''
            metadata = getattr(res, 'metadata', {}) or {}
            title = metadata.get('title', 'Untitled')
            source = metadata.get('source', 'Unknown')
            
            formatted_results.append(
                f"Source: {source} (Section: {title})\n"
                f"Content:\n{content}\n"
            )
        
        return "\n---\n".join(formatted_results)
        
    except Exception as e:
        return f"Error occurred during search: {str(e)}"
```

**Points de sécurité** :
-  **AVANT** : Credentials hardcodés dans le code
-  **APRÈS** : Utilisation de `os.getenv()` avec fichier `.env`

### Module 3 : `agent.py`

**Rôle** : Configurer l'agent conversationnel avec ses instructions et outils.

**Configuration de l'agent** :

```python
portfolio_agent = Agent(
    name="Portfolio Assistant",
    instructions="""Tu es Hind Kharbouche. Réponds aux questions 
    à la première personne (je, mon, ma, mes).
    Réponds aux questions sur ton expérience, tes projets, 
    ta formation et tes compétences en te basant UNIQUEMENT 
    sur le contexte fourni.
    
    Tu as accès à un outil `search_portfolio` qui cherche 
    des informations pertinentes dans ta base de données.
    Utilise TOUJOURS cet outil quand on te pose des questions 
    sur ton parcours. N'invente rien.
    Si les résultats de recherche ne contiennent pas la réponse, 
    dis poliment que tu ne sais pas.
    
    Sois professionnelle, concise et utile. 
    Réponds toujours en français à la première personne.""",
    model="gpt-4.1-nano",
    tools=[search_portfolio],
    model_settings=ModelSettings(temperature=0.7)
)
```

**Choix de design** :
- **Première personne** : Expérience plus naturelle et personnelle
- **Instructions strictes** : Éviter les hallucinations en forçant l'utilisation du tool

### Module 4 : `app.py`

**Rôle** : Interface utilisateur web avec Streamlit.

**Fonctionnalités** :
- Chat interactif avec historique des messages
- Gestion d'état avec `st.session_state`
- Affichage des erreurs de manière user-friendly
- Design minimaliste et professionnel

**Extrait de code** :

```python
# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("Votre question..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    
    with st.chat_message("assistant"):
        with st.spinner("..."):
            try:
                result = Runner.run_sync(portfolio_agent, prompt)
                response = result.final_output
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_message = f"Erreur : {str(e)}"
                st.error(error_message)
```

## 3.3 Respect des bonnes pratiques

### Qualité du code

| Critère | Implémentation | Exemple |
|---------|----------------|---------|
| **Docstrings** | Tous les modules et fonctions documentés | Format Google avec Args/Returns |
| **Type hints** | Typage systématique | `def func(x: str) -> List[str]:` |
| **Modularité** | Séparation des responsabilités | 4 modules distincts |
| **Sécurité** | Variables d'environnement | `.env` + `.gitignore` |
| **Gestion d'erreurs** | Try/except avec messages clairs | Affichage user-friendly |

### Exemple de code bien typé

```python
from typing import List

def chunk_markdown(text: str) -> List[str]:
    """Découpe le texte Markdown en chunks."""
    chunks: List[str] = []
    current_chunk: str = ""
    # ... reste du code
    return chunks
```


# 4. CONCLUSION ET PERSPECTIVES

## 4.1 Objectifs atteints

 **Chatbot fonctionnel** : Architecture RAG complète et opérationnelle  
 **Interface intuitive** : Application Streamlit simple et épurée  
 **Code de qualité** : Respect des bonnes pratiques (docstrings, typage, modularité)  
 **Données complètes** : 6 fichiers Markdown couvrant tout le parcours professionnel  
 **Sécurité** : Pas de credentials hardcodés, utilisation de variables d'environnement  
 **Historique persistant** : Sauvegarde des conversations avec Upstash Redis (bonus réalisé)  
 **Performance optimisée** : Mise en cache des requêtes et affichage du temps de réponse  

## 4.2 Compétences mobilisées

### Compétences techniques
- **Développement Python** : POO, typage, gestion d'erreurs, async
- **Intelligence Artificielle** : LLM, embeddings, RAG, agents IA
- **Bases de données** : Bases vectorielles, recherche sémantique
- **Développement web** : Streamlit, interfaces utilisateur
- **DevOps** : Variables d'environnement, déploiement cloud

### Compétences transversales
- **Analyse** : Compréhension des besoins et conception d'architecture
- **Documentation** : Rédaction de code lisible et documenté
- **Rigueur** : Tests et validation systématiques
- **Autonomie** : Recherche de solutions et résolution de problèmes

## 4.3 Difficultés rencontrées et solutions

| Difficulté | Solution apportée |
|------------|-------------------|
| Chunking optimal des documents | Découpage par titres Markdown pour cohérence sémantique |
| Hallucinations du LLM | Instructions strictes + température 0.7 + validation du contexte |
| Gestion des credentials | Migration vers variables d'environnement avec `.env` |
| Interface trop complexe | Simplification radicale : suppression sidebar et emojis |

## 4.4 Améliorations de l'expérience utilisateur

**Améliorations de l'expérience utilisateur** : Pour optimiser l'accessibilité et l'engagement des visiteurs, le chatbot a été enrichi de fonctionnalités majeures :

1. **Questions suggérées cliquables** : Affichage de 6 questions prédéfinies au démarrage de l'application, organisées en deux colonnes, permettant une interaction immédiate sans que l'utilisateur ait besoin de réfléchir à quoi demander. Cette fonctionnalité améliore significativement le taux d'engagement initial.

2. **Interface épurée et professionnelle** : Design minimaliste avec titre clair "Hind Kharbouche - Portfolio Assistant", message d'accueil personnalisé, et bouton de réinitialisation pour recommencer une conversation.

3. **Gestion d'erreurs user-friendly** : Affichage de messages d'erreur clairs et professionnels en cas de problème technique, avec suggestion de contacter par email.

Ces améliorations transforment le chatbot d'un simple outil de consultation en une expérience interactive, transparente et professionnelle.

## 4.5 Fonctionnalités bonus implémentées

 **Historique persistant avec Upstash Redis** : Le module `history_manager.py` sauvegarde automatiquement toutes les conversations dans Upstash Redis avec une expiration de 30 jours. Chaque session utilisateur possède un identifiant unique (UUID) permettant de restaurer l'historique en cas de rafraîchissement de la page.

**Implémentation technique** :
```python
class HistoryManager:
    def __init__(self, session_id: str):
        self.session_id = f"chat_history:{session_id}"
        self.redis = Redis(url=Config.get_redis_url(), 
                          token=Config.get_redis_token())
    
    def save_message(self, role: str, content: str, suggestions: list = None):
        message = {
            "role": role,
            "content": content,
            "suggestions": suggestions or [],
            "timestamp": datetime.now().isoformat()
        }
        self.redis.rpush(self.session_id, json.dumps(message))
        self.redis.expire(self.session_id, 30 * 24 * 60 * 60)  # 30 jours
```

 **Mise en cache des requêtes** : Utilisation du décorateur `@st.cache_data` avec TTL de 600 secondes pour éviter de refaire les mêmes appels API et améliorer les performances.

 **Configuration centralisée** : Le module `config.py` centralise toutes les configurations (modèle, température, nombre de résultats, messages suggérés) pour faciliter la maintenance.

## 4.6 Perspectives d'amélioration futures

### Court terme
- **Déploiement** : Mise en ligne sur Streamlit Cloud avec URL publique (en cours)
- **Enrichissement** : Ajout de projets récents et mise à jour continue
- **Tests utilisateurs** : Collecte de feedback pour améliorer les réponses

### Moyen terme
- **Nouveaux tools** : Envoi d'email, téléchargement CV, génération de recommandations
- **Analytics** : Suivi des questions les plus posées et des langues utilisées pour optimiser le contenu
- **Feedback utilisateur** : Boutons 👍/👎 pour évaluer la qualité des réponses

### Long terme
- **Voice interface** : Intégration de la reconnaissance vocale multilingue
- **Personnalisation** : Adaptation du ton selon le profil du visiteur (recruteur, étudiant, etc.)
- **Extension linguistique** : Optimisation des instructions pour davantage de langues spécifiques

## 4.7 Résultats et Validation

### Métriques du système

| Métrique | Valeur | Description |
|----------|--------|-------------|
| **Fichiers sources** | 6 fichiers Markdown | formation.md, Experience.md, Projet.md, Activite.md, Competences.md, Profil.md |
| **Chunks indexés** | 32 morceaux | Découpage intelligent par titres Markdown |
| **Base vectorielle** | Upstash Vector | Recherche hybride (dense + sparse BM25) |
| **Modèle embedding** | BAAI/bge-m3 | Modèle multilingue performant |
| **Top-k résultats** | 5 chunks | Nombre de résultats retournés par recherche |
| **Température LLM** | 0.4 | Équilibre entre créativité et précision |

### Tests réalisés

**Test de connexion Upstash Vector** : Validation de l'indexation et de la recherche sémantique  
**Test de l'agent IA** : Vérification des réponses avec 20 questions types  
**Test d'interface** : Navigation, boutons, gestion d'erreurs  
**Test de performance** : Temps de réponse < 5 secondes en moyenne  
**Test de sécurité** : Vérification que les credentials ne sont pas exposés  

## 4.8 Déploiement

### Préparation au déploiement

Le projet est configuré pour un déploiement sur **Streamlit Cloud** :

1. **Sécurité** : Le fichier `.gitignore` exclut correctement le fichier `.env` pour éviter l'exposition des clés API
2. **Template de configuration** : Le fichier `.env.example` documente toutes les variables d'environnement nécessaires
3. **Dépendances** : Le fichier `requirements.txt` liste toutes les bibliothèques avec leurs versions exactes


## 4.9 Conclusion générale


Ce projet m'a permis de mettre en pratique les connaissances acquises en IA générative et en développement Python, tout en créant un outil concret et utile pour ma recherche d'emploi. L'architecture RAG garantit des réponses factuelles et pertinentes, tandis que l'interface Streamlit offre une expérience utilisateur fluide et moderne.

Au-delà de l'aspect technique, ce portfolio interactif illustre ma capacité à mener un projet de bout en bout : de l'analyse des besoins à la mise en production, en passant par la conception, le développement et les tests. C'est également une vitrine de mes compétences en data science, développement et IA, domaines dans lesquels je souhaite poursuivre ma carrière.
