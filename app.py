import streamlit as st
import logging
from config import Config
from history_manager import HistoryManager
import time
from agent import run_agent


# Configuration du logging (pour voir les erreurs ou infos dans la console)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  CACHE DES REQUÊTES 
@st.cache_data(ttl=600, show_spinner=False)
def cached_run_agent(question):
    """Exécute l'agent avec mise en cache pour éviter de refaire les mêmes requêtes."""
    start_time = time.time()
    result = run_agent(question)
    end_time = time.time()
    # On renvoie seulement le texte pour éviter les problèmes de sérialisation du cache
    return result.final_output, end_time - start_time



# CONFIGURATION DE LA PAGE 
# On définit le titre de l'onglet et l'icône
st.set_page_config(
    page_title="Hind Kharbouche - Portfolio Assistant",
    page_icon="✨",
    layout="wide"
)


# ETAT DE LA SESSION (SESSION STATE) 
# Streamlit s'exécute de haut en bas à chaque clic. On utilise session_state pour garder les infos.
if "messages" not in st.session_state:
    st.session_state.messages = [] # Historique des messages affichés
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4()) # ID unique pour l'historique Redis

history_manager = HistoryManager(st.session_state.session_id)

# FONCTIONS DE RAPPEL (CALLBACKS) 
def reset_chat_callback():
    """Efface tout l'historique de la discussion."""
    st.session_state.messages = []
    history_manager.clear_history()
    st.session_state.current_user_input = ""

def reset_text_callback():
    """Efface seulement le texte que l'utilisateur est en train d'écrire."""
    st.session_state.current_user_input = ""

def send_message_callback():
    """Gère l'envoi du message quand on appuie sur le bouton ou Entrée."""
    user_input = st.session_state.current_user_input
    if user_input.strip():
        # On ajoute le message de l'utilisateur à l'historique
        st.session_state.messages.append({"role": "user", "content": user_input})
        history_manager.save_message("user", user_input)
        # On vide la zone de texte
        st.session_state.current_user_input = ""

# EN-TÊTE (HEADER) 
col_logo, col_actions = st.columns([1, 1])
with col_logo:
    st.markdown("✨ **Assistant Portfolio**")

with col_actions:
    sub_col1, sub_col2  = st.columns([2, 1])
    with sub_col2:
        st.button("Réinitialiser", type="secondary", use_container_width=True, on_click=reset_chat_callback)

# MESSAGE D'ACCUEIL 
if not st.session_state.messages:
    st.markdown("# Bonjour !")
    st.markdown("# Posez-moi une question sur mon parcours.")
    
    with st.container(border=True):
        st.markdown("Une question sur mes expériences ? Mes projets ou mes compétences ? Écrivez-moi ci-dessous.")

    st.write("") 
    st.caption("✨ Quelques idées pour commencer :")
    
    # Boutons de suggestions rapides
    cols = st.columns(2)
    suggestions = Config.SUGGESTED_QUESTIONS_FR[:4]
    for i, q in enumerate(suggestions):
        if cols[i % 2].button(q, key=f"init_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            history_manager.save_message("user", q)
            st.rerun()




#    AFFICHAGE DE LA DISCUSSION 
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Affichage des suggestions de boutons SI c'est le dernier message de l'assistant
        if msg["role"] == "assistant" and msg.get("suggestions") and i == len(st.session_state.messages) - 1:
            st.caption("🔍 Sujets suggérés :")
            sug_cols = st.columns(len(msg["suggestions"]))
            for j, sug in enumerate(msg["suggestions"]):
                if sug_cols[j].button(sug, key=f"sug_{i}_{j}"):
                    new_content = f"Parle-moi de tes projets en {sug}"
                    st.session_state.messages.append({"role": "user", "content": new_content})
                    history_manager.save_message("user", new_content)
                    st.rerun()

# GÉNÉRATION DE LA RÉPONSE IA 
# On vérifie si le dernier message vient de l'utilisateur
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            try:
                # On lance l'agent (cherche dans les données + génère le texte)
                # Exécution de l'agent avec calcul du temps
                full_response, duration = cached_run_agent(last_user_prompt)
                
                # Gestion des suggestions à la fin du texte de l'IA
                suggestions_list = []
                response_text = full_response
                if Config.SUGGESTIONS_TAG in full_response:
                    parts = full_response.split(Config.SUGGESTIONS_TAG)
                    response_text = parts[0].strip()
                    if len(parts) > 1:
                        suggestions_list = [s.strip() for s in parts[1].split(",") if s.strip()]
                
                # On affiche la réponse et on l'enregistre
                st.markdown(response_text)
                st.caption(f"⏱️ {duration:.2f}s")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "suggestions": suggestions_list,
                    "duration": duration
                })
                history_manager.save_message("assistant", response_text, suggestions_list)
                st.rerun() # On recharge pour mettre à jour l'affichage
                
            except Exception as e:
                logger.error(f"Erreur IA : {e}")
                st.error(f"Désolé, une erreur technique est survenue. Vérifiez vos clés API ou votre connexion.")
                st.session_state.messages.append({"role": "assistant", "content": "Erreur technique."})

# ZONE DE SAISIE (INPUT AREA) 
with st.container(border=True):
    st.text_area(
        "Votre message", 
        max_chars=400, 
        label_visibility="collapsed", 
        placeholder="Écrivez votre question ici...",
        key="current_user_input"
    )
    
    c_reset, c_spacer, c_send = st.columns([1, 2, 0.3])
    with c_reset:
        st.button("Vider le texte", type="secondary", on_click=reset_text_callback)
    with c_send:
        st.button("↑", type="primary", use_container_width=True, on_click=send_message_callback)

# PIED DE PAGE (FOOTER) 
st.divider()
