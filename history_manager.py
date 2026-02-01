
import json
from datetime import datetime
from upstash_redis import Redis
from config import Config

class HistoryManager:
    """Gère la persistance de l'historique des conversations avec Upstash Redis."""
    
    def __init__(self, session_id: str):
        self.session_id = f"chat_history:{session_id}"
        url = Config.get_redis_url()
        token = Config.get_redis_token()
        
        if not url or not token:
            self.redis = None
            print("Warning: Upstash Redis credentials missing. History won't be persisted.")
        else:
            self.redis = Redis(url=url, token=token)

    def save_message(self, role: str, content: str, suggestions: list = None):
        """Sauvegarde un message dans Redis."""
        if not self.redis:
            return
            
        message = {
            "role": role,
            "content": content,
            "suggestions": suggestions or [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Ajouter à la liste Redis (RPUSH)
        try:
            self.redis.rpush(self.session_id, json.dumps(message))
            # Optionnel : Définir une expiration (ex: 30 jours)
            self.redis.expire(self.session_id, 30 * 24 * 60 * 60)
        except Exception as e:
            print(f"Error saving message to Redis: {e}")

    def load_history(self):
        """Charge l'historique depuis Redis."""
        if not self.redis:
            return []
            
        try:
            # Récupérer tous les éléments de la liste
            items = self.redis.lrange(self.session_id, 0, -1)
            return [json.loads(item) for item in items]
        except Exception as e:
            print(f"Error loading history from Redis: {e}")
            return []

    def clear_history(self):
        """Supprime l'historique de Redis."""
        if not self.redis:
            return
            
        try:
            self.redis.delete(self.session_id)
        except Exception as e:
            print(f"Error clearing history from Redis: {e}")
