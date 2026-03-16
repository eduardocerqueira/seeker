#date: 2026-03-16T17:43:40Z
#url: https://api.github.com/gists/befd3d7c917b9b1c1c82a6930ce45c5a
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: """
      2: Outils utilitaires pour Circuit Breaker SubAgent
      3: """
      4: 
      5: import logging
      6: import time
      7: from typing import Dict, Any, Optional, List
      8: from datetime import datetime, timedelta
      9: import json
     10: 
     11: logger = logging.getLogger(__name__)
     12: 
     13: 
     14: def calculate_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
     15:     """
     16:     Calcule le délai de backoff exponentiel pour les tentatives
     17:     
     18:     Args:
     19:         attempt: Numéro de tentative (1, 2, 3...)
     20:         base_delay: Délai de base en secondes
     21:         max_delay: Délai maximum en secondes
     22:     
     23:     Returns:
     24:         Délai à attendre en secondes
     25:     """
     26:     delay = base_delay * (2 ** (attempt - 1))
     27:     return min(delay, max_delay)
     28: 
     29: 
     30: def format_circuit_state(state: str) -> str:
     31:     """
     32:     Formate un état de circuit pour affichage
     33:     
     34:     Args:
     35:         state: État du circuit (closed, open, half_open, etc.)
     36:     
     37:     Returns:
     38:         État formaté avec emoji
     39:     """
     40:     emojis = {
     41:         "closed": "🟢 FERMÉ",
     42:         "open": "🔴 OUVERT",
     43:         "half_open": "🟡 MI-OUVERT",
     44:         "half-open": "🟡 MI-OUVERT",
     45:         "forced_open": "⚫ FORCÉ",
     46:         "forced-open": "⚫ FORCÉ",
     47:         "disabled": "⚪ DÉSACTIVÉ"
     48:     }
     49:     return emojis.get(state.lower(), f"❓ {state}")
     50: 
     51: 
     52: def analyze_circuit_health(stats: Dict[str, Any]) -> Dict[str, Any]:
     53:     """
     54:     Analyse la santé d'un circuit à partir de ses statistiques
     55:     
     56:     Args:
     57:         stats: Statistiques du circuit
     58:     
     59:     Returns:
     60:         Analyse de santé avec recommandations
     61:     """
     62:     health_score = 100
     63:     issues = []
     64:     recommendations = []
     65:     
     66:     failure_rate = stats.get("failure_rate", 0)
     67:     consecutive_failures = stats.get("consecutive_failures", 0)
     68:     state = stats.get("state", "unknown")
     69:     
     70:     # Évaluer le taux d'échec
     71:     if failure_rate > 50:
     72:         health_score -= 30
     73:         issues.append(f"Taux d'échec critique: {failure_rate}%")
     74:         recommendations.append("Vérifier le service cible immédiatement")
     75:     elif failure_rate > 20:
     76:         health_score -= 15
     77:         issues.append(f"Taux d'échec élevé: {failure_rate}%")
     78:         recommendations.append("Surveiller le service cible")
     79:     
     80:     # Évaluer les échecs consécutifs
     81:     if consecutive_failures > 10:
     82:         health_score -= 25
     83:         issues.append(f"Échecs consécutifs: {consecutive_failures}")
     84:         recommendations.append("Circuit devrait être ouvert")
     85:     elif consecutive_failures > 5:
     86:         health_score -= 10
     87:         issues.append(f"Échecs consécutifs: {consecutive_failures}")
     88:     
     89:     # Évaluer l'état
     90:     if state == "open":
     91:         health_score -= 20
     92:         recommendations.append("Vérifier la récupération du service")
     93:     elif state == "half_open":
     94:         health_score -= 5
     95:         recommendations.append("Tests de reprise en cours")
     96:     
     97:     return {
     98:         "health_score": max(0, health_score),
     99:         "status": "CRITICAL" if health_score < 50 else "WARNING" if health_score < 80 else "HEALTHY",
    100:         "issues": issues,
    101:         "recommendations": recommendations[:3]  # Top 3 recommandations
    102:     }
    103: 
    104: 
    105: def serialize_circuit_data(data: Dict[str, Any]) -> str:
    106:     """
    107:     Sérialise les données d'un circuit pour persistance
    108:     
    109:     Args:
    110:         data: Données du circuit
    111:     
    112:     Returns:
    113:         JSON string
    114:     """
    115:     def json_serializer(obj):
    116:         if isinstance(obj, datetime):
    117:             return obj.isoformat()
    118:         if isinstance(obj, timedelta):
    119:             return obj.total_seconds()
    120:         raise TypeError(f"Type {type(obj)} non sérialisable")
    121:     
    122:     return json.dumps(data, default=json_serializer, indent=2)
    123: 
    124: 
    125: def deserialize_circuit_data(json_str: str) -> Dict[str, Any]:
    126:     """
    127:     Désérialise les données d'un circuit
    128:     
    129:     Args:
    130:         json_str: JSON string
    131:     
    132:     Returns:
    133:         Données du circuit
    134:     """
    135:     return json.loads(json_str)
    136: 
    137: 
    138: def merge_circuit_configs(base_config: Dict, override_config: Dict) -> Dict:
    139:     """
    140:     Fusionne deux configurations de circuit (base + override)
    141:     
    142:     Args:
    143:         base_config: Configuration de base
    144:         override_config: Configuration à surcharger
    145:     
    146:     Returns:
    147:         Configuration fusionnée
    148:     """
    149:     result = base_config.copy()
    150:     
    151:     for key, value in override_config.items():
    152:         if key in result and isinstance(result[key], dict) and isinstance(value, dict):
    153:             result[key] = merge_circuit_configs(result[key], value)
    154:         else:
    155:             result[key] = value
    156:     
    157:     return result
    158: 
    159: 
    160: def calculate_recovery_time(stats: Dict[str, Any]) -> Optional[float]:
    161:     """
    162:     Calcule le temps estimé de récupération
    163:     
    164:     Args:
    165:         stats: Statistiques du circuit
    166:     
    167:     Returns:
    168:         Temps estimé en secondes, ou None si non disponible
    169:     """
    170:     if stats.get("state") != "open":
    171:         return None
    172:     
    173:     opened_at = stats.get("opened_at")
    174:     if not opened_at:
    175:         return None
    176:     
    177:     try:
    178:         if isinstance(opened_at, str):
    179:             opened_at = datetime.fromisoformat(opened_at)
    180:         
    181:         timeout = stats.get("timeout_seconds", 30)
    182:         elapsed = (datetime.now() - opened_at).total_seconds()
    183:         
    184:         return max(0, timeout - elapsed)
    185:     except:
    186:         return None