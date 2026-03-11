#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: """
      2: ⚡ Optimisateur Performance Web3 - Sous-agent spécialisé
      3: Version: 2.0.0
      4: 
      5: Expert en optimisation des performances Web3
      6: """
      7: 
      8: import logging
      9: import sys
     10: import asyncio
     11: from datetime import datetime
     12: from pathlib import Path
     13: from typing import Dict, Any, Optional
     14: 
     15: # Configuration des imports
     16: current_dir = Path(__file__).parent.absolute()
     17: project_root = current_dir.parent.parent.parent.parent.parent
     18: if str(project_root) not in sys.path:
     19:     sys.path.insert(0, str(project_root))
     20: 
     21: from agents.base_agent.base_agent import BaseAgent, AgentStatus, Message, MessageType
     22: 
     23: logger = logging.getLogger(__name__)
     24: 
     25: 
     26: class PerformanceOptimizerSubAgent(BaseAgent):
     27:     """
     28:     Expert en optimisation des performances Web3
     29:     """
     30:     
     31:     def __init__(self, config_path: str = ""):
     32:         """Initialise le sous-agent"""
     33:         if not config_path:
     34:             config_path = str(current_dir / "config.yaml")
     35: 
     36:         super().__init__(config_path)
     37: 
     38:         self._display_name = "⚡ Optimisateur Performance Web3"
     39:         self._initialized = False
     40:         self._components = {}
     41:         
     42:         # Statistiques
     43:         self._stats = {
     44:             'tasks_processed': 0,
     45:             'components_generated': 0,
     46:             'start_time': datetime.now().isoformat()
     47:         }
     48: 
     49:         logger.info(f"{self._display_name} créé - v{self._version}")
     50: 
     51:     async def initialize(self) -> bool:
     52:         """Initialise le sous-agent"""
     53:         try:
     54:             self._set_status(AgentStatus.INITIALIZING)
     55:             logger.info(f"Initialisation de {self._display_name}...")
     56: 
     57:             base_result = await super().initialize()
     58:             if not base_result:
     59:                 return False
     60: 
     61:             await self._initialize_components()
     62: 
     63:             self._initialized = True
     64:             self._set_status(AgentStatus.READY)
     65:             logger.info(f"✅ {self._display_name} prêt")
     66:             return True
     67: 
     68:         except Exception as e:
     69:             logger.error(f"❌ Erreur initialisation: {e}")
     70:             self._set_status(AgentStatus.ERROR)
     71:             return False
     72: 
     73:     async def _initialize_components(self) -> bool:
     74:         """Initialise les composants du sous-agent"""
     75:         logger.info("Initialisation des composants...")
     76:         
     77:         self._components = {
     78:             "capabilities": [
     79:         "bundle_optimization",
     80:         "core_web_vitals",
     81:         "caching_strategy",
     82:         "image_optimization",
     83:         "web3_performance",
     84:             ],
     85:             "enabled": True,
     86:             "version": "2.0.0"
     87:         }
     88:         
     89:         logger.info(f"✅ Composants: {list(self._components.keys())}")
     90:         return True
     91: 
     92:     async def _handle_custom_message(self, message: Message) -> Optional[Message]:
     93:         """Gère les messages personnalisés"""
     94:         try:
     95:             msg_type = message.message_type
     96:             logger.debug(f"Message reçu: {msg_type} de {message.sender}")
     97: 
     98:             handlers = {
     99:                 f"{self.name}.status": self._handle_status,
    100:                 f"{self.name}.metrics": self._handle_metrics,
    101:                 f"{self.name}.health": self._handle_health,
    102:             }
    103: 
    104:             if msg_type in handlers:
    105:                 return await handlers[msg_type](message)
    106: 
    107:             return None
    108: 
    109:         except Exception as e:
    110:             logger.error(f"Erreur traitement message: {e}")
    111:             return Message(
    112:                 sender=self.name,
    113:                 recipient=message.sender,
    114:                 content={"error": str(e)},
    115:                 message_type=MessageType.ERROR.value,
    116:                 correlation_id=message.message_id
    117:             )
    118: 
    119:     async def _handle_status(self, message: Message) -> Message:
    120:         """Retourne le statut général"""
    121:         return Message(
    122:             sender=self.name,
    123:             recipient=message.sender,
    124:             content={
    125:                 'status': self._status.value,
    126:                 'initialized': self._initialized,
    127:                 'stats': self._stats
    128:             },
    129:             message_type=f"{self.name}.status_response",
    130:             correlation_id=message.message_id
    131:         )
    132: 
    133:     async def _handle_metrics(self, message: Message) -> Message:
    134:         """Retourne les métriques"""
    135:         return Message(
    136:             sender=self.name,
    137:             recipient=message.sender,
    138:             content=self._stats,
    139:             message_type=f"{self.name}.metrics_response",
    140:             correlation_id=message.message_id
    141:         )
    142: 
    143:     async def _handle_health(self, message: Message) -> Message:
    144:         """Retourne l'état de santé"""
    145:         health = await self.health_check()
    146:         return Message(
    147:             sender=self.name,
    148:             recipient=message.sender,
    149:             content=health,
    150:             message_type=f"{self.name}.health_response",
    151:             correlation_id=message.message_id
    152:         )
    153: 
    154:     async def shutdown(self) -> bool:
    155:         """Arrête le sous-agent"""
    156:         logger.info(f"Arrêt de {self._display_name}...")
    157:         self._set_status(AgentStatus.SHUTTING_DOWN)
    158:         await super().shutdown()
    159:         logger.info(f"✅ {self._display_name} arrêté")
    160:         return True
    161: 
    162:     async def pause(self) -> bool:
    163:         """Met en pause le sous-agent"""
    164:         logger.info(f"Pause de {self._display_name}...")
    165:         self._set_status(AgentStatus.PAUSED)
    166:         return True
    167: 
    168:     async def resume(self) -> bool:
    169:         """Reprend l'activité"""
    170:         logger.info(f"Reprise de {self._display_name}...")
    171:         self._set_status(AgentStatus.READY)
    172:         return True
    173: 
    174:     async def health_check(self) -> Dict[str, Any]:
    175:         """Vérifie la santé du sous-agent"""
    176:         base_health = await super().health_check()
    177: 
    178:         uptime = None
    179:         if self._stats.get('start_time'):
    180:             start = datetime.fromisoformat(self._stats['start_time'])
    181:             uptime = str(datetime.now() - start)
    182: 
    183:         return {
    184:             **base_health,
    185:             "agent": self.name,
    186:             "display_name": self._display_name,
    187:             "status": self._status.value,
    188:             "ready": self._status == AgentStatus.READY,
    189:             "initialized": self._initialized,
    190:             "uptime": uptime,
    191:             "components": list(self._components.keys()),
    192:             "stats": self._stats,
    193:             "timestamp": datetime.now().isoformat()
    194:         }
    195: 
    196:     def get_agent_info(self) -> Dict[str, Any]:
    197:         """Informations pour le registre"""
    198:         return {
    199:             "id": self.name,
    200:             "name": "PerformanceOptimizerSubAgent",
    201:             "display_name": self._display_name,
    202:             "version": "2.0.0",
    203:             "description": """Expert en optimisation des performances Web3""",
    204:             "status": self._status.value,
    205:             "capabilities": [
    206:         "bundle_optimization",
    207:         "core_web_vitals",
    208:         "caching_strategy",
    209:         "image_optimization",
    210:         "web3_performance",
    211:             ],
    212:             "stats": self._stats
    213:         }
    214: 
    215: 
    216: def create_performance_optimizer_agent(config_path: str = "") -> PerformanceOptimizerSubAgent:
    217:     """Crée une instance du sous-agent"""
    218:     return PerformanceOptimizerSubAgent(config_path)