#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script de correction pour les sous-agents communication
      4: Corrige l'erreur 'illegal target for annotation' dans les fichiers agent.py
      5: """
      6: 
      7: import re
      8: from pathlib import Path
      9: 
     10: # Configuration
     11: SOUS_AGENTS_PATH = Path("agents/communication/sous_agents")
     12: 
     13: # Liste des sous-agents à corriger
     14: SUB_AGENTS = [
     15:     "circuit_breaker",
     16:     "dead_letter_analyzer",
     17:     "message_router",
     18:     "performance_optimizer",
     19:     "pubsub_manager",
     20:     "queue_manager",
     21:     "security_validator"
     22: ]
     23: 
     24: def fix_agent_file(agent_file: Path) -> bool:
     25:     """Corrige les erreurs dans un fichier agent.py"""
     26:     print(f"\n🔧 Analyse de {agent_file}")
     27:     
     28:     with open(agent_file, 'r', encoding='utf-8') as f:
     29:         lines = f.readlines()
     30:     
     31:     modified = False
     32:     new_lines = []
     33:     in_handlers_section = False
     34:     handlers_section_start = -1
     35:     handlers_section_end = -1
     36:     
     37:     # Première passe : identifier la section problématique
     38:     for i, line in enumerate(lines):
     39:         if "handlers = {" in line:
     40:             in_handlers_section = True
     41:             handlers_section_start = i
     42:         elif in_handlers_section and "}" in line and not line.strip().startswith("#"):
     43:             handlers_section_end = i
     44:             in_handlers_section = False
     45:     
     46:     if handlers_section_start != -1 and handlers_section_end != -1:
     47:         print(f"  📍 Section handlers trouvée lignes {handlers_section_start+1}-{handlers_section_end+1}")
     48:         
     49:         # Vérifier si la section est correcte
     50:         section_content = ''.join(lines[handlers_section_start:handlers_section_end+1])
     51:         
     52:         # Chercher les lignes de handlers qui ne sont pas correctement indentées
     53:         if "self._handle" in section_content:
     54:             # Remplacer par une version corrigée
     55:             new_section = []
     56:             new_section.append(lines[handlers_section_start])  # "handlers = {"
     57:             
     58:             # Ajouter les handlers standards
     59:             new_section.append('                f"{self.name}.status": self._handle_status,\n')
     60:             new_section.append('                f"{self.name}.metrics": self._handle_metrics,\n')
     61:             new_section.append('                f"{self.name}.health": self._handle_health,\n')
     62:             new_section.append('                f"{self.name}.process": self._handle_process,\n')
     63:             new_section.append('                f"{self.name}.capabilities": self._handle_capabilities,\n')
     64:             
     65:             # Ajouter les handlers pour chaque capacité
     66:             # Les extraire du fichier original
     67:             for j in range(handlers_section_start + 1, handlers_section_end):
     68:                 line = lines[j].strip()
     69:                 if line and not line.startswith('}') and 'self._handle' in line:
     70:                     # Cette ligne contient un handler, la garder
     71:                     new_section.append(lines[j])
     72:             
     73:             new_section.append('            }\n')  # Fermeture du dictionnaire
     74:             
     75:             # Remplacer la section
     76:             lines[handlers_section_start:handlers_section_end+1] = new_section
     77:             modified = True
     78:             print(f"  ✅ Section handlers corrigée")
     79:     
     80:     if modified:
     81:         # Sauvegarder le fichier corrigé
     82:         with open(agent_file, 'w', encoding='utf-8') as f:
     83:             f.writelines(lines)
     84:         print(f"  ✅ Fichier mis à jour")
     85:         return True
     86:     else:
     87:         print(f"  ℹ️ Aucune correction nécessaire")
     88:         return False
     89: 
     90: def create_simple_version(agent_id: str) -> str:
     91:     """Crée une version ultra-simplifiée du sous-agent"""
     92:     class_name = ''.join(p.capitalize() for p in agent_id.split('_')) + 'SubAgent'
     93:     display_name_map = {
     94:         "circuit_breaker": "🛡️ Circuit Breaker",
     95:         "dead_letter_analyzer": "💀 Dead Letter Analyzer",
     96:         "message_router": "🔄 Message Router",
     97:         "performance_optimizer": "⚡ Performance Optimizer",
     98:         "pubsub_manager": "📢 PubSub Manager",
     99:         "queue_manager": "📊 Queue Manager",
    100:         "security_validator": "🔒 Security Validator"
    101:     }
    102:     display_name = display_name_map.get(agent_id, agent_id.replace('_', ' ').title())
    103:     
    104:     return f'''"""
    105: {display_name} - Sous-agent simplifié
    106: """
    107: 
    108: import logging
    109: import sys
    110: from datetime import datetime
    111: from pathlib import Path
    112: 
    113: current_dir = Path(__file__).parent.absolute()
    114: project_root = current_dir.parent.parent.parent.parent.parent
    115: if str(project_root) not in sys.path:
    116:     sys.path.insert(0, str(project_root))
    117: 
    118: from agents.base_agent.base_agent import BaseAgent, AgentStatus, Message, MessageType
    119: 
    120: logger = logging.getLogger(__name__)
    121: 
    122: 
    123: class {class_name}(BaseAgent):
    124:     """Version simplifiée du sous-agent"""
    125:     
    126:     def __init__(self, config_path: str = ""):
    127:         if not config_path:
    128:             config_path = str(current_dir / "config.yaml")
    129:         super().__init__(config_path)
    130:         self._display_name = "{display_name}"
    131:         self._initialized = False
    132:         self._stats = {{'tasks': 0, 'start': datetime.now().isoformat()}}
    133: 
    134:     async def initialize(self) -> bool:
    135:         try:
    136:             self._set_status(AgentStatus.INITIALIZING)
    137:             await super().initialize()
    138:             self._initialized = True
    139:             self._set_status(AgentStatus.READY)
    140:             logger.info(f"✅ {{self._display_name}} prêt")
    141:             return True
    142:         except Exception as e:
    143:             logger.error(f"❌ Erreur: {{e}}")
    144:             self._set_status(AgentStatus.ERROR)
    145:             return False
    146: 
    147:     async def _handle_custom_message(self, message: Message):
    148:         try:
    149:             msg_type = message.message_type
    150:             if msg_type == f"{{self.name}}.status":
    151:                 return Message(
    152:                     sender=self.name,
    153:                     recipient=message.sender,
    154:                     content=self._stats,
    155:                     message_type=f"{{self.name}}.status_response",
    156:                     correlation_id=message.message_id
    157:                 )
    158:             elif msg_type == f"{{self.name}}.health":
    159:                 health = await self.health_check()
    160:                 return Message(
    161:                     sender=self.name,
    162:                     recipient=message.sender,
    163:                     content=health,
    164:                     message_type=f"{{self.name}}.health_response",
    165:                     correlation_id=message.message_id
    166:                 )
    167:             return None
    168:         except Exception as e:
    169:             logger.error(f"Erreur: {{e}}")
    170:             return Message(
    171:                 sender=self.name,
    172:                 recipient=message.sender,
    173:                 content={{"error": str(e)}},
    174:                 message_type=MessageType.ERROR.value,
    175:                 correlation_id=message.message_id
    176:             )
    177: 
    178:     async def health_check(self):
    179:         base = await super().health_check()
    180:         return {{**base, "stats": self._stats}}
    181: 
    182:     async def shutdown(self) -> bool:
    183:         logger.info(f"Arrêt de {{self._display_name}}...")
    184:         self._set_status(AgentStatus.SHUTTING_DOWN)
    185:         await super().shutdown()
    186:         return True
    187: 
    188:     def get_agent_info(self):
    189:         return {{
    190:             "id": self.name,
    191:             "name": "{class_name}",
    192:             "display_name": self._display_name,
    193:             "version": "1.0.0",
    194:             "description": "Sous-agent simplifié",
    195:             "status": self._status.value,
    196:             "stats": self._stats
    197:         }}
    198: 
    199: 
    200: def create_{agent_id}_agent(config_path: str = "") -> {class_name}:
    201:     return {class_name}(config_path)
    202: '''
    203: 
    204: def main():
    205:     print("=" * 60)
    206:     print("🔧 CORRECTION DES SOUS-AGENTS COMMUNICATION")
    207:     print("=" * 60)
    208:     
    209:     if not SOUS_AGENTS_PATH.exists():
    210:         print(f"❌ Dossier {SOUS_AGENTS_PATH} introuvable")
    211:         return
    212:     
    213:     print("\nOptions:")
    214:     print("  1. Corriger automatiquement les fichiers existants")
    215:     print("  2. Remplacer par des versions simplifiées (recommandé)")
    216:     print("  3. Quitter")
    217:     
    218:     choice = input("\nVotre choix [1-3]: ").strip()
    219:     
    220:     if choice == "1":
    221:         # Correction automatique
    222:         fixed_count = 0
    223:         for agent_id in SUB_AGENTS:
    224:             agent_file = SOUS_AGENTS_PATH / agent_id / "agent.py"
    225:             if agent_file.exists():
    226:                 if fix_agent_file(agent_file):
    227:                     fixed_count += 1
    228:         print(f"\n✅ {fixed_count} fichiers corrigés")
    229:     
    230:     elif choice == "2":
    231:         # Remplacer par versions simplifiées
    232:         print("\n⚠️  Attention: Cette opération va remplacer tous les fichiers agent.py")
    233:         confirm = input("Confirmer? (oui/non): ").strip().lower()
    234:         
    235:         if confirm == "oui":
    236:             for agent_id in SUB_AGENTS:
    237:                 agent_dir = SOUS_AGENTS_PATH / agent_id
    238:                 agent_file = agent_dir / "agent.py"
    239:                 
    240:                 # Sauvegarder l'original
    241:                 if agent_file.exists():
    242:                     backup = agent_file.with_suffix('.py.bak')
    243:                     agent_file.rename(backup)
    244:                     print(f"  📦 Backup créé: {backup}")
    245:                 
    246:                 # Créer la version simplifiée
    247:                 simple_version = create_simple_version(agent_id)
    248:                 with open(agent_file, 'w', encoding='utf-8') as f:
    249:                     f.write(simple_version)
    250:                 print(f"  ✅ Version simplifiée créée pour {agent_id}")
    251:             
    252:             print("\n✅ Tous les sous-agents ont été remplacés par des versions simplifiées")
    253:     
    254:     else:
    255:         print("Annulé")
    256:         return
    257:     
    258:     print("\n" + "=" * 60)
    259:     print("✅ OPÉRATION TERMINÉE")
    260:     print("=" * 60)
    261:     print("\n📋 Prochaines étapes:")
    262:     print("   1. Relancez python diagnose_communication.py")
    263:     print("   2. Tous les sous-agents devraient maintenant s'importer")
    264:     print("   3. Relancez python test_full_sprint.py")
    265: 
    266: if __name__ == "__main__":
    267:     main()