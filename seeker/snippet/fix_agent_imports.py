#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script de correction automatique des fichiers agent.py
      4: Version 4.0 - Approche simplifiée
      5: """
      6: 
      7: import os
      8: import re
      9: import shutil
     10: import logging
     11: from pathlib import Path
     12: from datetime import datetime
     13: 
     14: # Configuration
     15: ROOT_DIR = Path("D:/Web3Projects/SmartContractDevPipeline")
     16: AGENTS_DIR = ROOT_DIR / "agents"
     17: 
     18: # Configuration du logging
     19: logging.basicConfig(
     20:     level=logging.INFO,
     21:     format='%(asctime)s - %(levelname)s - %(message)s',
     22:     datefmt='%H:%M:%S'
     23: )
     24: logger = logging.getLogger("fix_agent")
     25: 
     26: def backup_file(file_path: Path) -> bool:
     27:     """Crée une sauvegarde."""
     28:     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
     29:     backup_path = file_path.with_suffix(f'.py.{timestamp}.bak')
     30:     shutil.copy2(file_path, backup_path)
     31:     logger.info(f"  ✅ Backup créé: {backup_path.name}")
     32:     return True
     33: 
     34: def fix_agent_file(file_path: Path) -> bool:
     35:     """Corrige un fichier agent.py de façon simple."""
     36:     logger.info(f"\n📝 Traitement de: {file_path.relative_to(ROOT_DIR)}")
     37:     
     38:     # Lire le contenu
     39:     with open(file_path, 'r', encoding='utf-8') as f:
     40:         content = f.read()
     41:     
     42:     # Créer une sauvegarde
     43:     backup_file(file_path)
     44:     
     45:     # 1. CORRECTION DES IMPORTS
     46:     # Remplacer les imports relatifs/incorrects
     47:     content = re.sub(
     48:         r'from\s+\.\.?base_agent\s+import\s+BaseAgent',
     49:         'from agents.base_agent.base_agent import BaseAgent',
     50:         content
     51:     )
     52:     content = re.sub(
     53:         r'from\s+base_agent\s+import\s+BaseAgent',
     54:         'from agents.base_agent.base_agent import BaseAgent',
     55:         content
     56:     )
     57:     content = re.sub(
     58:         r'from\s+\.base_agent\s+import\s+BaseAgent',
     59:         'from agents.base_agent.base_agent import BaseAgent',
     60:         content
     61:     )
     62:     
     63:     # Ajouter AgentStatus si nécessaire
     64:     if 'AgentStatus' not in content and 'BaseAgent' in content:
     65:         content = content.replace(
     66:             'from agents.base_agent.base_agent import BaseAgent',
     67:             'from agents.base_agent.base_agent import BaseAgent, AgentStatus'
     68:         )
     69:     
     70:     # 2. AJOUTER LES IMPORTS MANQUANTS
     71:     imports_to_add = []
     72:     if 'import os' not in content:
     73:         imports_to_add.append('import os')
     74:     if 'import sys' not in content:
     75:         imports_to_add.append('import sys')
     76:     if 'import logging' not in content:
     77:         imports_to_add.append('import logging')
     78:     if 'from typing import' not in content:
     79:         imports_to_add.append('from typing import Dict, Any, List, Optional')
     80:     if 'from datetime import datetime' not in content:
     81:         imports_to_add.append('from datetime import datetime')
     82:     
     83:     if imports_to_add:
     84:         # Ajouter après les imports existants
     85:         import_section = '\n'.join(imports_to_add) + '\n\n'
     86:         content = import_section + content
     87:     
     88:     # 3. CORRECTION DE L'HÉRITAGE
     89:     # Vérifier si la classe hérite de BaseAgent
     90:     class_match = re.search(r'class\s+(\w+)\s*(?:\(.*?\))?\s*:', content)
     91:     if class_match:
     92:         class_name = class_match.group(1)
     93:         if 'BaseAgent' not in class_match.group(0):
     94:             # Ajouter l'héritage
     95:             content = content.replace(
     96:                 f'class {class_name}:',
     97:                 f'class {class_name}(BaseAgent):'
     98:             )
     99:     
    100:     # 4. AJOUTER LES MÉTHODES REQUISES SI ELLES MANQUENT
    101:     if 'async def _initialize_components' not in content:
    102:         # Ajouter après la classe
    103:         methods = '''
    104:     async def _initialize_components(self):
    105:         """Initialise les composants spécifiques."""
    106:         self.logger.info(f"Initialisation des composants de {class_name}...")
    107:         return True
    108: '''
    109:         # Insérer après le __init__
    110:         if '__init__' in content:
    111:             content = content.replace('__init__', '__init__' + methods)
    112:     
    113:     if 'async def _handle_custom_message' not in content:
    114:         method = '''
    115:     async def _handle_custom_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
    116:         """Gère les messages personnalisés."""
    117:         msg_type = message.get("type", "unknown")
    118:         self.logger.info(f"Message reçu: {msg_type}")
    119:         return {"status": "received", "type": msg_type}
    120: '''
    121:         content += method
    122:     
    123:     if 'async def health_check' not in content:
    124:         method = '''
    125:     async def health_check(self) -> Dict[str, Any]:
    126:         """Vérifie la santé de l'agent."""
    127:         return {
    128:             "agent": self.name,
    129:             "status": "healthy",
    130:             "timestamp": datetime.now().isoformat()
    131:         }
    132: '''
    133:         content += method
    134:     
    135:     if 'def get_agent_info' not in content:
    136:         method = '''
    137:     def get_agent_info(self) -> Dict[str, Any]:
    138:         """Retourne les informations de l'agent."""
    139:         return {
    140:             "id": self.name,
    141:             "name": "{class_name}",
    142:             "version": getattr(self, 'version', '1.0.0'),
    143:             "status": self._status.value if hasattr(self._status, 'value') else str(self._status)
    144:         }
    145: '''.format(class_name=class_name)
    146:         content += method
    147:     
    148:     # 5. AJOUTER LE LOGGER SI NÉCESSAIRE
    149:     if 'logger = logging.getLogger' not in content:
    150:         content = content.replace(
    151:             'import logging',
    152:             'import logging\n\nlogger = logging.getLogger(__name__)'
    153:         )
    154:     
    155:     # Sauvegarder
    156:     with open(file_path, 'w', encoding='utf-8') as f:
    157:         f.write(content)
    158:     
    159:     logger.info(f"  ✅ Fichier corrigé avec succès")
    160:     return True
    161: 
    162: def main():
    163:     print("\n" + "="*70)
    164:     print("🔧 SCRIPT DE CORRECTION SIMPLIFIÉ - VERSION 4.0")
    165:     print("="*70)
    166:     
    167:     if not AGENTS_DIR.exists():
    168:         logger.error(f"❌ Dossier agents introuvable: {AGENTS_DIR}")
    169:         return
    170:     
    171:     # Trouver tous les fichiers agent.py (sauf base_agent)
    172:     agent_files = []
    173:     for file_path in AGENTS_DIR.rglob("agent.py"):
    174:         if "base_agent" not in str(file_path):
    175:             agent_files.append(file_path)
    176:     
    177:     logger.info(f"📊 {len(agent_files)} fichiers agent.py à traiter\n")
    178:     
    179:     success = 0
    180:     failed = 0
    181:     
    182:     for file_path in agent_files:
    183:         try:
    184:             if fix_agent_file(file_path):
    185:                 success += 1
    186:         except Exception as e:
    187:             logger.error(f"❌ Erreur sur {file_path.name}: {e}")
    188:             failed += 1
    189:     
    190:     print("\n" + "="*70)
    191:     print(f"✅ Fichiers corrigés: {success}")
    192:     print(f"❌ Fichiers en échec: {failed}")
    193:     print("="*70)
    194: 
    195: if __name__ == "__main__":
    196:     main()