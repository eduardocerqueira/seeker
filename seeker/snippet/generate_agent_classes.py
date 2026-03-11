#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script de génération automatique des classes d'agents manquantes
      4: Version corrigée - bug des f-strings résolu
      5: """
      6: 
      7: import os
      8: import sys
      9: from pathlib import Path
     10: 
     11: # Configuration
     12: ROOT_DIR = Path("D:/Web3Projects\SmartContractDevPipeline")
     13: AGENTS_DIR = ROOT_DIR / "agents"
     14: 
     15: # Mapping des noms de dossiers vers les noms de classes et descriptions
     16: AGENT_CLASSES = {
     17:     # Agents principaux
     18:     'architect': {
     19:         'class_name': 'ArchitectAgent',
     20:         'description': 'Agent responsable de la conception architecturale complète',
     21:         'version': '3.0.0'
     22:     },
     23:     'coder': {
     24:         'class_name': 'CoderAgent',
     25:         'description': 'Agent responsable du développement complet du code',
     26:         'version': '2.2.0'
     27:     },
     28:     'communication': {
     29:         'class_name': 'CommunicationAgent',
     30:         'description': 'Agent gérant la communication inter-agents',
     31:         'version': '1.0.0'
     32:     },
     33:     'database': {
     34:         'class_name': 'DatabaseAgent',
     35:         'description': 'Agent spécialisé dans la conception de bases de données',
     36:         'version': '1.0.0'
     37:     },
     38:     'documenter': {
     39:         'class_name': 'DocumenterAgent',
     40:         'description': 'Agent de documentation technique',
     41:         'version': '2.2.0'
     42:     },
     43:     'formal_verification': {
     44:         'class_name': 'FormalVerificationAgent',
     45:         'description': 'Agent de vérification formelle des propriétés',
     46:         'version': '1.0.0'
     47:     },
     48:     'frontend_web3': {
     49:         'class_name': 'FrontendWeb3Agent',
     50:         'description': 'Agent de développement frontend Web3',
     51:         'version': '2.2.0'
     52:     },
     53:     'fuzzing_simulation': {
     54:         'class_name': 'FuzzingSimulationAgent',
     55:         'description': 'Agent de tests de sécurité par fuzzing',
     56:         'version': '1.0.0'
     57:     },
     58:     'learning': {
     59:         'class_name': 'LearningAgent',
     60:         'description': 'Agent d\'apprentissage automatique',
     61:         'version': '1.0.0'
     62:     },
     63:     'monitoring': {
     64:         'class_name': 'MonitoringAgent',
     65:         'description': 'Agent de surveillance et monitoring',
     66:         'version': '1.0.0'
     67:     },
     68:     'orchestrator': {
     69:         'class_name': 'OrchestratorAgent',
     70:         'description': 'Agent d\'orchestration des workflows',
     71:         'version': '2.2.0'
     72:     },
     73:     'registry': {
     74:         'class_name': 'RegistryAgent',
     75:         'description': 'Agent de gestion du registre',
     76:         'version': '2.0.0'
     77:     },
     78:     'smart_contract': {
     79:         'class_name': 'SmartContractAgent',
     80:         'description': 'Agent expert en contrats intelligents',
     81:         'version': '2.2.0'
     82:     },
     83:     'storage': {
     84:         'class_name': 'StorageAgent',
     85:         'description': 'Agent de gestion des données',
     86:         'version': '1.0.0'
     87:     },
     88:     'tester': {
     89:         'class_name': 'TesterAgent',
     90:         'description': 'Agent de tests et assurance qualité',
     91:         'version': '2.2.0'
     92:     },
     93:     
     94:     # Sous-agents Architect
     95:     'blockchain_architect': {
     96:         'class_name': 'BlockchainArchitectSubAgent',
     97:         'description': 'Sous-agent spécialisé en architecture blockchain',
     98:         'parent': 'architect',
     99:         'version': '1.0.0'
    100:     },
    101:     'cloud_architect': {
    102:         'class_name': 'CloudArchitectSubAgent',
    103:         'description': 'Sous-agent spécialisé en architecture cloud',
    104:         'parent': 'architect',
    105:         'version': '1.0.0'
    106:     },
    107:     'microservices_architect': {
    108:         'class_name': 'MicroservicesArchitectSubAgent',
    109:         'description': 'Sous-agent spécialisé en microservices',
    110:         'parent': 'architect',
    111:         'version': '1.0.0'
    112:     },
    113:     
    114:     # Sous-agents Coder
    115:     'backend_coder': {
    116:         'class_name': 'BackendCoderSubAgent',
    117:         'description': 'Sous-agent spécialisé en développement backend',
    118:         'parent': 'coder',
    119:         'version': '1.0.0'
    120:     },
    121:     'devops_coder': {
    122:         'class_name': 'DevopsCoderSubAgent',
    123:         'description': 'Sous-agent spécialisé en DevOps',
    124:         'parent': 'coder',
    125:         'version': '1.0.0'
    126:     },
    127:     'frontend_coder': {
    128:         'class_name': 'FrontendCoderSubAgent',
    129:         'description': 'Sous-agent spécialisé en développement frontend',
    130:         'parent': 'coder',
    131:         'version': '1.0.0'
    132:     },
    133:     
    134:     # Sous-agents Frontend Web3
    135:     'react_expert': {
    136:         'class_name': 'ReactExpertSubAgent',
    137:         'description': 'Sous-agent expert en React',
    138:         'parent': 'frontend_web3',
    139:         'version': '1.0.0'
    140:     },
    141:     'ui_ux_expert': {
    142:         'class_name': 'UiUxExpertSubAgent',
    143:         'description': 'Sous-agent expert en UI/UX',
    144:         'parent': 'frontend_web3',
    145:         'version': '1.0.0'
    146:     },
    147:     'web3_integration': {
    148:         'class_name': 'Web3IntegrationSubAgent',
    149:         'description': 'Sous-agent expert en intégration Web3',
    150:         'parent': 'frontend_web3',
    151:         'version': '1.0.0'
    152:     },
    153:     
    154:     # Sous-agents Smart Contract
    155:     'formal_verification': {
    156:         'class_name': 'FormalVerificationSubAgent',
    157:         'description': 'Sous-agent spécialisé en vérification formelle',
    158:         'parent': 'smart_contract',
    159:         'version': '1.0.0'
    160:     },
    161:     'gas_optimizer': {
    162:         'class_name': 'GasOptimizerSubAgent',
    163:         'description': 'Sous-agent spécialisé en optimisation gas',
    164:         'parent': 'smart_contract',
    165:         'version': '1.0.0'
    166:     },
    167:     'security_expert': {
    168:         'class_name': 'SecurityExpertSubAgent',
    169:         'description': 'Sous-agent expert en sécurité',
    170:         'parent': 'smart_contract',
    171:         'version': '1.0.0'
    172:     },
    173:     'solidity_expert': {
    174:         'class_name': 'SolidityExpertSubAgent',
    175:         'description': 'Sous-agent expert en Solidity',
    176:         'parent': 'smart_contract',
    177:         'version': '1.0.0'
    178:     },
    179:     
    180:     # Sous-agents Tester
    181:     'e2e_tester': {
    182:         'class_name': 'E2ETesterSubAgent',
    183:         'description': 'Sous-agent spécialisé en tests E2E',
    184:         'parent': 'tester',
    185:         'version': '1.0.0'
    186:     },
    187:     'fuzzing_expert': {
    188:         'class_name': 'FuzzingExpertSubAgent',
    189:         'description': 'Sous-agent expert en fuzzing',
    190:         'parent': 'tester',
    191:         'version': '1.0.0'
    192:     },
    193:     'integration_tester': {
    194:         'class_name': 'IntegrationTesterSubAgent',
    195:         'description': 'Sous-agent spécialisé en tests d\'intégration',
    196:         'parent': 'tester',
    197:         'version': '1.0.0'
    198:     },
    199:     'unit_tester': {
    200:         'class_name': 'UnitTesterSubAgent',
    201:         'description': 'Sous-agent spécialisé en tests unitaires',
    202:         'parent': 'tester',
    203:         'version': '1.0.0'
    204:     }
    205: }
    206: 
    207: def get_agent_base_template(class_name, description, version="1.0.0"):
    208:     """Génère le template pour un agent principal."""
    209:     return f'''"""
    210: {description}
    211: Version {version}
    212: """
    213: 
    214: import os
    215: import sys
    216: import logging
    217: from typing import Dict, Any, List, Optional
    218: from datetime import datetime
    219: 
    220: from agents.base_agent.base_agent import BaseAgent, AgentStatus
    221: 
    222: logger = logging.getLogger(__name__)
    223: 
    224: class {class_name}(BaseAgent):
    225:     """
    226:     {description}
    227:     """
    228:     
    229:     def __init__(self, config_path: str = None):
    230:         """
    231:         Initialise l'agent.
    232:         
    233:         Args:
    234:             config_path: Chemin vers le fichier de configuration
    235:         """
    236:         if config_path is None:
    237:             config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    238:         
    239:         super().__init__(config_path)
    240:         self.logger = logging.getLogger(f"agent.{class_name}")
    241:         self.logger.info(f"Agent {class_name} créé (config: {{config_path}})")
    242:         self.version = "{version}"
    243:     
    244:     async def _initialize_components(self):
    245:         """Initialise les composants spécifiques à l'agent."""
    246:         self.logger.info(f"Initialisation des composants de {class_name}...")
    247:         return True
    248:     
    249:     async def _handle_custom_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
    250:         """
    251:         Gère les messages personnalisés.
    252:         
    253:         Args:
    254:             message: Message reçu
    255:             
    256:         Returns:
    257:             Réponse au message
    258:         """
    259:         msg_type = message.get("type", "unknown")
    260:         self.logger.info(f"Message reçu: {{msg_type}}")
    261:         return {{"status": "received", "type": msg_type}}
    262:     
    263:     async def execute(self, task_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    264:         """
    265:         Exécute une tâche.
    266:         
    267:         Args:
    268:             task_data: Données de la tâche
    269:             context: Contexte d'exécution
    270:             
    271:         Returns:
    272:             Résultat de l'exécution
    273:         """
    274:         self.logger.info(f"Exécution de la tâche: {{task_data.get('task_type', 'unknown')}}")
    275:         return {{
    276:             "status": "success",
    277:             "agent": self.name,
    278:             "result": {{"message": "Tâche exécutée avec succès"}},
    279:             "timestamp": datetime.now().isoformat()
    280:         }}
    281:     
    282:     async def health_check(self) -> Dict[str, Any]:
    283:         """
    284:         Vérifie la santé de l'agent.
    285:         
    286:         Returns:
    287:             Rapport de santé
    288:         """
    289:         return {{
    290:             "agent": self.name,
    291:             "status": "healthy",
    292:             "version": self.version,
    293:             "timestamp": datetime.now().isoformat()
    294:         }}
    295:     
    296:     def get_agent_info(self) -> Dict[str, Any]:
    297:         """
    298:         Retourne les informations de l'agent.
    299:         
    300:         Returns:
    301:             Informations de l'agent
    302:         """
    303:         return {{
    304:             "id": self.name,
    305:             "name": "{class_name}",
    306:             "version": self.version,
    307:             "status": self._status.value if hasattr(self._status, 'value') else str(self._status)
    308:         }}
    309: '''
    310: 
    311: def get_subagent_template(class_name, description, parent, version="1.0.0"):
    312:     """Génère le template pour un sous-agent."""
    313:     return f'''"""
    314: {description}
    315: Sous-agent de {parent}
    316: Version {version}
    317: """
    318: 
    319: import os
    320: import sys
    321: import logging
    322: from typing import Dict, Any, List, Optional
    323: from datetime import datetime
    324: 
    325: from agents.base_agent.base_agent import BaseAgent, AgentStatus
    326: 
    327: logger = logging.getLogger(__name__)
    328: 
    329: class {class_name}(BaseAgent):
    330:     """
    331:     {description}
    332:     """
    333:     
    334:     def __init__(self, config_path: str = None):
    335:         """
    336:         Initialise le sous-agent.
    337:         
    338:         Args:
    339:             config_path: Chemin vers le fichier de configuration
    340:         """
    341:         if config_path is None:
    342:             config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    343:         
    344:         super().__init__(config_path)
    345:         self.logger = logging.getLogger(f"agent.{class_name}")
    346:         self.logger.info(f"Sous-agent {class_name} créé")
    347:         self.version = "{version}"
    348:         self.parent = "{parent}"
    349:         self.specialization = class_name.replace('SubAgent', '').replace('Agent', '')
    350:     
    351:     async def _initialize_components(self):
    352:         """Initialise les composants spécifiques."""
    353:         self.logger.info(f"Initialisation des composants de {class_name}...")
    354:         return True
    355:     
    356:     async def _handle_custom_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
    357:         """
    358:         Gère les messages personnalisés.
    359:         
    360:         Args:
    361:             message: Message reçu
    362:             
    363:         Returns:
    364:             Réponse au message
    365:         """
    366:         msg_type = message.get("type", "unknown")
    367:         self.logger.info(f"Message reçu: {{msg_type}}")
    368:         
    369:         result = await self._execute_specialized(message)
    370:         
    371:         return {{
    372:             "status": "success",
    373:             "agent": self.name,
    374:             "specialization": self.specialization,
    375:             "result": result,
    376:             "timestamp": datetime.now().isoformat()
    377:         }}
    378:     
    379:     async def _execute_specialized(self, message: Dict[str, Any]) -> Dict[str, Any]:
    380:         """
    381:         Exécute une tâche spécialisée.
    382:         
    383:         Args:
    384:             message: Message avec les données de la tâche
    385:             
    386:         Returns:
    387:             Résultat de l'exécution
    388:         """
    389:         return {{"message": "Tâche exécutée par le sous-agent spécialisé"}}
    390:     
    391:     async def execute(self, task_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    392:         """
    393:         Exécute une tâche.
    394:         
    395:         Args:
    396:             task_data: Données de la tâche
    397:             context: Contexte d'exécution
    398:             
    399:         Returns:
    400:             Résultat de l'exécution
    401:         """
    402:         self.logger.info(f"Exécution de la tâche spécialisée: {{task_data.get('task_type', 'unknown')}}")
    403:         return {{
    404:             "status": "success",
    405:             "agent": self.name,
    406:             "specialization": self.specialization,
    407:             "result": {{"message": "Tâche exécutée avec succès"}},
    408:             "timestamp": datetime.now().isoformat()
    409:         }}
    410:     
    411:     async def health_check(self) -> Dict[str, Any]:
    412:         """
    413:         Vérifie la santé du sous-agent.
    414:         
    415:         Returns:
    416:             Rapport de santé
    417:         """
    418:         return {{
    419:             "agent": self.name,
    420:             "status": "healthy",
    421:             "type": "sub_agent",
    422:             "specialization": self.specialization,
    423:             "version": self.version,
    424:             "timestamp": datetime.now().isoformat()
    425:         }}
    426:     
    427:     def get_agent_info(self) -> Dict[str, Any]:
    428:         """
    429:         Retourne les informations du sous-agent.
    430:         
    431:         Returns:
    432:             Informations du sous-agent
    433:         """
    434:         return {{
    435:             "id": self.name,
    436:             "name": "{class_name}",
    437:             "type": "sub_agent",
    438:             "parent": self.parent,
    439:             "specialization": self.specialization,
    440:             "version": self.version,
    441:             "status": self._status.value if hasattr(self._status, 'value') else str(self._status)
    442:         }}
    443: '''
    444: 
    445: def create_agent_file(agent_dir: Path, agent_info: dict):
    446:     """Crée le fichier agent.py pour un agent donné."""
    447:     agent_path = agent_dir / "agent.py"
    448:     
    449:     # Ne pas écraser si le fichier existe déjà et n'est pas vide
    450:     if agent_path.exists() and agent_path.stat().st_size > 100:
    451:         print(f"  ⏭️  {agent_dir.name} existe déjà - ignoré")
    452:         return
    453:     
    454:     print(f"  ✨ Création de {agent_dir.name}/agent.py")
    455:     
    456:     if 'parent' in agent_info:
    457:         # C'est un sous-agent
    458:         content = get_subagent_template(
    459:             agent_info['class_name'],
    460:             agent_info['description'],
    461:             agent_info['parent'],
    462:             agent_info.get('version', '1.0.0')
    463:         )
    464:     else:
    465:         # C'est un agent principal
    466:         content = get_agent_base_template(
    467:             agent_info['class_name'],
    468:             agent_info['description'],
    469:             agent_info.get('version', '1.0.0')
    470:         )
    471:     
    472:     with open(agent_path, 'w', encoding='utf-8') as f:
    473:         f.write(content)
    474: 
    475: def main():
    476:     """Parcourt tous les dossiers d'agents et crée les fichiers manquants."""
    477:     print("\n" + "="*70)
    478:     print("🚀 GÉNÉRATION AUTOMATIQUE DES CLASSES D'AGENTS - VERSION CORRIGÉE")
    479:     print("="*70)
    480:     
    481:     if not AGENTS_DIR.exists():
    482:         print(f"❌ Dossier agents introuvable: {AGENTS_DIR}")
    483:         return
    484:     
    485:     print(f"\n📂 Scan du dossier: {AGENTS_DIR}")
    486:     
    487:     created = 0
    488:     skipped = 0
    489:     not_found = 0
    490:     
    491:     # Créer les agents principaux
    492:     for agent_name, agent_info in AGENT_CLASSES.items():
    493:         # Chercher le dossier correspondant
    494:         found = False
    495:         for agent_dir in AGENTS_DIR.iterdir():
    496:             if agent_dir.is_dir() and agent_dir.name == agent_name:
    497:                 create_agent_file(agent_dir, agent_info)
    498:                 created += 1
    499:                 found = True
    500:                 break
    501:         
    502:         if not found:
    503:             # Chercher dans les sous-dossiers
    504:             for agent_dir in AGENTS_DIR.rglob(agent_name):
    505:                 if agent_dir.is_dir():
    506:                     create_agent_file(agent_dir, agent_info)
    507:                     created += 1
    508:                     found = True
    509:                     break
    510:         
    511:         if not found:
    512:             print(f"  ❌ Dossier {agent_name} non trouvé")
    513:             not_found += 1
    514:     
    515:     print("\n" + "="*70)
    516:     print(f"✅ Fichiers créés: {created}")
    517:     print(f"⏭️  Fichiers ignorés: {skipped}")
    518:     print(f"❌ Dossiers non trouvés: {not_found}")
    519:     print("="*70)
    520: 
    521: if __name__ == "__main__":
    522:     main()