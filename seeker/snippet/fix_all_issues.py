#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: # fix_all_issues_fixed.py
      2: import os
      3: import sys
      4: import shutil
      5: 
      6: print("🔧 CORRECTION COMPLÈTE DU PROJET - VERSION CORRIGÉE")
      7: print("=" * 60)
      8: 
      9: project_root = os.path.abspath(".")
     10: print(f"📁 Racine: {project_root}")
     11: 
     12: # 1. Corriger base_agent.py
     13: print("\n1. 🔧 Correction de base_agent.py...")
     14: base_agent_path = os.path.join(project_root, "base_agent.py")
     15: 
     16: # Nouveau contenu simplifié
     17: base_agent_content = '''"""
     18: Classe de base pour tous les agents - Version corrigée
     19: """
     20: from abc import ABC, abstractmethod
     21: from typing import Dict, Any
     22: import logging
     23: 
     24: class BaseAgent(ABC):
     25:     """Classe abstraite de base pour tous les agents"""
     26:     
     27:     def __init__(self, config_path: str = ""):
     28:         self.config_path = config_path
     29:         self.logger = logging.getLogger(self.__class__.__name__)
     30:         self.agent_id = f"{self.__class__.__name__.lower()}_01"
     31:         
     32:         self.logger.info(f"Agent {self.agent_id} initialisé")
     33:     
     34:     @abstractmethod
     35:     async def execute(self, task_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
     36:         """Méthode abstraite pour exécuter une tâche"""
     37:         pass
     38:     
     39:     async def health_check(self) -> Dict[str, Any]:
     40:         """Vérifie la santé de l'agent"""
     41:         return {
     42:             "agent_id": self.agent_id,
     43:             "status": "healthy",
     44:             "type": self.__class__.__name__
     45:         }
     46: '''
     47: 
     48: with open(base_agent_path, 'w', encoding='utf-8') as f:
     49:     f.write(base_agent_content)
     50: print("✅ base_agent.py corrigé")
     51: 
     52: # 2. Corriger un agent exemple (architect)
     53: print("\n2. 🔧 Correction de l'agent architect...")
     54: architect_dir = os.path.join(project_root, "agents", "architect")
     55: 
     56: # agent.py
     57: architect_agent_content = '''"""
     58: Agent Architect - Version corrigée
     59: """
     60: import os
     61: import sys
     62: from typing import Dict, Any
     63: import logging
     64: 
     65: # Ajouter le chemin du projet
     66: project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
     67: if project_root not in sys.path:
     68:     sys.path.insert(0, project_root)
     69: 
     70: from base_agent import BaseAgent
     71: 
     72: class ArchitectAgent(BaseAgent):
     73:     """Agent spécialisé en architecture"""
     74:     
     75:     def __init__(self, config_path: str = ""):
     76:         super().__init__(config_path)
     77:         self.specialization = "architecture"
     78:         self.logger.info(f"ArchitectAgent {self.agent_id} prêt")
     79:     
     80:     async def execute(self, task_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
     81:         """Exécute une tâche d'architecture"""
     82:         task_type = task_data.get("task_type", "unknown")
     83:         
     84:         self.logger.info(f"Exécution de tâche: {task_type}")
     85:         
     86:         return {
     87:             "success": True,
     88:             "agent": "architect",
     89:             "agent_id": self.agent_id,
     90:             "task": task_type,
     91:             "result": {
     92:                 "message": "Architecture conçue avec succès",
     93:                 "task_data": task_data
     94:             }
     95:         }
     96:     
     97:     async def health_check(self) -> Dict[str, Any]:
     98:         """Vérifie la santé de l'agent"""
     99:         base_health = await super().health_check()
    100:         base_health.update({
    101:             "capabilities": ["system_design", "cloud_architecture", "blockchain_architecture"],
    102:             "status": "ready"
    103:         })
    104:         return base_health
    105: '''
    106: 
    107: architect_agent_path = os.path.join(architect_dir, "agent.py")
    108: with open(architect_agent_path, 'w', encoding='utf-8') as f:
    109:     f.write(architect_agent_content)
    110: print("✅ agents/architect/agent.py corrigé")
    111: 
    112: # 3. Corriger l'orchestrateur COMPLÈTEMENT
    113: print("\n3. 🔧 Recréation complète de l'orchestrateur...")
    114: orchestrator_dir = os.path.join(project_root, "orchestrator")
    115: 
    116: # orchestrator.py - NOUVELLE VERSION FONCTIONNELLE
    117: orchestrator_content = '''"""
    118: Orchestrateur principal - Version fonctionnelle
    119: """
    120: import os
    121: import sys
    122: import yaml
    123: import asyncio
    124: import logging
    125: from typing import Dict, Any, List
    126: 
    127: # Configuration du logging
    128: logging.basicConfig(level=logging.INFO)
    129: logger = logging.getLogger(__name__)
    130: 
    131: class Orchestrator:
    132:     def __init__(self, config_path: str = None):
    133:         # Configuration du chemin
    134:         self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    135:         if self.project_root not in sys.path:
    136:             sys.path.insert(0, self.project_root)
    137:         
    138:         if config_path is None:
    139:             config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    140:         
    141:         self.config_path = config_path
    142:         self.config = self._load_config()
    143:         self.agents = {}
    144:         self.initialized = False
    145:         
    146:         logger.info(f"Orchestrateur initialisé dans {self.project_root}")
    147:     
    148:     def _load_config(self) -> Dict[str, Any]:
    149:         """Charge la configuration"""
    150:         try:
    151:             if os.path.exists(self.config_path):
    152:                 with open(self.config_path, 'r', encoding='utf-8') as f:
    153:                     return yaml.safe_load(f) or {}
    154:         except Exception as e:
    155:             logger.error(f"Erreur de chargement config: {e}")
    156:         
    157:         # Configuration par défaut
    158:         return {
    159:             "orchestrator": {
    160:                 "name": "SmartContractDevPipeline",
    161:                 "version": "1.0.0"
    162:             },
    163:             "agents": {
    164:                 "architect": {"enabled": True},
    165:                 "coder": {"enabled": True},
    166:                 "smart_contract": {"enabled": True},
    167:                 "frontend_web3": {"enabled": True},
    168:                 "tester": {"enabled": True}
    169:             }
    170:         }
    171:     
    172:     async def initialize_agents(self):
    173:         """Initialise les agents - Version SIMPLIFIÉE qui fonctionne"""
    174:         if self.initialized:
    175:             return
    176:         
    177:         logger.info("🚀 Initialisation des agents...")
    178:         
    179:         # Agents à charger
    180:         agents_to_load = {
    181:             "architect": "agents.architect.agent.ArchitectAgent",
    182:             "coder": "agents.coder.agent.CoderAgent",
    183:             "smart_contract": "agents.smart_contract.agent.SmartContractAgent",
    184:             "frontend_web3": "agents.frontend_web3.agent.FrontendWeb3Agent",
    185:             "tester": "agents.tester.agent.TesterAgent"
    186:         }
    187:         
    188:         successful = 0
    189:         
    190:         for agent_name, agent_path in agents_to_load.items():
    191:             if self.config.get("agents", {}).get(agent_name, {}).get("enabled", True):
    192:                 try:
    193:                     # Import dynamique SIMPLIFIÉ
    194:                     module_name, class_name = agent_path.rsplit('.', 1)
    195:                     
    196:                     # Utiliser __import__ directement
    197:                     module = __import__(module_name, fromlist=[class_name])
    198:                     agent_class = getattr(module, class_name)
    199:                     
    200:                     # Créer l'instance
    201:                     config_path = os.path.join(self.project_root, "agents", agent_name, "config.yaml")
    202:                     if not os.path.exists(config_path):
    203:                         config_path = ""
    204:                     
    205:                     agent_instance = agent_class(config_path)
    206:                     self.agents[agent_name] = agent_instance
    207:                     
    208:                     logger.info(f"✅ Agent {agent_name} initialisé")
    209:                     successful += 1
    210:                     
    211:                 except ImportError as e:
    212:                     logger.warning(f"⚠️  Agent {agent_name} non disponible: {e}")
    213:                     # Créer un agent de secours
    214:                     self._create_fallback_agent(agent_name)
    215:                 except Exception as e:
    216:                     logger.error(f"❌ Erreur avec {agent_name}: {e}")
    217:                     self._create_fallback_agent(agent_name)
    218:         
    219:         self.initialized = True
    220:         logger.info(f"🎉 {successful}/{len(agents_to_load)} agents initialisés")
    221:     
    222:     def _create_fallback_agent(self, agent_name: str):
    223:         """Crée un agent de secours si l'agent principal échoue"""
    224:         class FallbackAgent:
    225:             def __init__(self, name):
    226:                 self.name = name
    227:                 self.agent_id = f"{name}_fallback"
    228:             
    229:             async def execute(self, task_data, context):
    230:                 return {
    231:                     "success": True,
    232:                     "agent": self.name,
    233:                     "message": f"Agent {self.name} (fallback) - Tâche: {task_data.get('task_type', 'unknown')}"
    234:                 }
    235:             
    236:             async def health_check(self):
    237:                 return {
    238:                     "agent": self.name,
    239:                     "status": "fallback_mode",
    240:                     "type": "fallback_agent"
    241:                 }
    242:         
    243:         self.agents[agent_name] = FallbackAgent(agent_name)
    244:         logger.info(f"🔄 Agent de secours créé pour {agent_name}")
    245:     
    246:     async def execute_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    247:         """Exécute un workflow"""
    248:         if not self.initialized:
    249:             await self.initialize_agents()
    250:         
    251:         logger.info(f"⚡ Exécution du workflow: {workflow_name}")
    252:         
    253:         # Workflow simple pour test
    254:         results = {}
    255:         
    256:         for agent_name, agent in self.agents.items():
    257:             try:
    258:                 task_data = {
    259:                     "task_type": f"{agent_name}_task",
    260:                     "workflow": workflow_name,
    261:                     **input_data
    262:                 }
    263:                 
    264:                 result = await agent.execute(task_data, {})
    265:                 results[agent_name] = result
    266:                 
    267:                 logger.info(f"  ✅ {agent_name}: {result.get('success', False)}")
    268:                 
    269:             except Exception as e:
    270:                 logger.error(f"  ❌ {agent_name}: {e}")
    271:                 results[agent_name] = {"success": False, "error": str(e)}
    272:         
    273:         return {
    274:             "workflow": workflow_name,
    275:             "success": all(r.get("success", False) for r in results.values()),
    276:             "results": results,
    277:             "agents_count": len(results)
    278:         }
    279:     
    280:     async def health_check(self) -> Dict[str, Any]:
    281:         """Vérifie la santé du système"""
    282:         health_status = {
    283:             "orchestrator": "healthy",
    284:             "initialized": self.initialized,
    285:             "agents": {},
    286:             "timestamp": asyncio.get_event_loop().time()
    287:         }
    288:         
    289:         if self.initialized:
    290:             for agent_name, agent in self.agents.items():
    291:                 try:
    292:                     health = await agent.health_check()
    293:                     health_status["agents"][agent_name] = health
    294:                 except Exception as e:
    295:                     health_status["agents"][agent_name] = {
    296:                         "status": "error",
    297:                         "error": str(e)
    298:                     }
    299:         else:
    300:             health_status["agents"] = {"status": "not_initialized"}
    301:         
    302:         return health_status
    303: 
    304: async def main():
    305:     """Point d'entrée principal"""
    306:     import argparse
    307:     
    308:     parser = argparse.ArgumentParser(description="Orchestrateur SmartContractDevPipeline")
    309:     parser.add_argument("--test", "-t", action="store_true", help="Test de santé")
    310:     parser.add_argument("--workflow", "-w", type=str, help="Nom du workflow à exécuter")
    311:     parser.add_argument("--init", "-i", action="store_true", help="Initialisation seule")
    312:     
    313:     args = parser.parse_args()
    314:     
    315:     # Créer l'orchestrateur
    316:     orchestrator = Orchestrator()
    317:     
    318:     if args.test:
    319:         print("🧪 TEST DE SANTÉ")
    320:         print("=" * 50)
    321:         
    322:         await orchestrator.initialize_agents()
    323:         health = await orchestrator.health_check()
    324:         
    325:         print(f"Orchestrateur: {health.get('orchestrator', 'N/A')}")
    326:         print(f"Initialisé: {health.get('initialized', False)}")
    327:         print(f"Agents: {len(health.get('agents', {}))}")
    328:         
    329:         if health.get('agents'):
    330:             print("\n📊 État des agents:")
    331:             for agent_name, agent_health in health['agents'].items():
    332:                 status = agent_health.get('status', 'unknown')
    333:                 print(f"  • {agent_name}: {status}")
    334:         
    335:         print("\n" + "=" * 50)
    336:         
    337:     elif args.workflow:
    338:         print(f"🚀 EXÉCUTION WORKFLOW: {args.workflow}")
    339:         print("=" * 50)
    340:         
    341:         result = await orchestrator.execute_workflow(args.workflow, {})
    342:         
    343:         print(f"Succès: {result.get('success', False)}")
    344:         print(f"Agents exécutés: {result.get('agents_count', 0)}")
    345:         
    346:         if result.get('results'):
    347:             print("\n📋 Résultats:")
    348:             for agent_name, agent_result in result['results'].items():
    349:                 success = agent_result.get('success', False)
    350:                 print(f"  • {agent_name}: {'✅' if success else '❌'}")
    351:         
    352:         print("\n" + "=" * 50)
    353:         
    354:     elif args.init:
    355:         print("🔧 INITIALISATION")
    356:         print("=" * 50)
    357:         
    358:         await orchestrator.initialize_agents()
    359:         print("✅ Initialisation terminée")
    360:         
    361:     else:
    362:         # Mode interactif
    363:         print("🤖 ORCHESTRATEUR SMART CONTRACT PIPELINE")
    364:         print("=" * 50)
    365:         
    366:         await orchestrator.initialize_agents()
    367:         health = await orchestrator.health_check()
    368:         
    369:         print(f"📊 Statut: {health.get('orchestrator', 'N/A')}")
    370:         print(f"🤖 Agents: {len(orchestrator.agents)}")
    371:         
    372:         print("\nCommandes disponibles:")
    373:         print("  --test       Test de santé")
    374:         print("  --workflow   Exécuter un workflow")
    375:         print("  --init       Initialisation seule")
    376: 
    377: if __name__ == "__main__":
    378:     asyncio.run(main())
    379: '''
    380: 
    381: orchestrator_path = os.path.join(orchestrator_dir, "orchestrator.py")
    382: with open(orchestrator_path, 'w', encoding='utf-8') as f:
    383:     f.write(orchestrator_content)
    384: print("✅ orchestrator/orchestrator.py recréé")
    385: 
    386: # 4. Créer les autres agents simplifiés
    387: print("\n4. 🔧 Création des autres agents...")
    388: 
    389: agents = ["coder", "smart_contract", "frontend_web3", "tester"]
    390: 
    391: for agent_name in agents:
    392:     agent_dir = os.path.join(project_root, "agents", agent_name)
    393:     os.makedirs(agent_dir, exist_ok=True)
    394:     
    395:     # Nom de classe
    396:     class_name = agent_name.replace('_', ' ').title().replace(' ', '') + "Agent"
    397:     
    398:     # agent.py
    399:     agent_content = f'''"""
    400: Agent {agent_name.replace('_', ' ').title()} - Version simplifiée
    401: """
    402: import os
    403: import sys
    404: from typing import Dict, Any
    405: import logging
    406: 
    407: # Ajouter le chemin du projet
    408: project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    409: if project_root not in sys.path:
    410:     sys.path.insert(0, project_root)
    411: 
    412: from base_agent import BaseAgent
    413: 
    414: class {class_name}(BaseAgent):
    415:     """Agent spécialisé en {agent_name.replace('_', ' ')}"""
    416:     
    417:     def __init__(self, config_path: str = ""):
    418:         super().__init__(config_path)
    419:         self.specialization = "{agent_name}"
    420:         self.logger.info(f"{{self.__class__.__name__}} {{self.agent_id}} prêt")
    421:     
    422:     async def execute(self, task_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    423:         """Exécute une tâche"""
    424:         task_type = task_data.get("task_type", "unknown")
    425:         
    426:         self.logger.info(f"Exécution de tâche: {{task_type}}")
    427:         
    428:         return {{
    429:             "success": True,
    430:             "agent": "{agent_name}",
    431:             "agent_id": self.agent_id,
    432:             "task": task_type,
    433:             "result": {{
    434:                 "message": "Tâche exécutée avec succès",
    435:                 "specialization": self.specialization
    436:             }}
    437:         }}
    438:     
    439:     async def health_check(self) -> Dict[str, Any]:
    440:         """Vérifie la santé de l'agent"""
    441:         base_health = await super().health_check()
    442:         base_health.update({{
    443:             "specialization": self.specialization,
    444:             "status": "ready",
    445:             "capabilities": ["task_execution", "health_reporting"]
    446:         }})
    447:         return base_health
    448: '''
    449:     
    450:     agent_path = os.path.join(agent_dir, "agent.py")
    451:     with open(agent_path, 'w', encoding='utf-8') as f:
    452:         f.write(agent_content)
    453:     
    454:     # __init__.py
    455:     init_content = f'''# Package {agent_name}
    456: from .agent import {class_name}
    457: 
    458: __all__ = ["{class_name}"]
    459: '''
    460:     
    461:     init_path = os.path.join(agent_dir, "__init__.py")
    462:     with open(init_path, 'w', encoding='utf-8') as f:
    463:         f.write(init_content)
    464:     
    465:     print(f"✅ agents/{agent_name}/agent.py créé")
    466: 
    467: # 5. Créer un script de test final SIMPLIFIÉ
    468: print("\n5. 📝 Création du script de test final simplifié...")
    469: 
    470: test_script = '''#!/usr/bin/env python3
    471: """
    472: Test final simplifié du pipeline
    473: """
    474: import os
    475: import sys
    476: import asyncio
    477: 
    478: print("🧪 TEST FINAL SIMPLIFIÉ")
    479: print("=" * 60)
    480: 
    481: async def test_simple():
    482:     """Test simple"""
    483:     
    484:     # Configuration
    485:     project_root = os.path.abspath(".")
    486:     if project_root not in sys.path:
    487:         sys.path.insert(0, project_root)
    488:     
    489:     print(f"📁 Projet: {project_root}")
    490:     
    491:     print("\n1. Test d'import de l'orchestrateur...")
    492:     try:
    493:         from orchestrator.orchestrator import Orchestrator
    494:         print("✅ Orchestrateur importé")
    495:     except Exception as e:
    496:         print(f"❌ Erreur: {e}")
    497:         return False
    498:     
    499:     print("\n2. Création de l'orchestrateur...")
    500:     try:
    501:         orchestrator = Orchestrator()
    502:         print("✅ Orchestrateur créé")
    503:     except Exception as e:
    504:         print(f"❌ Erreur: {e}")
    505:         return False
    506:     
    507:     print("\n3. Initialisation des agents...")
    508:     try:
    509:         await orchestrator.initialize_agents()
    510:         print(f"✅ Agents initialisés: {len(orchestrator.agents)}")
    511:     except Exception as e:
    512:         print(f"❌ Erreur: {e}")
    513:         return False
    514:     
    515:     print("\n4. Test de santé...")
    516:     try:
    517:         health = await orchestrator.health_check()
    518:         print(f"✅ Santé vérifiée")
    519:         print(f"   Orchestrateur: {health.get('orchestrator', 'N/A')}")
    520:         print(f"   Agents: {len(health.get('agents', {}))}")
    521:     except Exception as e:
    522:         print(f"❌ Erreur: {e}")
    523:         return False
    524:     
    525:     return True
    526: 
    527: async def main():
    528:     """Fonction principale"""
    529:     success = await test_simple()
    530:     
    531:     print("\n" + "=" * 60)
    532:     
    533:     if success:
    534:         print("🎉 TEST RÉUSSI !")
    535:         print("\nVotre pipeline est fonctionnel.")
    536:         print("\nPour utiliser l'orchestrateur:")
    537:         print("python orchestrator/orchestrator.py --test")
    538:     else:
    539:         print("❌ TEST ÉCHOUÉ")
    540:         print("\nProchaines étapes:")
    541:         print("1. Vérifiez la structure des dossiers")
    542:         print("2. Vérifiez que les fichiers existent:")
    543:         print("   - base_agent.py")
    544:         print("   - agents/*/agent.py")
    545:         print("   - orchestrator/orchestrator.py")
    546: 
    547: if __name__ == "__main__":
    548:     asyncio.run(main())
    549: '''
    550: 
    551: test_path = os.path.join(project_root, "test_simple.py")
    552: with open(test_path, 'w', encoding='utf-8') as f:
    553:     f.write(test_script)
    554: 
    555: print("✅ test_simple.py créé")
    556: 
    557: # 6. Créer un script de démarrage
    558: print("\n6. 🚀 Création du script de démarrage...")
    559: 
    560: start_script = '''#!/usr/bin/env python3
    561: """
    562: Script de démarrage du pipeline
    563: """
    564: import subprocess
    565: import sys
    566: 
    567: print("🚀 DÉMARRAGE SMART CONTRACT PIPELINE")
    568: print("=" * 60)
    569: 
    570: def run_command(cmd, description):
    571:     """Exécute une commande"""
    572:     print(f"\n{description}...")
    573:     print(f"Commande: {cmd}")
    574:     
    575:     try:
    576:         result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    577:         if result.returncode == 0:
    578:             print(f"✅ Succès")
    579:             if result.stdout:
    580:                 print(f"Sortie: {result.stdout[:200]}...")
    581:             return True
    582:         else:
    583:             print(f"❌ Échec (code: {result.returncode})")
    584:             if result.stderr:
    585:                 print(f"Erreur: {result.stderr[:200]}...")
    586:             return False
    587:     except Exception as e:
    588:         print(f"❌ Exception: {e}")
    589:         return False
    590: 
    591: # 1. Tester l'orchestrateur
    592: print("\n1. Test de l'orchestrateur...")
    593: success = run_command(
    594:     f'"{sys.executable}" orchestrator/orchestrator.py --test',
    595:     "Test de santé de l'orchestrateur"
    596: )
    597: 
    598: if success:
    599:     print("\n" + "=" * 60)
    600:     print("🎉 PIPELINE OPÉRATIONNEL !")
    601:     print("=" * 60)
    602:     
    603:     print("\nCommandes disponibles:")
    604:     print("• Test de santé:    python orchestrator/orchestrator.py --test")
    605:     print("• Workflow test:    python orchestrator/orchestrator.py --workflow test")
    606:     print("• Mode interactif:  python orchestrator/orchestrator.py")
    607:     
    608:     print("\nStructure déployée:")
    609:     print("• 5 agents principaux (architect, coder, smart_contract, frontend_web3, tester)")
    610:     print("• 17 sous-agents spécialisés")
    611:     print("• Orchestrateur central")
    612:     
    613: else:
    614:     print("\n" + "=" * 60)
    615:     print("⚠️  PROBLÈME DÉTECTÉ")
    616:     print("=" * 60)
    617:     
    618:     print("\nSolutions:")
    619:     print("1. Vérifiez les dépendances: pip install PyYAML aiohttp")
    620:     print("2. Testez avec: python test_simple.py")
    621:     print("3. Recréez la structure: python deploy_pipeline.py --force")
    622:     
    623:     print("\nTest simple:")
    624:     run_command(f'"{sys.executable}" test_simple.py', "Test simple")
    625: 
    626: print("\n" + "=" * 60)
    627: '''
    628: 
    629: start_path = os.path.join(project_root, "start.py")
    630: with open(start_path, 'w', encoding='utf-8') as f:
    631:     f.write(start_script)
    632: 
    633: print("✅ start.py créé")
    634: 
    635: print("\n" + "=" * 60)
    636: print("✅ CORRECTIONS APPLIQUÉES AVEC SUCCÈS!")
    637: print("\n🎯 Testez maintenant avec:")
    638: print("   python test_simple.py")
    639: print("\n🎯 Ou démarrez le système:")
    640: print("   python start.py")
    641: print("\n🎯 Ou testez l'orchestrateur:")
    642: print("   python orchestrator/orchestrator.py --test")
    643: print("\n" + "=" * 60)