#date: 2026-03-16T17:43:30Z
#url: https://api.github.com/gists/66714a642202d1a738257924d121f72a
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script de déploiement complet pour SmartContractDevPipeline
      4: Déploie l'orchestrateur, les agents principaux et leurs sous-agents.
      5: Date: 2026-02-03
      6: Auteur: SmartContractDevPipeline
      7: """
      8: import os
      9: import sys
     10: import yaml
     11: import asyncio
     12: import subprocess
     13: from pathlib import Path
     14: from typing import Dict, List, Any
     15: import logging
     16: 
     17: # Configuration du logging
     18: logging.basicConfig(
     19:     level=logging.INFO,
     20:     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
     21: )
     22: logger = logging.getLogger(__name__)
     23: 
     24: class AgentDeployer:
     25:     """Classe principale pour déployer tous les composants du pipeline"""
     26:     
     27:     def __init__(self, project_root: str = None):
     28:         """Initialise le déployeur avec le chemin du projet"""
     29:         if project_root:
     30:             self.project_root = os.path.abspath(project_root)
     31:         else:
     32:             # Utiliser le dossier courant
     33:             self.project_root = os.path.abspath(".")
     34:         self.agents_path = os.path.join(self.project_root, "agents")
     35:         self.orchestrator_path = os.path.join(self.project_root, "orchestrator")
     36:         
     37:         # Structure des agents (identique à votre PS1)
     38:         self.agent_structure = {
     39:             "architect": [
     40:                 {"name": "cloud_architect", "type": "Cloud Architecture"},
     41:                 {"name": "blockchain_architect", "type": "Blockchain Architecture"},
     42:                 {"name": "microservices_architect", "type": "Microservices Architecture"},
     43:             ],
     44:             "coder": [
     45:                 {"name": "backend_coder", "type": "Backend Development"},
     46:                 {"name": "frontend_coder", "type": "Frontend Development"},
     47:                 {"name": "devops_coder", "type": "DevOps"},
     48:             ],
     49:             "smart_contract": [
     50:                 {"name": "solidity_expert", "type": "Solidity Development"},
     51:                 {"name": "security_expert", "type": "Smart Contract Security"},
     52:                 {"name": "gas_optimizer", "type": "Gas Optimization"},
     53:                 {"name": "formal_verification", "type": "Formal Verification"},
     54:             ],
     55:             "frontend_web3": [
     56:                 {"name": "react_expert", "type": "React/Next.js"},
     57:                 {"name": "web3_integration", "type": "Web3 Integration"},
     58:                 {"name": "ui_ux_expert", "type": "UI/UX Design"},
     59:             ],
     60:             "tester": [
     61:                 {"name": "unit_tester", "type": "Unit Testing"},
     62:                 {"name": "integration_tester", "type": "Integration Testing"},
     63:                 {"name": "e2e_tester", "type": "E2E Testing"},
     64:                 {"name": "fuzzing_expert", "type": "Fuzzing"},
     65:             ]
     66:         }
     67:     
     68:     def check_existing_deployment(self) -> Dict[str, bool]:
     69:         """Vérifie quels composants sont déjà déployés"""
     70:         status = {
     71:             "orchestrator": False,
     72:             "main_agents": {},
     73:             "sub_agents": {}
     74:         }
     75:         
     76:         # Vérifier l'orchestrateur
     77:         orchestrator_files = ["orchestrator.py", "config.yaml", "requirements.txt"]
     78:         if os.path.exists(self.orchestrator_path):
     79:             status["orchestrator"] = all(
     80:                 os.path.exists(os.path.join(self.orchestrator_path, f))
     81:                 for f in orchestrator_files
     82:             )
     83:         
     84:         # Vérifier les agents principaux
     85:         for agent in self.agent_structure.keys():
     86:             agent_dir = os.path.join(self.agents_path, agent)
     87:             if os.path.exists(agent_dir):
     88:                 main_files = ["agent.py", "config.yaml", "__init__.py"]
     89:                 status["main_agents"][agent] = all(
     90:                     os.path.exists(os.path.join(agent_dir, f))
     91:                     for f in main_files
     92:                 )
     93:             
     94:             # Vérifier les sous-agents
     95:             for sub_agent in self.agent_structure[agent]:
     96:                 sub_agent_dir = os.path.join(agent_dir, "sous_agents", sub_agent["name"])
     97:                 if os.path.exists(sub_agent_dir):
     98:                     sub_files = ["config.yaml", "agent.py", "tools.py", "__init__.py"]
     99:                     key = f"{agent}/{sub_agent['name']}"
    100:                     status["sub_agents"][key] = all(
    101:                         os.path.exists(os.path.join(sub_agent_dir, f))
    102:                         for f in sub_files
    103:                     )
    104:         
    105:         return status
    106:     
    107:     def create_orchestrator(self) -> bool:
    108:         """Crée et configure l'orchestrateur s'il n'existe pas"""
    109:         logger.info("🔧 Configuration de l'orchestrateur...")
    110:         
    111:         # Créer le dossier
    112:         os.makedirs(self.orchestrator_path, exist_ok=True)
    113:         
    114:         # Fichier orchestrateur.py
    115:         orchestrator_py = '''"""
    116: Orchestrateur principal du pipeline de développement
    117: Coordinate les agents et sous-agents
    118: """
    119: import asyncio
    120: import yaml
    121: from typing import Dict, List, Any
    122: from pathlib import Path
    123: import logging
    124: 
    125: class Orchestrator:
    126:     def __init__(self, config_path: str = "config.yaml"):
    127:         self.config_path = config_path
    128:         self.config = self.load_config()
    129:         self.agents = {}
    130:         self.logger = logging.getLogger(__name__)
    131:         self.initialized = False
    132:     
    133:     def load_config(self) -> Dict[str, Any]:
    134:         """Charge la configuration depuis le fichier YAML"""
    135:         try:
    136:             with open(self.config_path, 'r') as f:
    137:                 return yaml.safe_load(f) or {}
    138:         except FileNotFoundError:
    139:             return {"agents": {}, "workflow": {}}
    140:     
    141:     async def initialize_agents(self):
    142:         """Initialise tous les agents du pipeline"""
    143:         if self.initialized:
    144:             return
    145:         
    146:         self.logger.info("Initialisation des agents...")
    147:         
    148:         # Dynamiquement importer les agents basés sur la config
    149:         agents_to_load = self.config.get("agents", {})
    150:         
    151:         for agent_name, agent_config in agents_to_load.items():
    152:             if agent_config.get("enabled", True):
    153:                 try:
    154:                     # Construction du chemin d'import
    155:                     module_path = agent_config.get("module", f"agents.{agent_name}.agent")
    156:                     agent_class_name = agent_config.get("class", f"{agent_name.capitalize()}Agent")
    157:                     
    158:                     # Import dynamique
    159:                     module = __import__(module_path, fromlist=[agent_class_name])
    160:                     agent_class = getattr(module, agent_class_name)
    161:                     
    162:                     # Instanciation
    163:                     agent_instance = agent_class(agent_config.get("config_path", ""))
    164:                     self.agents[agent_name] = agent_instance
    165:                     
    166:                     self.logger.info(f"✅ Agent {agent_name} initialisé")
    167:                     
    168:                 except Exception as e:
    169:                     self.logger.error(f"❌ Erreur lors de l'initialisation de {agent_name}: {e}")
    170:         
    171:         self.initialized = True
    172:         self.logger.info(f"✅ {len(self.agents)} agents initialisés")
    173:     
    174:     async def execute_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    175:         """Exécute un workflow prédéfini"""
    176:         self.logger.info(f"Exécution du workflow: {workflow_name}")
    177:         
    178:         if not self.initialized:
    179:             await self.initialize_agents()
    180:         
    181:         workflow = self.config.get("workflow", {}).get(workflow_name, {})
    182:         steps = workflow.get("steps", [])
    183:         
    184:         results = {}
    185:         current_data = input_data.copy()
    186:         
    187:         for step in steps:
    188:             agent_name = step.get("agent")
    189:             task = step.get("task")
    190:             parameters = step.get("parameters", {})
    191:             
    192:             if agent_name in self.agents:
    193:                 try:
    194:                     self.logger.info(f"  → Étape: {agent_name}.{task}")
    195:                     
    196:                     # Fusionner les paramètres
    197:                     task_data = {**parameters, **current_data}
    198:                     
    199:                     # Exécuter la tâche
    200:                     result = await self.agents[agent_name].execute(task_data, {})
    201:                     
    202:                     # Mettre à jour les données pour les étapes suivantes
    203:                     if result.get("success"):
    204:                         current_data.update(result.get("output", {}))
    205:                         results[step.get("id", task)] = result
    206:                     else:
    207:                         self.logger.error(f"Échec de l'étape {task}")
    208:                         break
    209:                         
    210:                 except Exception as e:
    211:                     self.logger.error(f"Erreur dans l'étape {task}: {e}")
    212:                     break
    213:         
    214:         return {
    215:             "workflow": workflow_name,
    216:             "success": len(results) == len(steps),
    217:             "results": results,
    218:             "output_data": current_data
    219:         }
    220:     
    221:     async def health_check(self) -> Dict[str, Any]:
    222:         """Vérifie la santé de tous les agents"""
    223:         health_status = {"orchestrator": "healthy", "agents": {}}
    224:         
    225:         for agent_name, agent_instance in self.agents.items():
    226:             try:
    227:                 health = await agent_instance.health_check()
    228:                 health_status["agents"][agent_name] = health
    229:             except Exception as e:
    230:                 health_status["agents"][agent_name] = {
    231:                     "status": "error",
    232:                     "error": str(e)
    233:                 }
    234:         
    235:         return health_status
    236: 
    237: # Point d'entrée principal
    238: async def main():
    239:     orchestrator = Orchestrator()
    240:     await orchestrator.initialize_agents()
    241:     
    242:     # Exemple d'exécution
    243:     health = await orchestrator.health_check()
    244:     print(f"État du système: {health}")
    245: 
    246: if __name__ == "__main__":
    247:     asyncio.run(main())
    248: '''
    249:         
    250:         # Fichier config.yaml pour l'orchestrateur
    251:         orchestrator_config = '''# Configuration de l'orchestrateur
    252: orchestrator:
    253:   name: "SmartContractDevPipeline"
    254:   version: "1.0.0"
    255:   log_level: "INFO"
    256: 
    257: # Configuration des agents
    258: agents:
    259:   architect:
    260:     enabled: true
    261:     module: "agents.architect.agent"
    262:     class: "ArchitectAgent"
    263:     config_path: "agents/architect/config.yaml"
    264:     priority: 1
    265:     
    266:   coder:
    267:     enabled: true
    268:     module: "agents.coder.agent"
    269:     class: "CoderAgent"
    270:     config_path: "agents/coder/config.yaml"
    271:     priority: 2
    272:     
    273:   smart_contract:
    274:     enabled: true
    275:     module: "agents.smart_contract.agent"
    276:     class: "SmartContractAgent"
    277:     config_path: "agents/smart_contract/config.yaml"
    278:     priority: 3
    279:     
    280:   frontend_web3:
    281:     enabled: true
    282:     module: "agents.frontend_web3.agent"
    283:     class: "FrontendWeb3Agent"
    284:     config_path: "agents/frontend_web3/config.yaml"
    285:     priority: 4
    286:     
    287:   tester:
    288:     enabled: true
    289:     module: "agents.tester.agent"
    290:     class: "TesterAgent"
    291:     config_path: "agents/tester/config.yaml"
    292:     priority: 5
    293: 
    294: # Définition des workflows
    295: workflow:
    296:   full_pipeline:
    297:     name: "Pipeline complet de développement"
    298:     description: "Workflow complet du développement d'un smart contract"
    299:     steps:
    300:       - id: "architecture"
    301:         agent: "architect"
    302:         task: "design_architecture"
    303:         parameters:
    304:           project_type: "smart_contract"
    305:           complexity: "medium"
    306:       
    307:       - id: "backend_dev"
    308:         agent: "coder"
    309:         task: "develop_backend"
    310:         parameters:
    311:           language: "python"
    312:           framework: "fastapi"
    313:       
    314:       - id: "smart_contract_dev"
    315:         agent: "smart_contract"
    316:         task: "develop_contract"
    317:         parameters:
    318:           blockchain: "ethereum"
    319:           standard: "ERC20"
    320:       
    321:       - id: "frontend_dev"
    322:         agent: "frontend_web3"
    323:         task: "develop_frontend"
    324:         parameters:
    325:           framework: "nextjs"
    326:           web3_library: "ethers"
    327:       
    328:       - id: "testing"
    329:         agent: "tester"
    330:         task: "run_full_tests"
    331:         parameters:
    332:           test_types: ["unit", "integration", "e2e"]
    333: '''
    334:         
    335:         # Fichier requirements.txt
    336:         requirements = '''# Dépendances de l'orchestrateur
    337: aiohttp>=3.9.0
    338: PyYAML>=6.0
    339: asyncio>=3.4.3
    340: pydantic>=2.5.0
    341: python-dotenv>=1.0.0
    342: web3>=6.0.0
    343: '''
    344:         
    345:         # Créer les fichiers
    346:         files_to_create = {
    347:             "orchestrator.py": orchestrator_py,
    348:             "config.yaml": orchestrator_config,
    349:             "requirements.txt": requirements,
    350:             "__init__.py": "# Orchestrator package\n"
    351:         }
    352:         
    353:         try:
    354:             for filename, content in files_to_create.items():
    355:                 filepath = os.path.join(self.orchestrator_path, filename)
    356:                 if not os.path.exists(filepath):
    357:                     with open(filepath, 'w', encoding='utf-8') as f:
    358:                         f.write(content)
    359:                     logger.info(f"  ✅ {filename} créé")
    360:                 else:
    361:                     logger.info(f"  ⏭️ {filename} existe déjà")
    362:             
    363:             return True
    364:         except Exception as e:
    365:             logger.error(f"❌ Erreur lors de la création de l'orchestrateur: {e}")
    366:             return False
    367:     
    368:     def create_main_agent(self, agent_name: str) -> bool:
    369:         """Crée un agent principal s'il n'existe pas"""
    370:         agent_dir = os.path.join(self.agents_path, agent_name)
    371:         os.makedirs(agent_dir, exist_ok=True)
    372:         
    373:         # Fichier agent.py pour l'agent principal
    374:         agent_py = f'''"""
    375: Agent {agent_name.capitalize()} - Agent principal
    376: """
    377: from typing import Dict, Any, List
    378: import yaml
    379: import logging
    380: from base_agent import BaseAgent
    381: 
    382: class {agent_name.capitalize()}Agent(BaseAgent):
    383:     """Agent spécialisé en {agent_name.replace('_', ' ').title()}"""
    384:     
    385:     def __init__(self, config_path: str):
    386:         super().__init__(config_path)
    387:         self.specialization = "{agent_name}"
    388:         self.sub_agents = {{}}
    389:         self._initialize_sub_agents()
    390:         self.logger.info(f"Agent {{self.agent_id}} initialisé")
    391:     
    392:     def _initialize_sub_agents(self):
    393:         """Initialise les sous-agents spécialisés"""
    394:         try:
    395:             from .sous_agents import *
    396:             
    397:             sub_agent_configs = self.config.get("sub_agents", {{}})
    398:             
    399:             for sub_agent_name, agent_config in sub_agent_configs.items():
    400:                 if agent_config.get("enabled", True):
    401:                     agent_class_name = f"{{sub_agent_name.capitalize().replace('_', '')}}SubAgent"
    402:                     agent_class = globals().get(agent_class_name)
    403:                     
    404:                     if agent_class:
    405:                         sub_agent = agent_class(agent_config.get("config_path", ""))
    406:                         self.sub_agents[sub_agent_name] = sub_agent
    407:                         self.logger.info(f"Sous-agent {{sub_agent_name}} initialisé")
    408:                     else:
    409:                         self.logger.warning(f"Classe non trouvée pour {{sub_agent_name}}")
    410:         
    411:         except ImportError as e:
    412:             self.logger.error(f"Erreur lors de l'import des sous-agents: {{e}}")
    413:         except Exception as e:
    414:             self.logger.error(f"Erreur lors de l'initialisation des sous-agents: {{e}}")
    415:     
    416:     async def execute(self, task_data: Dict[str, Any], workflow_context: Dict[str, Any]) -> Dict[str, Any]:
    417:         """Exécute une tâche"""
    418:         task_type = task_data.get("task_type", "unknown")
    419:         
    420:         # Vérifier si on doit déléguer à un sous-agent
    421:         sub_agent_mapping = self.config.get("sub_agent_mapping", {{}})
    422:         
    423:         for pattern, agent_name in sub_agent_mapping.items():
    424:             if task_type.startswith(pattern):
    425:                 if agent_name in self.sub_agents:
    426:                     self.logger.info(f"Délégation au sous-agent {{agent_name}}")
    427:                     return await self.sub_agents[agent_name].execute(task_data, workflow_context)
    428:         
    429:         # Exécuter localement
    430:         self.logger.info(f"Exécution de la tâche {{task_type}}")
    431:         
    432:         # Implémentation spécifique à l'agent
    433:         result = await self._execute_{agent_name}(task_data, workflow_context)
    434:         
    435:         return {{
    436:             "success": True,
    437:             "agent": "{agent_name}",
    438:             "task": task_type,
    439:             "result": result,
    440:             "sub_agents_used": list(self.sub_agents.keys())
    441:         }}
    442:     
    443:     async def _execute_{agent_name}(self, task_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    444:         """Méthode spécifique à implémenter par chaque agent"""
    445:         # À implémenter selon la spécialisation
    446:         return {{
    447:             "message": f"Tâche exécutée par l'agent {agent_name}",
    448:             "input_data": task_data,
    449:             "context": context
    450:         }}
    451:     
    452:     async def health_check(self) -> Dict[str, Any]:
    453:         """Vérifie la santé de l'agent et de ses sous-agents"""
    454:         status = {{
    455:             "agent": "{agent_name}",
    456:             "status": "healthy",
    457:             "sub_agents": {{}}
    458:         }}
    459:         
    460:         for sub_agent_name, sub_agent in self.sub_agents.items():
    461:             try:
    462:                 sub_health = await sub_agent.health_check()
    463:                 status["sub_agents"][sub_agent_name] = sub_health
    464:             except Exception as e:
    465:                 status["sub_agents"][sub_agent_name] = {{
    466:                     "status": "error",
    467:                     "error": str(e)
    468:                 }}
    469:         
    470:         return status
    471: '''
    472:         
    473:         # Fichier config.yaml pour l'agent principal
    474:         config_yaml = f'''# Configuration de l'agent {agent_name}
    475: agent:
    476:   id: "{agent_name}_01"
    477:   name: "{agent_name.capitalize()} Agent"
    478:   version: "1.0.0"
    479:   description: "Agent spécialisé en {agent_name.replace('_', ' ').title()}"
    480:   
    481:   capabilities:
    482:     - "task_execution"
    483:     - "sub_agent_management"
    484:     - "health_monitoring"
    485:   
    486:   parameters:
    487:     max_concurrent_tasks: 5
    488:     timeout_seconds: 300
    489:     retry_attempts: 3
    490: 
    491: # Sous-agents (à adapter selon l'agent)
    492: sub_agents:{self._generate_sub_agent_config(agent_name)}
    493: 
    494: # Mapping des tâches vers les sous-agents
    495: sub_agent_mapping:{self._generate_sub_agent_mapping(agent_name)}
    496: 
    497: logging:
    498:   level: "INFO"
    499:   file: "logs/{agent_name}.log"
    500:   format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    501: 
    502: api:
    503:   enabled: true
    504:   port: {self._get_agent_port(agent_name)}
    505:   endpoints:
    506:     - "/execute"
    507:     - "/health"
    508:     - "/status"
    509: '''
    510:         
    511:         # Fichiers à créer
    512:         files_to_create = {
    513:             "agent.py": agent_py,
    514:             "config.yaml": config_yaml,
    515:             "__init__.py": f"# {agent_name.capitalize()} Agent package\n",
    516:             "tools.py": f"# Outils pour l'agent {agent_name}\n\nclass {agent_name.capitalize()}Tools:\n    pass\n"
    517:         }
    518:         
    519:         try:
    520:             for filename, content in files_to_create.items():
    521:                 filepath = os.path.join(agent_dir, filename)
    522:                 if not os.path.exists(filepath):
    523:                     with open(filepath, 'w', encoding='utf-8') as f:
    524:                         f.write(content)
    525:                     logger.info(f"  ✅ {agent_name}/{filename} créé")
    526:             
    527:             return True
    528:         except Exception as e:
    529:             logger.error(f"❌ Erreur lors de la création de l'agent {agent_name}: {e}")
    530:             return False
    531:     
    532:     def _generate_sub_agent_config(self, parent_agent: str) -> str:
    533:         """Génère la configuration YAML pour les sous-agents"""
    534:         if parent_agent not in self.agent_structure:
    535:             return ""
    536:         
    537:         config_lines = []
    538:         for sub_agent in self.agent_structure[parent_agent]:
    539:             config_lines.append(f'''
    540:   {sub_agent["name"]}:
    541:     enabled: true
    542:     config_path: "agents/{parent_agent}/sous_agents/{sub_agent['name']}/config.yaml"
    543:     specialization: "{sub_agent['type']}"
    544:     priority: 1''')
    545:         
    546:         return "".join(config_lines)
    547:     
    548:     def _generate_sub_agent_mapping(self, parent_agent: str) -> str:
    549:         """Génère le mapping des tâches pour les sous-agents"""
    550:         mappings = {
    551:             "architect": '''
    552:   "cloud_": "cloud_architect"
    553:   "aws_": "cloud_architect"
    554:   "azure_": "cloud_architect"
    555:   "gcp_": "cloud_architect"
    556:   "blockchain_": "blockchain_architect"
    557:   "web3_": "blockchain_architect"
    558:   "smart_contract_": "blockchain_architect"
    559:   "microservices_": "microservices_architect"
    560:   "service_": "microservices_architect"
    561:   "api_": "microservices_architect"''',
    562:             
    563:             "coder": '''
    564:   "backend_": "backend_coder"
    565:   "server_": "backend_coder"
    566:   "api_": "backend_coder"
    567:   "frontend_": "frontend_coder"
    568:   "ui_": "frontend_coder"
    569:   "react_": "frontend_coder"
    570:   "devops_": "devops_coder"
    571:   "deploy_": "devops_coder"
    572:   "ci_cd_": "devops_coder"''',
    573:             
    574:             "smart_contract": '''
    575:   "solidity_": "solidity_expert"
    576:   "contract_": "solidity_expert"
    577:   "security_": "security_expert"
    578:   "audit_": "security_expert"
    579:   "gas_": "gas_optimizer"
    580:   "optimize_": "gas_optimizer"
    581:   "formal_": "formal_verification"
    582:   "verify_": "formal_verification"''',
    583:             
    584:             "frontend_web3": '''
    585:   "react_": "react_expert"
    586:   "nextjs_": "react_expert"
    587:   "web3_": "web3_integration"
    588:   "wallet_": "web3_integration"
    589:   "ui_": "ui_ux_expert"
    590:   "ux_": "ui_ux_expert"
    591:   "design_": "ui_ux_expert"''',
    592:             
    593:             "tester": '''
    594:   "unit_": "unit_tester"
    595:   "test_unit": "unit_tester"
    596:   "integration_": "integration_tester"
    597:   "test_integration": "integration_tester"
    598:   "e2e_": "e2e_tester"
    599:   "end_to_end": "e2e_tester"
    600:   "fuzz_": "fuzzing_expert"
    601:   "fuzzing_": "fuzzing_expert"'''
    602:         }
    603:         
    604:         return mappings.get(parent_agent, "")
    605:     
    606:     def _get_agent_port(self, agent_name: str) -> int:
    607:         """Retourne un port unique pour chaque agent"""
    608:         port_mapping = {
    609:             "architect": 8001,
    610:             "coder": 8002,
    611:             "smart_contract": 8003,
    612:             "frontend_web3": 8004,
    613:             "tester": 8005,
    614:             "orchestrator": 8000
    615:         }
    616:         return port_mapping.get(agent_name, 8080)
    617:     
    618:     def create_sub_agents(self, parent_agent: str) -> bool:
    619:         """Crée les sous-agents pour un agent parent"""
    620:         if parent_agent not in self.agent_structure:
    621:             return False
    622:         
    623:         logger.info(f"  📁 Création des sous-agents pour {parent_agent}...")
    624:         
    625:         for sub_agent_info in self.agent_structure[parent_agent]:
    626:             sub_agent_name = sub_agent_info["name"]
    627:             sub_agent_dir = os.path.join(
    628:                 self.agents_path, 
    629:                 parent_agent, 
    630:                 "sous_agents", 
    631:                 sub_agent_name
    632:             )
    633:             
    634:             # Créer le dossier
    635:             os.makedirs(sub_agent_dir, exist_ok=True)
    636:             
    637:             # Nom de classe formaté
    638:             class_name = sub_agent_name.replace("_", " ").title().replace(" ", "")
    639:             
    640:             # Fichier agent.py pour le sous-agent
    641:             sub_agent_py = f'''"""
    642: Sous-agent {sub_agent_info['type']}
    643: Spécialisation: {sub_agent_info['type']}
    644: """
    645: from typing import Dict, Any
    646: import yaml
    647: import logging
    648: 
    649: class {class_name}SubAgent:
    650:     """Sous-agent spécialisé en {sub_agent_info['type']}"""
    651:     
    652:     def __init__(self, config_path: str = ""):
    653:         self.config_path = config_path
    654:         self.config = self._load_config()
    655:         self.logger = logging.getLogger(__name__)
    656:         self.agent_id = f"{sub_agent_name}_sub_01"
    657:         
    658:         self.logger.info(f"Sous-agent {{self.agent_id}} initialisé")
    659:     
    660:     def _load_config(self) -> Dict[str, Any]:
    661:         """Charge la configuration"""
    662:         if self.config_path and os.path.exists(self.config_path):
    663:             try:
    664:                 with open(self.config_path, 'r') as f:
    665:                     return yaml.safe_load(f)
    666:             except Exception as e:
    667:                 self.logger.error(f"Erreur de chargement config: {{e}}")
    668:         
    669:         # Configuration par défaut
    670:         return {{
    671:             "agent": {{
    672:                 "name": "{sub_agent_info['type']}",
    673:                 "specialization": "{sub_agent_info['type']}",
    674:                 "version": "1.0.0"
    675:             }},
    676:             "capabilities": ["task_execution", "specialized_operation"]
    677:         }}
    678:     
    679:     async def execute(self, task_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    680:         """Exécute une tâche spécialisée"""
    681:         task_type = task_data.get("task_type", "unknown")
    682:         
    683:         self.logger.info(f"Exécution de la tâche {{task_type}}")
    684:         
    685:         # Implémentation spécifique au sous-agent
    686:         result = await self._execute_specialized(task_data, context)
    687:         
    688:         return {{
    689:             "success": True,
    690:             "sub_agent": "{sub_agent_name}",
    691:             "task": task_type,
    692:             "result": result,
    693:             "specialization": "{sub_agent_info['type']}"
    694:         }}
    695:     
    696:     async def _execute_specialized(self, task_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    697:         """Méthode spécialisée à implémenter"""
    698:         # À implémenter selon la spécialisation
    699:         return {{
    700:             "message": "Tâche exécutée par le sous-agent spécialisé",
    701:             "specialization": "{sub_agent_info['type']}",
    702:             "input": task_data
    703:         }}
    704:     
    705:     async def health_check(self) -> Dict[str, Any]:
    706:         """Vérifie la santé du sous-agent"""
    707:         return {{
    708:             "agent": "{sub_agent_name}",
    709:             "status": "healthy",
    710:             "type": "sub_agent",
    711:             "specialization": "{sub_agent_info['type']}",
    712:             "config_loaded": bool(self.config)
    713:         }}
    714:     
    715:     def get_agent_info(self) -> Dict[str, Any]:
    716:         """Retourne les informations du sous-agent"""
    717:         return {{
    718:             "id": self.agent_id,
    719:             "name": "{sub_agent_info['type']}",
    720:             "type": "sub_agent",
    721:             "parent": "{parent_agent}",
    722:             "specialization": "{sub_agent_info['type']}",
    723:             "version": "1.0.0"
    724:         }}
    725: '''
    726:             
    727:             # Fichier config.yaml pour le sous-agent
    728:             sub_config_yaml = f'''# Configuration du sous-agent {sub_agent_name}
    729: sub_agent:
    730:   id: "{sub_agent_name}_01"
    731:   name: "{sub_agent_info['type']}"
    732:   parent: "{parent_agent}"
    733:   specialization: "{sub_agent_info['type']}"
    734:   version: "1.0.0"
    735: 
    736: capabilities:
    737:   - "specialized_task_execution"
    738:   - "domain_expertise"
    739: 
    740: parameters:
    741:   timeout_seconds: 60
    742:   max_retries: 2
    743: 
    744: logging:
    745:   level: "INFO"
    746: '''
    747:             
    748:             # Fichiers à créer
    749:             sub_files = {
    750:                 "agent.py": sub_agent_py,
    751:                 "config.yaml": sub_config_yaml,
    752:                 "tools.py": f"# Outils pour {sub_agent_info['type']}\n",
    753:                 "__init__.py": f"# {class_name}SubAgent package\n"
    754:             }
    755:             
    756:             try:
    757:                 for filename, content in sub_files.items():
    758:                     filepath = os.path.join(sub_agent_dir, filename)
    759:                     if not os.path.exists(filepath):
    760:                         with open(filepath, 'w', encoding='utf-8') as f:
    761:                             f.write(content)
    762:                 
    763:                 logger.info(f"    ✅ {parent_agent}/{sub_agent_name}")
    764:                 
    765:             except Exception as e:
    766:                 logger.error(f"    ❌ Erreur avec {sub_agent_name}: {e}")
    767:                 return False
    768:         
    769:         return True
    770:     
    771:     def create_init_files(self):
    772:         """Crée les fichiers __init__.py pour l'importation"""
    773:         logger.info("📄 Création des fichiers d'import...")
    774:         
    775:         # Fichier __init__.py principal
    776:         main_init = os.path.join(self.agents_path, "__init__.py")
    777:         with open(main_init, 'w', encoding='utf-8') as f:
    778:             f.write('''# Package agents
    779: from .architect.agent import ArchitectAgent
    780: from .coder.agent import CoderAgent
    781: from .smart_contract.agent import SmartContractAgent
    782: from .frontend_web3.agent import FrontendWeb3Agent
    783: from .tester.agent import TesterAgent
    784: 
    785: __all__ = [
    786:     "ArchitectAgent",
    787:     "CoderAgent",
    788:     "SmartContractAgent",
    789:     "FrontendWeb3Agent",
    790:     "TesterAgent"
    791: ]
    792: ''')
    793:         
    794:         # Fichiers __init__.py pour les sous-agents de chaque parent
    795:         for parent_agent in self.agent_structure.keys():
    796:             init_dir = os.path.join(self.agents_path, parent_agent, "sous_agents")
    797:             init_file = os.path.join(init_dir, "__init__.py")
    798:             
    799:             # Générer les imports dynamiquement
    800:             imports = ["# Import des sous-agents\n"]
    801:             all_list = []
    802:             
    803:             for sub_agent in self.agent_structure[parent_agent]:
    804:                 class_name = sub_agent["name"].replace("_", " ").title().replace(" ", "") + "SubAgent"
    805:                 imports.append(f"from .{sub_agent['name']}.agent import {class_name}")
    806:                 all_list.append(f'"{class_name}"')
    807:             
    808:             imports.append(f"\n__all__ = [{', '.join(all_list)}]")
    809:             
    810:             with open(init_file, 'w', encoding='utf-8') as f:
    811:                 f.write("\n".join(imports))
    812:             
    813:             logger.info(f"  ✅ {parent_agent}/sous_agents/__init__.py")
    814:     
    815:     def create_base_agent(self):
    816:         """Crée la classe BaseAgent si elle n'existe pas"""
    817:         base_agent_path = os.path.join(self.project_root, "base_agent.py")
    818:         
    819:         if not os.path.exists(base_agent_path):
    820:             base_agent_code = '''"""
    821: Classe de base pour tous les agents
    822: """
    823: from abc import ABC, abstractmethod
    824: from typing import Dict, Any, Optional
    825: import yaml
    826: import logging
    827: import uuid
    828: from datetime import datetime
    829: 
    830: class BaseAgent(ABC):
    831:     """Classe abstraite de base pour tous les agents"""
    832:     
    833:     def __init__(self, config_path: str = ""):
    834:         self.config_path = config_path
    835:         self.config = self._load_config()
    836:         self.agent_id = f"{self.__class__.__name__.lower()}_{uuid.uuid4().hex[:8]}"
    837:         self.logger = logging.getLogger(self.__class__.__name__)
    838:         self.created_at = datetime.now()
    839:         self.last_activity = datetime.now()
    840:         
    841:         self.logger.info(f"Agent {self.agent_id} initialisé")
    842:     
    843:     def _load_config(self) -> Dict[str, Any]:
    844:         """Charge la configuration depuis un fichier YAML"""
    845:         if self.config_path and os.path.exists(self.config_path):
    846:             try:
    847:                 with open(self.config_path, 'r') as f:
    848:                     return yaml.safe_load(f) or {}
    849:             except Exception as e:
    850:                 self.logger.error(f"Erreur de chargement de la config: {e}")
    851:         
    852:         # Configuration par défaut
    853:         return {
    854:             "agent": {
    855:                 "name": self.__class__.__name__,
    856:                 "version": "1.0.0"
    857:             },
    858:             "logging": {
    859:                 "level": "INFO"
    860:             }
    861:         }
    862:     
    863:     @abstractmethod
    864:     async def execute(self, task_data: Dict[str, Any], workflow_context: Dict[str, Any]) -> Dict[str, Any]:
    865:         """Méthode abstraite pour exécuter une tâche"""
    866:         pass
    867:     
    868:     async def health_check(self) -> Dict[str, Any]:
    869:         """Vérifie la santé de l'agent"""
    870:         return {
    871:             "agent_id": self.agent_id,
    872:             "status": "healthy",
    873:             "uptime": str(datetime.now() - self.created_at),
    874:             "last_activity": self.last_activity.isoformat(),
    875:             "config_loaded": bool(self.config)
    876:         }
    877:     
    878:     def get_agent_info(self) -> Dict[str, Any]:
    879:         """Retourne les informations de l'agent"""
    880:         return {
    881:             "id": self.agent_id,
    882:             "name": self.config.get("agent", {}).get("name", self.__class__.__name__),
    883:             "class": self.__class__.__name__,
    884:             "created_at": self.created_at.isoformat(),
    885:             "config_path": self.config_path
    886:         }
    887:     
    888:     def update_activity(self):
    889:         """Met à jour le timestamp de la dernière activité"""
    890:         self.last_activity = datetime.now()
    891:     
    892:     async def validate_input(self, task_data: Dict[str, Any]) -> bool:
    893:         """Valide les données d'entrée"""
    894:         required_fields = self.config.get("required_fields", [])
    895:         
    896:         for field in required_fields:
    897:             if field not in task_data:
    898:                 self.logger.error(f"Champ requis manquant: {field}")
    899:                 return False
    900:         
    901:         return True
    902: '''
    903:             
    904:             with open(base_agent_path, 'w', encoding='utf-8') as f:
    905:                 f.write(base_agent_code)
    906:             
    907:             logger.info("✅ base_agent.py créé")
    908:     
    909:     def create_requirements_file(self):
    910:         """Crée le fichier requirements.txt global"""
    911:         requirements_path = os.path.join(self.project_root, "requirements.txt")
    912:         
    913:         requirements = '''# Dépendances du projet SmartContractDevPipeline
    914: 
    915: # Core
    916: python>=3.9
    917: PyYAML>=6.0
    918: pydantic>=2.5.0
    919: python-dotenv>=1.0.0
    920: 
    921: # Async
    922: aiohttp>=3.9.0
    923: asyncio>=3.4.3
    924: 
    925: # Web3 & Blockchain
    926: web3>=6.0.0
    927: eth-account>=0.11.0
    928: eth-typing>=3.0.0
    929: cryptography>=41.0.0
    930: 
    931: # Development
    932: black>=23.0.0
    933: pytest>=7.0.0
    934: pytest-asyncio>=0.21.0
    935: mypy>=1.0.0
    936: 
    937: # API
    938: fastapi>=0.104.0
    939: uvicorn>=0.24.0
    940: httpx>=0.25.0
    941: 
    942: # Utils
    943: jinja2>=3.1.0
    944: markdown>=3.5.0
    945: rich>=13.0.0
    946: '''
    947:         
    948:         with open(requirements_path, 'w', encoding='utf-8') as f:
    949:             f.write(requirements)
    950:         
    951:         logger.info("✅ requirements.txt créé")
    952:     
    953:     def create_docker_compose(self):
    954:         """Crée un docker-compose.yml pour déployer tous les agents"""
    955:         docker_path = os.path.join(self.project_root, "docker-compose.yml")
    956:         
    957:         docker_compose = '''version: '3.8'
    958: 
    959: services:
    960:   orchestrator:
    961:     build: 
    962:       context: .
    963:       dockerfile: Dockerfile
    964:     container_name: smartcontract-orchestrator
    965:     ports:
    966:       - "8000:8000"
    967:     volumes:
    968:       - ./orchestrator:/app/orchestrator
    969:       - ./agents:/app/agents
    970:       - ./logs:/app/logs
    971:     environment:
    972:       - LOG_LEVEL=INFO
    973:       - ENVIRONMENT=production
    974:     command: python orchestrator/orchestrator.py
    975:     restart: unless-stopped
    976:     networks:
    977:       - agent-network
    978: 
    979:   architect:
    980:     build:
    981:       context: .
    982:       dockerfile: Dockerfile.agent
    983:     container_name: agent-architect
    984:     ports:
    985:       - "8001:8001"
    986:     volumes:
    987:       - ./agents/architect:/app/agent
    988:       - ./logs:/app/logs
    989:     environment:
    990:       - AGENT_TYPE=architect
    991:       - PARENT_ORCHESTRATOR=http://orchestrator:8000
    992:     depends_on:
    993:       - orchestrator
    994:     restart: unless-stopped
    995:     networks:
    996:       - agent-network
    997: 
    998:   coder:
    999:     build:
   1000:       context: .
   1001:       dockerfile: Dockerfile.agent
   1002:     container_name: agent-coder
   1003:     ports:
   1004:       - "8002:8002"
   1005:     volumes:
   1006:       - ./agents/coder:/app/agent
   1007:       - ./logs:/app/logs
   1008:     environment:
   1009:       - AGENT_TYPE=coder
   1010:       - PARENT_ORCHESTRATOR=http://orchestrator:8000
   1011:     depends_on:
   1012:       - orchestrator
   1013:     restart: unless-stopped
   1014:     networks:
   1015:       - agent-network
   1016: 
   1017:   smart_contract:
   1018:     build:
   1019:       context: .
   1020:       dockerfile: Dockerfile.agent
   1021:     container_name: agent-smart-contract
   1022:     ports:
   1023:       - "8003:8003"
   1024:     volumes:
   1025:       - ./agents/smart_contract:/app/agent
   1026:       - ./logs:/app/logs
   1027:     environment:
   1028:       - AGENT_TYPE=smart_contract
   1029:       - PARENT_ORCHESTRATOR=http://orchestrator:8000
   1030:     depends_on:
   1031:       - orchestrator
   1032:     restart: unless-stopped
   1033:     networks:
   1034:       - agent-network
   1035: 
   1036:   frontend_web3:
   1037:     build:
   1038:       context: .
   1039:       dockerfile: Dockerfile.agent
   1040:     container_name: agent-frontend-web3
   1041:     ports:
   1042:       - "8004:8004"
   1043:     volumes:
   1044:       - ./agents/frontend_web3:/app/agent
   1045:       - ./logs:/app/logs
   1046:     environment:
   1047:       - AGENT_TYPE=frontend_web3
   1048:       - PARENT_ORCHESTRATOR=http://orchestrator:8000
   1049:     depends_on:
   1050:       - orchestrator
   1051:     restart: unless-stopped
   1052:     networks:
   1053:       - agent-network
   1054: 
   1055:   tester:
   1056:     build:
   1057:       context: .
   1058:       dockerfile: Dockerfile.agent
   1059:     container_name: agent-tester
   1060:     ports:
   1061:       - "8005:8005"
   1062:     volumes:
   1063:       - ./agents/tester:/app/agent
   1064:       - ./logs:/app/logs
   1065:     environment:
   1066:       - AGENT_TYPE=tester
   1067:       - PARENT_ORCHESTRATOR=http://orchestrator:8000
   1068:     depends_on:
   1069:       - orchestrator
   1070:     restart: unless-stopped
   1071:     networks:
   1072:       - agent-network
   1073: 
   1074: networks:
   1075:   agent-network:
   1076:     driver: bridge
   1077: 
   1078: volumes:
   1079:   agent-data:
   1080:     driver: local
   1081: '''
   1082:         
   1083:         with open(docker_path, 'w', encoding='utf-8') as f:
   1084:             f.write(docker_compose)
   1085:         
   1086:         logger.info("✅ docker-compose.yml créé")
   1087:     
   1088:     # CORRECTION BUG PRINCIPAL : Méthode create_readme simplifiée
   1089:     def create_readme(self):
   1090:         """Crée un fichier README.md avec des instructions"""
   1091:         readme_path = os.path.join(self.project_root, "README.md")
   1092:         
   1093:         # Construction simple sans f-string multiligne complexe
   1094:         project_name = os.path.basename(self.project_root)
   1095:         
   1096:         content = f"# SmartContractDevPipeline\n\n"
   1097:         content += f"Pipeline de développement automatisé pour smart contracts avec agents IA.\n\n"
   1098:         content += f"## 📁 Structure du projet\n\n"
   1099:         content += f"```\n"
   1100:         content += f"{project_name}/\n"
   1101:         content += f"├── agents/                    # Agents principaux\n"
   1102:         content += f"│   ├── architect/            # Agent architecte\n"
   1103:         content += f"│   │   ├── sous_agents/      # Sous-agents spécialisés\n"
   1104:         content += f"│   │   │   ├── cloud_architect/\n"
   1105:         content += f"│   │   │   ├── blockchain_architect/\n"
   1106:         content += f"│   │   │   └── microservices_architect/\n"
   1107:         content += f"│   │   ├── agent.py         # Agent principal\n"
   1108:         content += f"│   │   └── config.yaml      # Configuration\n"
   1109:         content += f"│   ├── coder/               # Agent développeur\n"
   1110:         content += f"│   ├── smart_contract/      # Agent smart contract\n"
   1111:         content += f"│   ├── frontend_web3/       # Agent frontend Web3\n"
   1112:         content += f"│   └── tester/              # Agent testeur\n"
   1113:         content += f"├── orchestrator/            # Orchestrateur principal\n"
   1114:         content += f"│   ├── orchestrator.py      # Code de l'orchestrateur\n"
   1115:         content += f"│   └── config.yaml         # Configuration globale\n"
   1116:         content += f"├── base_agent.py           # Classe de base pour tous les agents\n"
   1117:         content += f"├── requirements.txt        # Dépendances Python\n"
   1118:         content += f"├── docker-compose.yml      # Déploiement Docker\n"
   1119:         content += f"└── README.md              # Ce fichier\n"
   1120:         content += f"```\n\n"
   1121:         content += f"## 🚀 Démarrage rapide\n\n"
   1122:         content += f"### 1. Installation des dépendances\n\n"
   1123:         content += f"```bash\n"
   1124:         content += f"pip install -r requirements.txt\n"
   1125:         content += f"```\n\n"
   1126:         content += f"### 2. Déploiement des agents\n\n"
   1127:         content += f"```bash\n"
   1128:         content += f"python deploy_pipeline.py\n"
   1129:         content += f"```\n\n"
   1130:         content += f"Options disponibles:\n"
   1131:         content += f"- `--path /chemin/vers/projet` : Chemin personnalisé du projet\n"
   1132:         content += f"- `--force` : Forcer le redéploiement complet\n"
   1133:         content += f"- `--verbose` : Mode détaillé\n\n"
   1134:         content += f"### 3. Tester l'orchestrateur\n\n"
   1135:         content += f"```bash\n"
   1136:         content += f"cd orchestrator\n"
   1137:         content += f"python orchestrator.py --test\n"
   1138:         content += f"```\n\n"
   1139:         content += f"### 4. Exécuter un workflow\n\n"
   1140:         content += f"```bash\n"
   1141:         content += f"python orchestrator.py --workflow full_pipeline\n"
   1142:         content += f"```\n\n"
   1143:         content += f"## 🔧 Agents et sous-agents\n\n"
   1144:         content += f"### Architecte (3 sous-agents)\n"
   1145:         content += f"- Cloud Architect\n"
   1146:         content += f"- Blockchain Architect\n"
   1147:         content += f"- Microservices Architect\n\n"
   1148:         content += f"### Développeur (3 sous-agents)\n"
   1149:         content += f"- Backend Developer\n"
   1150:         content += f"- Frontend Developer\n"
   1151:         content += f"- DevOps Engineer\n\n"
   1152:         content += f"### Smart Contract (4 sous-agents)\n"
   1153:         content += f"- Solidity Expert\n"
   1154:         content += f"- Security Expert\n"
   1155:         content += f"- Gas Optimizer\n"
   1156:         content += f"- Formal Verification\n\n"
   1157:         content += f"### Frontend Web3 (3 sous-agents)\n"
   1158:         content += f"- React/Next.js Expert\n"
   1159:         content += f"- Web3 Integration\n"
   1160:         content += f"- UI/UX Designer\n\n"
   1161:         content += f"### Testeur (4 sous-agents)\n"
   1162:         content += f"- Unit Tester\n"
   1163:         content += f"- Integration Tester\n"
   1164:         content += f"- E2E Tester\n"
   1165:         content += f"- Fuzzing Expert\n\n"
   1166:         content += f"## 🐛 Dépannage\n\n"
   1167:         content += f"### Problèmes d'import\n"
   1168:         content += f"```bash\n"
   1169:         content += f"export PYTHONPATH=\"$PYTHONPATH:{self.project_root}\"\n"
   1170:         content += f"```\n\n"
   1171:         content += f"Ou exécuter depuis la racine du projet:\n"
   1172:         content += f"```bash\n"
   1173:         content += f"cd {self.project_root}\n"
   1174:         content += f"python deploy_pipeline.py\n"
   1175:         content += f"```\n\n"
   1176:         content += f"## 📝 Personnalisation\n\n"
   1177:         content += f"1. Modifier les configurations dans `agents/*/config.yaml`\n"
   1178:         content += f"2. Ajouter de nouveaux sous-agents dans `deploy_pipeline.py`\n"
   1179:         content += f"3. Créer de nouveaux workflows dans `orchestrator/config.yaml`\n\n"
   1180:         content += f"## 📄 Licence\n\n"
   1181:         content += f"Projet SmartContractDevPipeline - Usage interne\n"
   1182:         
   1183:         try:
   1184:             with open(readme_path, 'w', encoding='utf-8') as f:
   1185:                 f.write(content)
   1186:             
   1187:             logger.info("✅ README.md créé")
   1188:             return True
   1189:         except Exception as e:
   1190:             logger.error(f"❌ Erreur lors de la création du README: {e}")
   1191:             return False
   1192:     
   1193:     async def deploy_all(self, force_redeploy: bool = False) -> Dict[str, Any]:
   1194:         """Déploie tous les composants du pipeline"""
   1195:         logger.info("🚀 Déploiement du SmartContractDevPipeline")
   1196:         logger.info(f"📂 Chemin du projet: {self.project_root}")
   1197:         
   1198:         # Vérifier l'état actuel
   1199:         deployment_status = self.check_existing_deployment()
   1200:         
   1201:         if not force_redeploy:
   1202:             logger.info("🔍 Vérification des composants existants...")
   1203:             
   1204:             # Afficher le statut
   1205:             for component, status in deployment_status.items():
   1206:                 if isinstance(status, dict):
   1207:                     for sub, sub_status in status.items():
   1208:                         if sub_status:
   1209:                             logger.info(f"  ✅ {component}/{sub} déjà déployé")
   1210:                 elif status:
   1211:                     logger.info(f"  ✅ {component} déjà déployé")
   1212:         
   1213:         # Créer la structure de base
   1214:         os.makedirs(self.project_root, exist_ok=True)
   1215:         os.makedirs(self.agents_path, exist_ok=True)
   1216:         os.makedirs(self.orchestrator_path, exist_ok=True)
   1217:         
   1218:         results = {
   1219:             "orchestrator": False,
   1220:             "main_agents": {},
   1221:             "sub_agents": {},
   1222:             "base_files": False
   1223:         }
   1224:         
   1225:         # 1. Créer la classe BaseAgent
   1226:         self.create_base_agent()
   1227:         results["base_files"] = True
   1228:         
   1229:         # 2. Créer l'orchestrateur (seulement si pas déjà fait ou force)
   1230:         if not deployment_status["orchestrator"] or force_redeploy:
   1231:             results["orchestrator"] = self.create_orchestrator()
   1232:         else:
   1233:             results["orchestrator"] = True
   1234:             logger.info("⏭️ Orchestrateur déjà déployé")
   1235:         
   1236:         # 3. Créer les agents principaux
   1237:         logger.info("\n👥 Déploiement des agents principaux...")
   1238:         for agent_name in self.agent_structure.keys():
   1239:             if not deployment_status["main_agents"].get(agent_name, False) or force_redeploy:
   1240:                 success = self.create_main_agent(agent_name)
   1241:                 results["main_agents"][agent_name] = success
   1242:                 
   1243:                 # Créer les sous-agents
   1244:                 if success:
   1245:                     sub_success = self.create_sub_agents(agent_name)
   1246:                     results["sub_agents"][agent_name] = sub_success
   1247:             else:
   1248:                 results["main_agents"][agent_name] = True
   1249:                 results["sub_agents"][agent_name] = True
   1250:                 logger.info(f"⏭️ Agent {agent_name} déjà déployé")
   1251:         
   1252:         # 4. Créer les fichiers d'import
   1253:         self.create_init_files()
   1254:         
   1255:         # 5. Créer le fichier requirements global
   1256:         self.create_requirements_file()
   1257:         
   1258:         # 6. Optionnel: créer docker-compose
   1259:         self.create_docker_compose()
   1260:         
   1261:         # 7. Créer README (CORRIGÉ)
   1262:         self.create_readme()
   1263:         
   1264:         # Résumé
   1265:         logger.info("\n" + "="*50)
   1266:         logger.info("📊 RÉSUMÉ DU DÉPLOIEMENT")
   1267:         logger.info("="*50)
   1268:         
   1269:         total_main = sum(1 for v in results["main_agents"].values() if v)
   1270:         total_sub = sum(1 for v in results["sub_agents"].values() if v)
   1271:         
   1272:         logger.info(f"Orchestrateur: {'✅' if results['orchestrator'] else '❌'}")
   1273:         logger.info(f"Agents principaux: {total_main}/{len(self.agent_structure)}")
   1274:         logger.info(f"Groupes de sous-agents: {total_sub}/{len(self.agent_structure)}")
   1275:         
   1276:         # Calculer le nombre total de sous-agents
   1277:         total_sub_agents = sum(len(subs) for subs in self.agent_structure.values())
   1278:         logger.info(f"Sous-agents individuels: {total_sub_agents} créés")
   1279:         
   1280:         logger.info(f"\n📁 Structure créée dans: {self.project_root}")
   1281:         logger.info("🎉 Déploiement terminé!")
   1282:         
   1283:         # Instructions
   1284:         logger.info("\n" + "="*50)
   1285:         logger.info("📋 PROCHAINES ÉTAPES")
   1286:         logger.info("="*50)
   1287:         logger.info("1. Installer les dépendances:")
   1288:         logger.info("   pip install -r requirements.txt")
   1289:         logger.info("\n2. Tester l'orchestrateur:")
   1290:         logger.info("   cd orchestrator && python orchestrator.py")
   1291:         logger.info("\n3. Démarrer avec Docker (optionnel):")
   1292:         logger.info("   docker-compose up -d")
   1293:         logger.info("\n4. Vérifier la santé:")
   1294:         logger.info("   curl http://localhost:8000/health")
   1295:         
   1296:         return results
   1297: 
   1298: 
   1299: # Point d'entrée principal
   1300: async def main():
   1301:     """Fonction principale"""
   1302:     import argparse
   1303:     
   1304:     parser = argparse.ArgumentParser(description="Déploiement du SmartContractDevPipeline")
   1305:     parser.add_argument("--path", "-p", type=str, default=None,
   1306:                        help="Chemin du projet (défaut: ~/Projects/SmartContractPipeline)")
   1307:     parser.add_argument("--force", "-f", action="store_true",
   1308:                        help="Forcer le redéploiement même si les composants existent")
   1309:     parser.add_argument("--verbose", "-v", action="store_true",
   1310:                        help="Mode verbeux")
   1311:     
   1312:     args = parser.parse_args()
   1313:     
   1314:     # Configurer le logging
   1315:     if args.verbose:
   1316:         logging.getLogger().setLevel(logging.DEBUG)
   1317:     
   1318:     # Créer et exécuter le déployeur
   1319:     deployer = AgentDeployer(args.path)
   1320:     
   1321:     try:
   1322:         await deployer.deploy_all(force_redeploy=args.force)
   1323:     except KeyboardInterrupt:
   1324:         logger.info("\n⚠️  Déploiement interrompu par l'utilisateur")
   1325:     except Exception as e:
   1326:         logger.error(f"❌ Erreur lors du déploiement: {e}")
   1327:         sys.exit(1)
   1328: 
   1329: if __name__ == "__main__":
   1330:     asyncio.run(main())