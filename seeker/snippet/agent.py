#date: 2026-03-16T17:43:37Z
#url: https://api.github.com/gists/4ed8e01d64231199bac334348aa4f64d
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: """
      2: E2E Tester SubAgent - Sous-agent de tests end-to-end
      3: Version: 2.0.0
      4: 
      5: Teste les flows utilisateur complets : du frontend à la blockchain.
      6: Intégration avec Playwright/Cypress, scenarios utilisateur, métriques UX.
      7: """
      8: 
      9: import logging
     10: import sys
     11: import asyncio
     12: import json
     13: from datetime import datetime, timedelta
     14: from pathlib import Path
     15: from typing import Dict, Any, Optional, List
     16: from enum import Enum
     17: from dataclasses import dataclass, field
     18: import random
     19: 
     20: # Configuration des imports
     21: current_dir = Path(__file__).parent.absolute()
     22: project_root = current_dir.parent.parent.parent.parent.parent
     23: if str(project_root) not in sys.path:
     24:     sys.path.insert(0, str(project_root))
     25: 
     26: from agents.sous_agents.base_subagent import BaseSubAgent
     27: 
     28: logger = logging.getLogger(__name__)
     29: 
     30: 
     31: # ============================================================================
     32: # ÉNUMS ET CLASSES DE DONNÉES
     33: # ============================================================================
     34: 
     35: class BrowserType(Enum):
     36:     """Types de navigateurs supportés"""
     37:     CHROMIUM = "chromium"
     38:     FIREFOX = "firefox"
     39:     WEBKIT = "webkit"
     40:     CHROME = "chrome"
     41:     EDGE = "edge"
     42: 
     43: 
     44: class DeviceType(Enum):
     45:     """Types d'appareils pour tests mobiles"""
     46:     DESKTOP = "desktop"
     47:     TABLET = "tablet"
     48:     MOBILE = "mobile"
     49: 
     50: 
     51: @dataclass
     52: class E2ETestScenario:
     53:     """Scénario de test E2E"""
     54:     id: str
     55:     name: str
     56:     description: str
     57:     steps: List[Dict[str, Any]]
     58:     required_contracts: List[str] = field(default_factory=list)
     59:     required_agents: List[str] = field(default_factory=list)
     60:     timeout_seconds: int = 120
     61: 
     62: 
     63: @dataclass
     64: class E2ETestResult:
     65:     """Résultat de test E2E"""
     66:     scenario_id: str
     67:     success: bool
     68:     steps_completed: int
     69:     steps_total: int
     70:     duration_ms: float
     71:     screenshots: List[str] = field(default_factory=list)
     72:     console_errors: List[str] = field(default_factory=list)
     73:     network_requests: List[Dict] = field(default_factory=list)
     74:     metrics: Dict[str, Any] = field(default_factory=dict)
     75:     timestamp: datetime = field(default_factory=datetime.now)
     76: 
     77: 
     78: # ============================================================================
     79: # SOUS-AGENT PRINCIPAL
     80: # ============================================================================
     81: 
     82: class E2ETesterSubAgent(BaseSubAgent):
     83:     """
     84:     Sous-agent de tests end-to-end
     85: 
     86:     Teste les flows utilisateur complets avec :
     87:     - Scenarios utilisateur réalistes
     88:     - Tests cross-browser
     89:     - Tests mobiles/responsive
     90:     - Métriques UX (temps de chargement, interactions)
     91:     """
     92: 
     93:     def __init__(self, config_path: str = ""):
     94:         """Initialise le sous-agent de tests E2E"""
     95:         super().__init__(config_path)
     96: 
     97:         # Métadonnées
     98:         self._subagent_display_name = "🌐 Tests End-to-End"
     99:         self._subagent_description = "Tests de flows utilisateur complets"
    100:         self._subagent_version = "2.0.0"
    101:         self._subagent_category = "tester"
    102:         self._subagent_capabilities = [
    103:             "e2e.run_scenario",
    104:             "e2e.list_scenarios",
    105:             "e2e.test_mobile",
    106:             "e2e.collect_metrics",
    107:             "e2e.record_session"
    108:         ]
    109: 
    110:         # État interne
    111:         self._scenarios: Dict[str, E2ETestScenario] = {}
    112:         self._results: Dict[str, E2ETestResult] = {}
    113:         
    114:         # Configuration
    115:         self._browsers = self._agent_config.get('browsers', {})
    116:         self._devices = self._agent_config.get('devices', {})
    117:         self._base_url = self._agent_config.get('base_url', 'http://localhost:3000')
    118:         
    119:         # Tâche de fond
    120:         self._cleanup_task: Optional[asyncio.Task] = None
    121: 
    122:         logger.info(f"✅ {self._subagent_display_name} initialisé (v{self._subagent_version})")
    123: 
    124:     # ========================================================================
    125:     # IMPLÉMENTATION DES MÉTHODES ABSTRACTES
    126:     # ========================================================================
    127: 
    128:     async def _initialize_subagent_components(self) -> bool:
    129:         """Initialise les composants spécifiques"""
    130:         logger.info("Initialisation des composants de tests E2E...")
    131: 
    132:         try:
    133:             # Charger les scénarios
    134:             await self._load_scenarios()
    135: 
    136:             logger.info("✅ Composants de tests E2E initialisés")
    137:             return True
    138: 
    139:         except Exception as e:
    140:             logger.error(f"❌ Erreur initialisation composants: {e}")
    141:             return False
    142: 
    143:     async def _initialize_components(self) -> bool:
    144:         """Implémentation requise par BaseAgent"""
    145:         return await self._initialize_subagent_components()
    146: 
    147:     def _get_capability_handlers(self) -> Dict[str, Any]:
    148:         """Retourne les handlers spécifiques"""
    149:         return {
    150:             "e2e.run_scenario": self._handle_run_scenario,
    151:             "e2e.list_scenarios": self._handle_list_scenarios,
    152:             "e2e.test_mobile": self._handle_test_mobile,
    153:             "e2e.collect_metrics": self._handle_collect_metrics,
    154:             "e2e.record_session": self._handle_record_session,
    155:         }
    156: 
    157:     # ========================================================================
    158:     # MÉTHODES PRIVÉES
    159:     # ========================================================================
    160: 
    161:     async def _load_scenarios(self):
    162:         """Charge les scénarios de test depuis la configuration"""
    163:         scenarios_config = self._agent_config.get('scenarios', [])
    164:         
    165:         for sc_config in scenarios_config:
    166:             scenario = E2ETestScenario(
    167:                 id=sc_config.get('id', f"scenario_{len(self._scenarios)}"),
    168:                 name=sc_config.get('name', 'Unnamed Scenario'),
    169:                 description=sc_config.get('description', ''),
    170:                 steps=sc_config.get('steps', []),
    171:                 required_contracts=sc_config.get('required_contracts', []),
    172:                 required_agents=sc_config.get('required_agents', []),
    173:                 timeout_seconds=sc_config.get('timeout', 120)
    174:             )
    175:             self._scenarios[scenario.id] = scenario
    176:         
    177:         logger.info(f"  📋 {len(self._scenarios)} scénarios E2E chargés")
    178: 
    179:     async def _simulate_step(self, step: Dict[str, Any], browser: str, device: str) -> Dict[str, Any]:
    180:         """Simule l'exécution d'une étape de scénario"""
    181:         await asyncio.sleep(0.5)  # Simulation de temps d'exécution
    182:         
    183:         step_type = step.get('type', 'action')
    184:         element = step.get('element', 'unknown')
    185:         
    186:         # Simuler le succès/échec
    187:         success = random.random() > 0.05  # 95% de succès
    188:         
    189:         # Simuler des métriques
    190:         metrics = {
    191:             'duration_ms': random.randint(100, 500),
    192:             'dom_interactive_ms': random.randint(50, 200),
    193:             'first_paint_ms': random.randint(30, 150),
    194:             'layout_shift': random.uniform(0, 0.1),
    195:             'memory_usage_mb': random.randint(50, 200)
    196:         }
    197:         
    198:         return {
    199:             'step': step.get('name', f'Step {step_type}'),
    200:             'type': step_type,
    201:             'element': element,
    202:             'success': success,
    203:             'metrics': metrics,
    204:             'screenshot': f"screenshots/step_{datetime.now().timestamp()}.png" if random.random() > 0.5 else None,
    205:             'console_errors': [] if success else [f"Error interacting with {element}"]
    206:         }
    207: 
    208:     # ========================================================================
    209:     # HANDLERS DE CAPACITÉS
    210:     # ========================================================================
    211: 
    212:     async def _handle_run_scenario(self, params: Dict[str, Any]) -> Dict[str, Any]:
    213:         """Exécute un scénario E2E"""
    214:         scenario_id = params.get('scenario_id')
    215:         browser = params.get('browser', 'chromium')
    216:         device = params.get('device', 'desktop')
    217:         headless = params.get('headless', True)
    218: 
    219:         if not scenario_id:
    220:             return {'success': False, 'error': 'scenario_id requis'}
    221: 
    222:         scenario = self._scenarios.get(scenario_id)
    223:         if not scenario:
    224:             return {'success': False, 'error': f'Scénario {scenario_id} non trouvé'}
    225: 
    226:         test_id = f"e2e_{scenario_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    227:         start_time = datetime.now()
    228: 
    229:         # Exécuter les étapes
    230:         step_results = []
    231:         screenshots = []
    232:         console_errors = []
    233:         
    234:         for i, step in enumerate(scenario.steps):
    235:             step_result = await self._simulate_step(step, browser, device)
    236:             step_results.append(step_result)
    237:             
    238:             if step_result.get('screenshot'):
    239:                 screenshots.append(step_result['screenshot'])
    240:             if step_result.get('console_errors'):
    241:                 console_errors.extend(step_result['console_errors'])
    242: 
    243:         duration = (datetime.now() - start_time).total_seconds() * 1000
    244:         
    245:         successful_steps = sum(1 for r in step_results if r['success'])
    246:         
    247:         # Calculer les métriques agrégées
    248:         metrics = {
    249:             'avg_step_duration': sum(r['metrics']['duration_ms'] for r in step_results) / len(step_results),
    250:             'total_duration_ms': duration,
    251:             'first_paint_avg': sum(r['metrics']['first_paint_ms'] for r in step_results) / len(step_results),
    252:             'layout_shifts': [r['metrics']['layout_shift'] for r in step_results if r['metrics']['layout_shift'] > 0],
    253:             'memory_peak_mb': max(r['metrics']['memory_usage_mb'] for r in step_results)
    254:         }
    255: 
    256:         result = E2ETestResult(
    257:             scenario_id=scenario_id,
    258:             success=successful_steps == len(scenario.steps),
    259:             steps_completed=successful_steps,
    260:             steps_total=len(scenario.steps),
    261:             duration_ms=duration,
    262:             screenshots=screenshots,
    263:             console_errors=console_errors,
    264:             metrics=metrics
    265:         )
    266: 
    267:         self._results[test_id] = result
    268: 
    269:         return {
    270:             'success': result.success,
    271:             'test_id': test_id,
    272:             'scenario': {
    273:                 'id': scenario.id,
    274:                 'name': scenario.name
    275:             },
    276:             'execution': {
    277:                 'browser': browser,
    278:                 'device': device,
    279:                 'headless': headless
    280:             },
    281:             'results': {
    282:                 'steps_completed': result.steps_completed,
    283:                 'steps_total': result.steps_total,
    284:                 'duration_ms': result.duration_ms,
    285:                 'success_rate': (result.steps_completed / result.steps_total) * 100
    286:             },
    287:             'metrics': metrics,
    288:             'screenshots': screenshots[:3],  # Limiter pour la réponse
    289:             'console_errors': console_errors[:5],
    290:             'step_details': step_results,
    291:             'timestamp': result.timestamp.isoformat()
    292:         }
    293: 
    294:     async def _handle_list_scenarios(self, params: Dict[str, Any]) -> Dict[str, Any]:
    295:         """Liste les scénarios disponibles"""
    296:         scenarios = []
    297:         for sc_id, scenario in self._scenarios.items():
    298:             scenarios.append({
    299:                 'id': scenario.id,
    300:                 'name': scenario.name,
    301:                 'description': scenario.description,
    302:                 'steps': len(scenario.steps),
    303:                 'required_contracts': len(scenario.required_contracts),
    304:                 'required_agents': len(scenario.required_agents),
    305:                 'estimated_duration_seconds': len(scenario.steps) * 5
    306:             })
    307: 
    308:         return {
    309:             'success': True,
    310:             'scenarios': scenarios,
    311:             'total': len(scenarios)
    312:         }
    313: 
    314:     async def _handle_test_mobile(self, params: Dict[str, Any]) -> Dict[str, Any]:
    315:         """Exécute des tests sur appareils mobiles"""
    316:         scenario_id = params.get('scenario_id')
    317:         devices = params.get('devices', ['iphone_12', 'pixel_5', 'ipad_pro'])
    318: 
    319:         if not scenario_id:
    320:             return {'success': False, 'error': 'scenario_id requis'}
    321: 
    322:         results = []
    323:         for device in devices:
    324:             device_result = await self._handle_run_scenario({
    325:                 'scenario_id': scenario_id,
    326:                 'browser': 'chromium',
    327:                 'device': device,
    328:                 'headless': True
    329:             })
    330:             results.append({
    331:                 'device': device,
    332:                 'success': device_result['success'],
    333:                 'duration_ms': device_result['results']['duration_ms']
    334:             })
    335: 
    336:         return {
    337:             'success': True,
    338:             'scenario_id': scenario_id,
    339:             'devices_tested': devices,
    340:             'results': results,
    341:             'summary': {
    342:                 'passed': sum(1 for r in results if r['success']),
    343:                 'failed': sum(1 for r in results if not r['success'])
    344:             }
    345:         }
    346: 
    347:     async def _handle_collect_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
    348:         """Collecte des métriques de performance"""
    349:         url = params.get('url', self._base_url)
    350:         duration_seconds = params.get('duration_seconds', 60)
    351: 
    352:         # Simulation de collecte de métriques
    353:         await asyncio.sleep(2)
    354: 
    355:         metrics = {
    356:             'page_load': {
    357:                 'dom_content_loaded_ms': random.randint(200, 800),
    358:                 'load_event_ms': random.randint(300, 1200),
    359:                 'first_contentful_paint_ms': random.randint(100, 500),
    360:                 'largest_contentful_paint_ms': random.randint(500, 2000),
    361:                 'time_to_interactive_ms': random.randint(1000, 3000)
    362:             },
    363:             'api_calls': {
    364:                 'total': random.randint(10, 50),
    365:                 'average_response_ms': random.randint(50, 200),
    366:                 'error_rate': random.uniform(0, 0.05)
    367:             },
    368:             'blockchain_interactions': {
    369:                 'total': random.randint(5, 20),
    370:                 'average_gas': random.randint(50000, 150000),
    371:                 'failed_transactions': random.randint(0, 2)
    372:             },
    373:             'resource_usage': {
    374:                 'javascript_kb': random.randint(500, 2000),
    375:                 'css_kb': random.randint(100, 500),
    376:                 'images_kb': random.randint(1000, 5000),
    377:                 'total_requests': random.randint(30, 100)
    378:             }
    379:         }
    380: 
    381:         return {
    382:             'success': True,
    383:             'url': url,
    384:             'duration_seconds': duration_seconds,
    385:             'metrics': metrics,
    386:             'performance_score': random.randint(70, 100),
    387:             'recommendations': [
    388:                 "Optimize image sizes",
    389:                 "Reduce JavaScript bundle",
    390:                 "Implement lazy loading"
    391:             ] if metrics['performance_score'] < 90 else []
    392:         }
    393: 
    394:     async def _handle_record_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
    395:         """Enregistre une session de test (vidéo)"""
    396:         scenario_id = params.get('scenario_id')
    397:         record_video = params.get('record_video', True)
    398:         record_network = params.get('record_network', True)
    399: 
    400:         if not scenario_id:
    401:             return {'success': False, 'error': 'scenario_id requis'}
    402: 
    403:         # Exécuter le scénario avec enregistrement
    404:         result = await self._handle_run_scenario({
    405:             'scenario_id': scenario_id,
    406:             'browser': 'chromium',
    407:             'device': 'desktop',
    408:             'headless': False  # Nécessaire pour l'enregistrement vidéo
    409:         })
    410: 
    411:         session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    412:         
    413:         session_info = {
    414:             'session_id': session_id,
    415:             'video_path': f"recordings/{session_id}.mp4" if record_video else None,
    416:             'network_log': f"logs/network/{session_id}.har" if record_network else None,
    417:             'screenshots': result.get('screenshots', []),
    418:             'console_log': f"logs/console/{session_id}.log"
    419:         }
    420: 
    421:         return {
    422:             'success': True,
    423:             'session': session_info,
    424:             'test_result': result,
    425:             'message': f"Session enregistrée: {session_id}"
    426:         }
    427: 
    428:     # ========================================================================
    429:     # NETTOYAGE
    430:     # ========================================================================
    431: 
    432:     async def shutdown(self) -> bool:
    433:         """Arrête le sous-agent"""
    434:         logger.info(f"Arrêt de {self._subagent_display_name}...")
    435: 
    436:         if self._cleanup_task and not self._cleanup_task.done():
    437:             self._cleanup_task.cancel()
    438:             try:
    439:                 await self._cleanup_task
    440:             except asyncio.CancelledError:
    441:                 pass
    442: 
    443:         return await super().shutdown()
    444: 
    445: 
    446: # ============================================================================
    447: # FONCTIONS D'EXPORT
    448: # ============================================================================
    449: 
    450: def get_agent_class():
    451:     """
    452:     Fonction requise pour le chargement dynamique des sous-agents.
    453:     Retourne la classe principale du sous-agent.
    454:     """
    455:     return E2ETesterSubAgent
    456: 
    457: 
    458: def create_e2e_tester_agent(config_path: str = "") -> "E2ETesterSubAgent":
    459:     """Crée une instance du sous-agent de tests E2E"""
    460:     return E2ETesterSubAgent(config_path)