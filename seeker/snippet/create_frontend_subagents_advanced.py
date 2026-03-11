#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script de création avancée de l'arborescence des sous-agents pour frontend_web3
      4: Crée des sous-agents riches en fonctionnalités avec configuration complète
      5: Version FINALE - Sans référence à 'self' dans les templates
      6: """
      7: 
      8: import os
      9: import sys
     10: from pathlib import Path
     11: from datetime import datetime
     12: from typing import List, Dict, Any, Optional
     13: 
     14: # Configuration
     15: PROJECT_ROOT = Path(__file__).parent if __file__ else Path.cwd()
     16: AGENT_PATH = PROJECT_ROOT / "agents" / "frontend_web3"
     17: SOUS_AGENTS_PATH = AGENT_PATH / "sous_agents"
     18: 
     19: # ============================================================================
     20: # SOUS-AGENT 1: REACT EXPERT
     21: # ============================================================================
     22: REACT_EXPERT = {
     23:     "id": "react_expert",
     24:     "display_name": "⚛️ Expert React/Next.js Avancé",
     25:     "description": "Expert en développement React 18+, Next.js 14+, hooks personnalisés, patterns avancés et optimisation de composants",
     26:     "version": "2.0.0",
     27:     "capabilities": [
     28:         {
     29:             "name": "react_component_generation_advanced",
     30:             "description": "Génération de composants React avec patterns avancés",
     31:             "confidence": 0.95
     32:         },
     33:         {
     34:             "name": "nextjs_app_router_master",
     35:             "description": "Maîtrise du App Router Next.js 14+",
     36:             "confidence": 0.98
     37:         },
     38:         {
     39:             "name": "react_hook_generator_advanced",
     40:             "description": "Génération de hooks personnalisés complexes",
     41:             "confidence": 0.95
     42:         },
     43:         {
     44:             "name": "performance_optimization_react",
     45:             "description": "Optimisation des performances React",
     46:             "confidence": 0.9
     47:         },
     48:         {
     49:             "name": "state_management_master",
     50:             "description": "Maîtrise des solutions de state management",
     51:             "confidence": 0.92
     52:         }
     53:     ],
     54:     "technologies": {
     55:         "frameworks": ["Next.js 14+", "React 18+", "Vite"],
     56:         "state_management": ["Zustand", "Redux Toolkit", "Jotai"]
     57:     },
     58:     "outputs": [
     59:         {"name": "react_component_library", "type": "code", "format": "typescript"},
     60:         {"name": "nextjs_application", "type": "code", "format": "typescript"}
     61:     ]
     62: }
     63: 
     64: # ============================================================================
     65: # SOUS-AGENT 2: WEB3 INTEGRATION EXPERT
     66: # ============================================================================
     67: WEB3_INTEGRATION = {
     68:     "id": "web3_integration",
     69:     "display_name": "🔗 Expert Intégration Web3 Avancé",
     70:     "description": "Expert en intégration blockchain avec wagmi, viem, ethers.js, support multi-chaînes",
     71:     "version": "2.0.0",
     72:     "capabilities": [
     73:         {
     74:             "name": "wallet_connection_advanced",
     75:             "description": "Connexion wallet multi-chaînes",
     76:             "confidence": 0.98
     77:         },
     78:         {
     79:             "name": "contract_interaction_master",
     80:             "description": "Interaction avancée avec smart contracts",
     81:             "confidence": 0.96
     82:         },
     83:         {
     84:             "name": "transaction_management",
     85:             "description": "Gestion complète des transactions",
     86:             "confidence": 0.95
     87:         },
     88:         {
     89:             "name": "multi_chain_support",
     90:             "description": "Support de multiples blockchains",
     91:             "confidence": 0.94
     92:         },
     93:         {
     94:             "name": "abi_management",
     95:             "description": "Gestion avancée des ABI",
     96:             "confidence": 0.92
     97:         }
     98:     ],
     99:     "technologies": {
    100:         "libraries": ["wagmi", "viem", "ethers.js v6"],
    101:         "wallets": ["RainbowKit", "WalletConnect v2", "Web3Modal"],
    102:         "chains": ["Ethereum", "Polygon", "Arbitrum", "Optimism"]
    103:     },
    104:     "outputs": [
    105:         {"name": "wallet_connection_system", "type": "code", "format": "typescript"},
    106:         {"name": "contract_interaction_hooks", "type": "code", "format": "typescript"}
    107:     ]
    108: }
    109: 
    110: # ============================================================================
    111: # SOUS-AGENT 3: UI/UX EXPERT
    112: # ============================================================================
    113: UI_UX_EXPERT = {
    114:     "id": "ui_ux_expert",
    115:     "display_name": "🎨 Expert UI/UX Design Système",
    116:     "description": "Expert en design system, composants accessibles, animations fluides",
    117:     "version": "2.0.0",
    118:     "capabilities": [
    119:         {
    120:             "name": "design_system_creation",
    121:             "description": "Création de design systems complets",
    122:             "confidence": 0.96
    123:         },
    124:         {
    125:             "name": "responsive_design_advanced",
    126:             "description": "Design responsive avancé",
    127:             "confidence": 0.98
    128:         },
    129:         {
    130:             "name": "animation_orchestration",
    131:             "description": "Orchestration d'animations complexes",
    132:             "confidence": 0.9
    133:         },
    134:         {
    135:             "name": "accessibility_compliance",
    136:             "description": "Conformité WCAG 2.1 AA/AAA",
    137:             "confidence": 0.95
    138:         },
    139:         {
    140:             "name": "dark_mode_system",
    141:             "description": "Système complet de dark/light mode",
    142:             "confidence": 0.94
    143:         }
    144:     ],
    145:     "technologies": {
    146:         "frameworks": ["Tailwind CSS", "Chakra UI", "Radix UI"],
    147:         "animations": ["Framer Motion", "GSAP", "React Spring"],
    148:         "charts": ["Recharts", "D3.js"]
    149:     },
    150:     "outputs": [
    151:         {"name": "design_system", "type": "code", "format": "typescript"},
    152:         {"name": "accessible_component_library", "type": "code", "format": "typescript"}
    153:     ]
    154: }
    155: 
    156: # ============================================================================
    157: # SOUS-AGENT 4: DEFI UI SPECIALIST
    158: # ============================================================================
    159: DEFI_UI_SPECIALIST = {
    160:     "id": "defi_ui_specialist",
    161:     "display_name": "📊 Spécialiste Interfaces DeFi Avancées",
    162:     "description": "Spécialiste des interfaces DeFi complexes (swap, lending, staking)",
    163:     "version": "2.0.0",
    164:     "capabilities": [
    165:         {
    166:             "name": "swap_interface_advanced",
    167:             "description": "Interface DEX avancée",
    168:             "confidence": 0.97
    169:         },
    170:         {
    171:             "name": "lending_interface_complete",
    172:             "description": "Interface lending/borrowing complète",
    173:             "confidence": 0.96
    174:         },
    175:         {
    176:             "name": "staking_interface_advanced",
    177:             "description": "Interface staking avancée",
    178:             "confidence": 0.95
    179:         },
    180:         {
    181:             "name": "vault_interface",
    182:             "description": "Interface vaults/yield optimizers",
    183:             "confidence": 0.93
    184:         },
    185:         {
    186:             "name": "defi_dashboard_comprehensive",
    187:             "description": "Dashboard DeFi complet",
    188:             "confidence": 0.95
    189:         }
    190:     ],
    191:     "protocols": ["Uniswap V3", "Aave V3", "Compound V3"],
    192:     "outputs": [
    193:         {"name": "dex_interface", "type": "code", "format": "typescript"},
    194:         {"name": "lending_dashboard", "type": "code", "format": "typescript"}
    195:     ]
    196: }
    197: 
    198: # ============================================================================
    199: # SOUS-AGENT 5: NFT UI SPECIALIST
    200: # ============================================================================
    201: NFT_UI_SPECIALIST = {
    202:     "id": "nft_ui_specialist",
    203:     "display_name": "🖼️ Spécialiste Interfaces NFT Premium",
    204:     "description": "Spécialiste des interfaces NFT (mint, gallery, marketplace)",
    205:     "version": "2.0.0",
    206:     "capabilities": [
    207:         {
    208:             "name": "mint_page_advanced",
    209:             "description": "Page de mint NFT avancée",
    210:             "confidence": 0.98
    211:         },
    212:         {
    213:             "name": "nft_gallery_pro",
    214:             "description": "Galerie NFT professionnelle",
    215:             "confidence": 0.97
    216:         },
    217:         {
    218:             "name": "marketplace_interface",
    219:             "description": "Interface marketplace complète",
    220:             "confidence": 0.96
    221:         },
    222:         {
    223:             "name": "nft_detail_view",
    224:             "description": "Vue détaillée NFT",
    225:             "confidence": 0.95
    226:         },
    227:         {
    228:             "name": "nft_auction_house",
    229:             "description": "Système d'enchères NFT",
    230:             "confidence": 0.92
    231:         }
    232:     ],
    233:     "standards": ["ERC721", "ERC1155", "ERC2981"],
    234:     "outputs": [
    235:         {"name": "nft_mint_page", "type": "code", "format": "typescript"},
    236:         {"name": "nft_marketplace", "type": "code", "format": "typescript"}
    237:     ]
    238: }
    239: 
    240: # ============================================================================
    241: # SOUS-AGENT 6: PERFORMANCE OPTIMIZER
    242: # ============================================================================
    243: PERFORMANCE_OPTIMIZER = {
    244:     "id": "performance_optimizer",
    245:     "display_name": "⚡ Optimisateur Performance Web3",
    246:     "description": "Expert en optimisation des performances Web3",
    247:     "version": "2.0.0",
    248:     "capabilities": [
    249:         {
    250:             "name": "bundle_optimization",
    251:             "description": "Optimisation de la taille du bundle",
    252:             "confidence": 0.95
    253:         },
    254:         {
    255:             "name": "core_web_vitals",
    256:             "description": "Optimisation Core Web Vitals",
    257:             "confidence": 0.96
    258:         },
    259:         {
    260:             "name": "caching_strategy",
    261:             "description": "Stratégies de caching avancées",
    262:             "confidence": 0.94
    263:         },
    264:         {
    265:             "name": "image_optimization",
    266:             "description": "Optimisation des images",
    267:             "confidence": 0.93
    268:         },
    269:         {
    270:             "name": "web3_performance",
    271:             "description": "Optimisation des appels Web3",
    272:             "confidence": 0.92
    273:         }
    274:     ],
    275:     "metrics": ["LCP < 2.5s", "FID < 100ms", "CLS < 0.1"],
    276:     "outputs": [
    277:         {"name": "performance_report", "type": "report", "format": "json"},
    278:         {"name": "optimization_config", "type": "config", "format": "json"}
    279:     ]
    280: }
    281: 
    282: # ============================================================================
    283: # SOUS-AGENT 7: SECURITY UI SPECIALIST
    284: # ============================================================================
    285: SECURITY_UI_SPECIALIST = {
    286:     "id": "security_ui_specialist",
    287:     "display_name": "🛡️ Spécialiste Sécurité UI Avancée",
    288:     "description": "Expert en sécurité des interfaces Web3",
    289:     "version": "2.0.0",
    290:     "capabilities": [
    291:         {
    292:             "name": "transaction_security",
    293:             "description": "Validation de transactions",
    294:             "confidence": 0.98
    295:         },
    296:         {
    297:             "name": "phishing_protection",
    298:             "description": "Protection anti-phishing",
    299:             "confidence": 0.97
    300:         },
    301:         {
    302:             "name": "wallet_security",
    303:             "description": "Sécurité wallet",
    304:             "confidence": 0.96
    305:         },
    306:         {
    307:             "name": "smart_contract_verification",
    308:             "description": "Vérification des smart contracts",
    309:             "confidence": 0.95
    310:         },
    311:         {
    312:             "name": "approval_management",
    313:             "description": "Gestion des approvals",
    314:             "confidence": 0.94
    315:         }
    316:     ],
    317:     "standards": ["EIP-712", "EIP-2612", "EIP-1271"],
    318:     "outputs": [
    319:         {"name": "security_configuration", "type": "config", "format": "json"},
    320:         {"name": "transaction_guard", "type": "code", "format": "typescript"}
    321:     ]
    322: }
    323: 
    324: # Liste complète des sous-agents
    325: SUB_AGENTS = [
    326:     REACT_EXPERT,
    327:     WEB3_INTEGRATION,
    328:     UI_UX_EXPERT,
    329:     DEFI_UI_SPECIALIST,
    330:     NFT_UI_SPECIALIST,
    331:     PERFORMANCE_OPTIMIZER,
    332:     SECURITY_UI_SPECIALIST
    333: ]
    334: 
    335: # ============================================================================
    336: # FONCTIONS DE GÉNÉRATION
    337: # ============================================================================
    338: 
    339: def create_agent_py(agent_data: Dict) -> str:
    340:     """Crée le contenu du fichier agent.py pour un sous-agent"""
    341:     agent_id = agent_data["id"]
    342:     display_name = agent_data["display_name"]
    343:     description = agent_data["description"]
    344:     version = agent_data["version"]
    345:     class_name = ''.join(p.capitalize() for p in agent_id.split('_')) + 'SubAgent'
    346:     
    347:     capabilities_list = '\n'.join([f'        "{cap["name"]}",' for cap in agent_data["capabilities"]])
    348:     
    349:     return f'''"""
    350: {display_name} - Sous-agent spécialisé
    351: Version: {version}
    352: 
    353: {description}
    354: """
    355: 
    356: import logging
    357: import sys
    358: import asyncio
    359: from datetime import datetime
    360: from pathlib import Path
    361: from typing import Dict, Any, Optional
    362: 
    363: # Configuration des imports
    364: current_dir = Path(__file__).parent.absolute()
    365: project_root = current_dir.parent.parent.parent.parent.parent
    366: if str(project_root) not in sys.path:
    367:     sys.path.insert(0, str(project_root))
    368: 
    369: from agents.base_agent.base_agent import BaseAgent, AgentStatus, Message, MessageType
    370: 
    371: logger = logging.getLogger(__name__)
    372: 
    373: 
    374: class {class_name}(BaseAgent):
    375:     """
    376:     {description}
    377:     """
    378:     
    379:     def __init__(self, config_path: str = ""):
    380:         """Initialise le sous-agent"""
    381:         if not config_path:
    382:             config_path = str(current_dir / "config.yaml")
    383: 
    384:         super().__init__(config_path)
    385: 
    386:         self._display_name = "{display_name}"
    387:         self._initialized = False
    388:         self._components = {{}}
    389:         
    390:         # Statistiques
    391:         self._stats = {{
    392:             'tasks_processed': 0,
    393:             'components_generated': 0,
    394:             'start_time': datetime.now().isoformat()
    395:         }}
    396: 
    397:         logger.info(f"{{self._display_name}} créé - v{{self._version}}")
    398: 
    399:     async def initialize(self) -> bool:
    400:         """Initialise le sous-agent"""
    401:         try:
    402:             self._set_status(AgentStatus.INITIALIZING)
    403:             logger.info(f"Initialisation de {{self._display_name}}...")
    404: 
    405:             base_result = await super().initialize()
    406:             if not base_result:
    407:                 return False
    408: 
    409:             await self._initialize_components()
    410: 
    411:             self._initialized = True
    412:             self._set_status(AgentStatus.READY)
    413:             logger.info(f"✅ {{self._display_name}} prêt")
    414:             return True
    415: 
    416:         except Exception as e:
    417:             logger.error(f"❌ Erreur initialisation: {{e}}")
    418:             self._set_status(AgentStatus.ERROR)
    419:             return False
    420: 
    421:     async def _initialize_components(self) -> bool:
    422:         """Initialise les composants du sous-agent"""
    423:         logger.info("Initialisation des composants...")
    424:         
    425:         self._components = {{
    426:             "capabilities": [
    427: {capabilities_list}
    428:             ],
    429:             "enabled": True,
    430:             "version": "{version}"
    431:         }}
    432:         
    433:         logger.info(f"✅ Composants: {{list(self._components.keys())}}")
    434:         return True
    435: 
    436:     async def _handle_custom_message(self, message: Message) -> Optional[Message]:
    437:         """Gère les messages personnalisés"""
    438:         try:
    439:             msg_type = message.message_type
    440:             logger.debug(f"Message reçu: {{msg_type}} de {{message.sender}}")
    441: 
    442:             handlers = {{
    443:                 f"{{self.name}}.status": self._handle_status,
    444:                 f"{{self.name}}.metrics": self._handle_metrics,
    445:                 f"{{self.name}}.health": self._handle_health,
    446:             }}
    447: 
    448:             if msg_type in handlers:
    449:                 return await handlers[msg_type](message)
    450: 
    451:             return None
    452: 
    453:         except Exception as e:
    454:             logger.error(f"Erreur traitement message: {{e}}")
    455:             return Message(
    456:                 sender=self.name,
    457:                 recipient=message.sender,
    458:                 content={{"error": str(e)}},
    459:                 message_type=MessageType.ERROR.value,
    460:                 correlation_id=message.message_id
    461:             )
    462: 
    463:     async def _handle_status(self, message: Message) -> Message:
    464:         """Retourne le statut général"""
    465:         return Message(
    466:             sender=self.name,
    467:             recipient=message.sender,
    468:             content={{
    469:                 'status': self._status.value,
    470:                 'initialized': self._initialized,
    471:                 'stats': self._stats
    472:             }},
    473:             message_type=f"{{self.name}}.status_response",
    474:             correlation_id=message.message_id
    475:         )
    476: 
    477:     async def _handle_metrics(self, message: Message) -> Message:
    478:         """Retourne les métriques"""
    479:         return Message(
    480:             sender=self.name,
    481:             recipient=message.sender,
    482:             content=self._stats,
    483:             message_type=f"{{self.name}}.metrics_response",
    484:             correlation_id=message.message_id
    485:         )
    486: 
    487:     async def _handle_health(self, message: Message) -> Message:
    488:         """Retourne l'état de santé"""
    489:         health = await self.health_check()
    490:         return Message(
    491:             sender=self.name,
    492:             recipient=message.sender,
    493:             content=health,
    494:             message_type=f"{{self.name}}.health_response",
    495:             correlation_id=message.message_id
    496:         )
    497: 
    498:     async def shutdown(self) -> bool:
    499:         """Arrête le sous-agent"""
    500:         logger.info(f"Arrêt de {{self._display_name}}...")
    501:         self._set_status(AgentStatus.SHUTTING_DOWN)
    502:         await super().shutdown()
    503:         logger.info(f"✅ {{self._display_name}} arrêté")
    504:         return True
    505: 
    506:     async def pause(self) -> bool:
    507:         """Met en pause le sous-agent"""
    508:         logger.info(f"Pause de {{self._display_name}}...")
    509:         self._set_status(AgentStatus.PAUSED)
    510:         return True
    511: 
    512:     async def resume(self) -> bool:
    513:         """Reprend l'activité"""
    514:         logger.info(f"Reprise de {{self._display_name}}...")
    515:         self._set_status(AgentStatus.READY)
    516:         return True
    517: 
    518:     async def health_check(self) -> Dict[str, Any]:
    519:         """Vérifie la santé du sous-agent"""
    520:         base_health = await super().health_check()
    521: 
    522:         uptime = None
    523:         if self._stats.get('start_time'):
    524:             start = datetime.fromisoformat(self._stats['start_time'])
    525:             uptime = str(datetime.now() - start)
    526: 
    527:         return {{
    528:             **base_health,
    529:             "agent": self.name,
    530:             "display_name": self._display_name,
    531:             "status": self._status.value,
    532:             "ready": self._status == AgentStatus.READY,
    533:             "initialized": self._initialized,
    534:             "uptime": uptime,
    535:             "components": list(self._components.keys()),
    536:             "stats": self._stats,
    537:             "timestamp": datetime.now().isoformat()
    538:         }}
    539: 
    540:     def get_agent_info(self) -> Dict[str, Any]:
    541:         """Informations pour le registre"""
    542:         return {{
    543:             "id": self.name,
    544:             "name": "{class_name}",
    545:             "display_name": self._display_name,
    546:             "version": "{version}",
    547:             "description": \"\"\"{description}\"\"\",
    548:             "status": self._status.value,
    549:             "capabilities": [
    550: {capabilities_list}
    551:             ],
    552:             "stats": self._stats
    553:         }}
    554: 
    555: 
    556: def create_{agent_id}_agent(config_path: str = "") -> {class_name}:
    557:     """Crée une instance du sous-agent"""
    558:     return {class_name}(config_path)
    559: '''
    560: 
    561: 
    562: def create_config_yaml(agent_data: Dict) -> str:
    563:     """Crée le contenu du fichier config.yaml pour un sous-agent"""
    564:     agent_id = agent_data["id"]
    565:     display_name = agent_data["display_name"]
    566:     description = agent_data["description"]
    567:     version = agent_data["version"]
    568:     class_name = ''.join(p.capitalize() for p in agent_id.split('_')) + 'SubAgent'
    569:     
    570:     capabilities_yaml = '\n'.join([f'  - name: "{cap["name"]}"' for cap in agent_data["capabilities"]])
    571:     
    572:     # Technologies
    573:     tech_items = []
    574:     for cat, items in agent_data.get("technologies", {}).items():
    575:         for item in items:
    576:             tech_items.append(item)
    577:     technologies_yaml = '\n'.join([f'    - "{tech}"' for tech in tech_items[:3]])  # Limiter à 3 pour lisibilité
    578:     
    579:     # Outputs
    580:     outputs_yaml = '\n'.join([f'  - name: "{out["name"]}"' for out in agent_data.get("outputs", [])])
    581:     
    582:     return f'''# ============================================================================
    583: # {display_name} - Configuration
    584: # Version: {version}
    585: # ============================================================================
    586: 
    587: agent:
    588:   name: "{class_name}"
    589:   display_name: "{display_name}"
    590:   description: |-
    591:     {description}
    592:   version: "{version}"
    593:   class_name: "{class_name}"
    594:   module_path: "agents.frontend_web3.sous_agents.{agent_id}.agent"
    595: 
    596: # ============================================================================
    597: # SYSTÈME
    598: # ============================================================================
    599: system:
    600:   log_level: "INFO"
    601:   timeout_seconds: 60
    602:   max_retries: 2
    603: 
    604: # ============================================================================
    605: # CAPACITÉS
    606: # ============================================================================
    607: capabilities:
    608: {capabilities_yaml}
    609: 
    610: # ============================================================================
    611: # TECHNOLOGIES SUPPORTÉES
    612: # ============================================================================
    613: technologies:
    614:   supported:
    615: {technologies_yaml}
    616: 
    617: # ============================================================================
    618: # OUTPUTS PRODUITS
    619: # ============================================================================
    620: outputs:
    621: {outputs_yaml}
    622: 
    623: # ============================================================================
    624: # MÉTADONNÉES
    625: # ============================================================================
    626: metadata:
    627:   author: "SmartContractDevPipeline"
    628:   maintainer: "dev@poolsync.io"
    629:   license: "Proprietary"
    630:   capabilities_count: {len(agent_data["capabilities"])}
    631:   last_reviewed: "{datetime.now().strftime("%Y-%m-%d")}"
    632: '''
    633: 
    634: 
    635: def create_directory_structure():
    636:     """Crée toute l'arborescence des dossiers et fichiers"""
    637:     
    638:     print("=" * 80)
    639:     print("🚀 CRÉATION AVANCÉE DES SOUS-AGENTS FRONTEND_WEB3")
    640:     print("=" * 80)
    641:     
    642:     # 1. Créer le dossier principal des sous-agents
    643:     print(f"\n📁 Création du dossier: {SOUS_AGENTS_PATH}")
    644:     SOUS_AGENTS_PATH.mkdir(parents=True, exist_ok=True)
    645:     
    646:     # 2. Créer le __init__.py du dossier sous_agents
    647:     init_file = SOUS_AGENTS_PATH / "__init__.py"
    648:     print(f"   📄 Création: {init_file}")
    649:     with open(init_file, 'w', encoding='utf-8') as f:
    650:         f.write('"""\nPackage des sous-agents Frontend Web3\nSous-agents spécialisés pour le développement d\'interfaces Web3\n"""\n\n__all__ = []\n')
    651:     
    652:     # 3. Créer chaque sous-agent
    653:     for agent in SUB_AGENTS:
    654:         agent_id = agent["id"]
    655:         display_name = agent["display_name"]
    656:         
    657:         agent_path = SOUS_AGENTS_PATH / agent_id
    658:         print(f"\n📁 Création du sous-agent: {agent_id}")
    659:         print(f"   📁 Dossier: {agent_path}")
    660:         agent_path.mkdir(exist_ok=True)
    661:         
    662:         # Créer __init__.py
    663:         init_file = agent_path / "__init__.py"
    664:         print(f"   📄 Création: {init_file}")
    665:         with open(init_file, 'w', encoding='utf-8') as f:
    666:             class_name = ''.join(p.capitalize() for p in agent_id.split('_')) + 'SubAgent'
    667:             f.write(f'''"""
    668: Package {display_name}
    669: Version: {agent["version"]}
    670: 
    671: {agent["description"]}
    672: """
    673: 
    674: from .agent import {class_name}
    675: 
    676: __all__ = ['{class_name}']
    677: __version__ = '{agent["version"]}'
    678: ''')
    679:         
    680:         # Créer agent.py
    681:         agent_file = agent_path / "agent.py"
    682:         print(f"   📄 Création: {agent_file}")
    683:         with open(agent_file, 'w', encoding='utf-8') as f:
    684:             f.write(create_agent_py(agent))
    685:         
    686:         # Créer config.yaml
    687:         config_file = agent_path / "config.yaml"
    688:         print(f"   📄 Création: {config_file}")
    689:         with open(config_file, 'w', encoding='utf-8') as f:
    690:             f.write(create_config_yaml(agent))
    691:         
    692:         # Créer tools.py
    693:         tools_file = agent_path / "tools.py"
    694:         print(f"   📄 Création: {tools_file}")
    695:         with open(tools_file, 'w', encoding='utf-8') as f:
    696:             f.write(f'''"""
    697: Outils utilitaires pour {display_name}
    698: Version: {agent["version"]}
    699: """
    700: 
    701: import logging
    702: from typing import Dict, Any, Optional, List
    703: from datetime import datetime
    704: 
    705: logger = logging.getLogger(__name__)
    706: 
    707: 
    708: def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    709:     """Formate un timestamp pour l'affichage"""
    710:     if timestamp is None:
    711:         timestamp = datetime.now()
    712:     return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    713: ''')
    714:         
    715:         print(f"   ✅ Sous-agent {agent_id} créé avec succès")
    716:     
    717:     print("\n" + "=" * 80)
    718:     print("✅ CRÉATION TERMINÉE AVEC SUCCÈS !")
    719:     print("=" * 80)
    720:     
    721:     # Résumé
    722:     print(f"\n📊 RÉSUMÉ:")
    723:     print(f"   • Dossier principal: {SOUS_AGENTS_PATH}")
    724:     print(f"   • Nombre de sous-agents: {len(SUB_AGENTS)}")
    725:     print(f"   • Fichiers créés: {len(SUB_AGENTS) * 4 + 1} fichiers")
    726: 
    727: 
    728: def verify_agent_parent():
    729:     """Vérifie que l'agent parent existe"""
    730:     agent_file = AGENT_PATH / "agent.py"
    731:     
    732:     if not agent_file.exists():
    733:         print(f"\n⚠️  Attention: {agent_file} n'existe pas!")
    734:         print("   Vous devez d'abord créer l'agent parent frontend_web3.")
    735:         return False
    736:     
    737:     return True
    738: 
    739: 
    740: if __name__ == "__main__":
    741:     # Vérifier qu'on est à la racine du projet
    742:     if not (PROJECT_ROOT / "agents").exists():
    743:         print(f"❌ Erreur: Le dossier 'agents' n'existe pas dans {PROJECT_ROOT}")
    744:         print("   Assurez-vous d'exécuter ce script depuis la racine du projet")
    745:         sys.exit(1)
    746:     
    747:     # Créer l'arborescence
    748:     create_directory_structure()
    749:     
    750:     # Vérifications finales
    751:     verify_agent_parent()
    752:     
    753:     print(f"\n🎉 Tous les sous-agents ont été créés avec succès!")
    754:     print(f"\n🚀 Vous pouvez maintenant lancer votre application!")