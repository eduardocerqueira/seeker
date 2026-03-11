#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script principal de déploiement des agents - Version corrigée
      4: """
      5: import os
      6: import sys
      7: import subprocess
      8: import yaml
      9: from pathlib import Path
     10: 
     11: def create_directory_structure():
     12:     """Crée la structure complète des répertoires"""
     13:     structure = {
     14:         # Agents principaux
     15:         "agents/main/architect": ["__init__.py", "config.yaml", "agent.py", "orchestrator.py"],
     16:         "agents/main/coder": ["__init__.py", "config.yaml", "agent.py", "orchestrator.py"],
     17:         "agents/main/smart_contract": ["__init__.py", "config.yaml", "agent.py", "orchestrator.py"],
     18:         "agents/main/communication": ["__init__.py", "config.yaml", "agent.py", "message_bus.py"],
     19:         "agents/main/documenter": ["__init__.py", "config.yaml", "agent.py", "orchestrator.py"],
     20:         "agents/main/formal_verification": ["__init__.py", "config.yaml", "agent.py", "orchestrator.py"],
     21:         "agents/main/frontend_web3": ["__init__.py", "config.yaml", "agent.py", "orchestrator.py"],
     22:         "agents/main/quality_metrics": ["__init__.py", "config.yaml", "agent.py", "orchestrator.py"],
     23:         "agents/main/tester": ["__init__.py", "config.yaml", "agent.py", "orchestrator.py"],
     24:         
     25:         # Sous-agents
     26:         "agents/sub/architecture/cloud": ["__init__.py", "config.yaml", "agent.py", "tools.py"],
     27:         "agents/sub/architecture/blockchain": ["__init__.py", "config.yaml", "agent.py", "tools.py"],
     28:         "agents/sub/architecture/microservices": ["__init__.py", "config.yaml", "agent.py", "tools.py"],
     29:         
     30:         # Micro-agents
     31:         "agents/micro/database": ["__init__.py", "config.yaml", "agent.py"],
     32:         "agents/micro/cache": ["__init__.py", "config.yaml", "agent.py"],
     33:         "agents/micro/api_gateway": ["__init__.py", "config.yaml", "agent.py"],
     34:         
     35:         # Communication
     36:         "agents/communication/zeromq": ["__init__.py", "config.yaml", "message_bus.py", "publisher.py", "subscriber.py"],
     37:         "agents/communication/redis": ["__init__.py", "config.yaml", "pubsub.py", "event_handler.py"],
     38:         
     39:         # Orchestration
     40:         "agents/orchestration/master": ["__init__.py", "config.yaml", "orchestrator.py", "scheduler.py", "monitor.py"],
     41:     }
     42:     
     43:     created_count = 0
     44:     for directory, files in structure.items():
     45:         # Créer le répertoire
     46:         Path(directory).mkdir(parents=True, exist_ok=True)
     47:         
     48:         # Créer les fichiers
     49:         for file in files:
     50:             file_path = Path(directory) / file
     51:             if not file_path.exists():
     52:                 if file.endswith('.py'):
     53:                     # Fichier Python minimal
     54:                     if file == "__init__.py":
     55:                         file_path.write_text("# Package initializer\n")
     56:                     elif file == "config.yaml":
     57:                         file_path.write_text(f"# Configuration for {directory}\nname: {Path(directory).name}\n")
     58:                     else:
     59:                         file_path.write_text(f"# {file} for {directory}\n# TODO: Implement this module\n")
     60:                 created_count += 1
     61:     
     62:     return created_count
     63: 
     64: def install_dependencies():
     65:     """Installe les dépendances Python requises"""
     66:     dependencies = [
     67:         "pyzmq>=25.0.0",
     68:         "redis>=5.0.0",
     69:         "pyyaml>=6.0",
     70:         "pydantic>=2.0.0",
     71:         "asyncio-mqtt>=0.13.0",
     72:         "websockets>=12.0",
     73:     ]
     74:     
     75:     print("📦 Installation des dépendances Python...")
     76:     for dep in dependencies:
     77:         try:
     78:             subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
     79:             print(f"  ✅ {dep}")
     80:         except subprocess.CalledProcessError:
     81:             print(f"  ❌ Échec installation: {dep}")
     82:     
     83:     return True
     84: 
     85: def create_main_config():
     86:     """Crée le fichier de configuration principal"""
     87:     config = {
     88:         "project": "PoolSync-Agents",
     89:         "version": "1.0.0",
     90:         "communication": {
     91:             "zeromq": {
     92:                 "host": "127.0.0.1",
     93:                 "pub_port": 5555,
     94:                 "sub_port": 5556,
     95:                 "router_port": 5557
     96:             },
     97:             "redis": {
     98:                 "host": "localhost",
     99:                 "port": 6379,
    100:                 "db": 0
    101:             }
    102:         },
    103:         "agents": {
    104:             "auto_start": ["architect", "coder", "communication"],
    105:             "health_check_interval": 30,
    106:             "log_level": "INFO"
    107:         },
    108:         "security": {
    109:             "api_keys": {},
    110:             "encryption_enabled": False,
    111:             "auth_required": True
    112:         }
    113:     }
    114:     
    115:     config_path = Path("config") / "main.yaml"
    116:     config_path.parent.mkdir(exist_ok=True)
    117:     
    118:     with open(config_path, 'w') as f:
    119:         yaml.dump(config, f, default_flow_style=False)
    120:     
    121:     print(f"📄 Configuration créée: {config_path}")
    122:     return config_path
    123: 
    124: def setup_logging():
    125:     """Configure le système de logging"""
    126:     logging_config = """version: 1
    127: formatters:
    128:   standard:
    129:     format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    130: handlers:
    131:   console:
    132:     class: logging.StreamHandler
    133:     formatter: standard
    134:     level: INFO
    135:   file:
    136:     class: logging.handlers.RotatingFileHandler
    137:     formatter: standard
    138:     filename: logs/agents.log
    139:     maxBytes: 10485760
    140:     backupCount: 5
    141:     level: DEBUG
    142: loggers:
    143:   agents:
    144:     level: DEBUG
    145:     handlers: [console, file]
    146:     propagate: no
    147: root:
    148:   level: INFO
    149:   handlers: [console]
    150: """
    151:     
    152:     logs_dir = Path("logs")
    153:     logs_dir.mkdir(exist_ok=True)
    154:     
    155:     config_path = Path("config") / "logging.yaml"
    156:     config_path.parent.mkdir(exist_ok=True)
    157:     
    158:     with open(config_path, 'w') as f:
    159:         f.write(logging_config)
    160:     
    161:     print(f"📝 Configuration logging créée: {config_path}")
    162:     return config_path
    163: 
    164: def create_example_agent():
    165:     """Crée un exemple d'agent de démonstration"""
    166:     example_code = '''"""
    167: Agent de démonstration pour PoolSync
    168: """
    169: import asyncio
    170: import logging
    171: from typing import Dict, Any
    172: from datetime import datetime
    173: 
    174: from agents.communication.zeromq.message_bus import ZeroMQMessageBus, Message
    175: 
    176: logger = logging.getLogger(__name__)
    177: 
    178: class DemoAgent:
    179:     """Agent de démonstration avec communication ZeroMQ"""
    180:     
    181:     def __init__(self, agent_id: str, message_bus: ZeroMQMessageBus):
    182:         self.agent_id = agent_id
    183:         self.message_bus = message_bus
    184:         self.running = False
    185:         
    186:     async def start(self):
    187:         """Démarre l'agent"""
    188:         self.running = True
    189:         
    190:         # S'abonner aux messages
    191:         self.message_bus.subscribe(
    192:             agent_id=self.agent_id,
    193:             message_types=["demo.command", "system.status"],
    194:             handler=self.handle_message
    195:         )
    196:         
    197:         logger.info(f"Agent {self.agent_id} démarré")
    198:         
    199:         # Boucle principale
    200:         while self.running:
    201:             try:
    202:                 # Envoyer un heartbeat
    203:                 await self.send_heartbeat()
    204:                 await asyncio.sleep(10)
    205:                 
    206:             except asyncio.CancelledError:
    207:                 break
    208:             except Exception as e:
    209:                 logger.error(f"Erreur dans l'agent {self.agent_id}: {e}")
    210:                 await asyncio.sleep(5)
    211:     
    212:     async def send_heartbeat(self):
    213:         """Envoie un heartbeat"""
    214:         message = Message(
    215:             id=f"heartbeat-{datetime.now().timestamp()}",
    216:             sender=self.agent_id,
    217:             receivers=["orchestrator"],
    218:             message_type="agent.heartbeat",
    219:             content={
    220:                 "agent_id": self.agent_id,
    221:                 "timestamp": datetime.now().isoformat(),
    222:                 "status": "running"
    223:             },
    224:             requires_ack=False
    225:         )
    226:         
    227:         self.message_bus.publish(message)
    228:         logger.debug(f"Heartbeat envoyé par {self.agent_id}")
    229:     
    230:     def handle_message(self, message: Message):
    231:         """Traite un message reçu"""
    232:         logger.info(f"{self.agent_id} reçu message: {message.message_type}")
    233:         
    234:         if message.message_type == "demo.command":
    235:             self.process_command(message.content)
    236:         elif message.message_type == "system.status":
    237:             self.report_status()
    238:     
    239:     def process_command(self, command: Dict[str, Any]):
    240:         """Traite une commande"""
    241:         action = command.get("action")
    242:         
    243:         if action == "echo":
    244:             response = Message(
    245:                 sender=self.agent_id,
    246:                 receivers=[command.get("from", "unknown")],
    247:                 message_type="demo.response",
    248:                 content={"echo": command.get("data", ""), "processed_by": self.agent_id}
    249:             )
    250:             self.message_bus.publish(response)
    251:             logger.info(f"{self.agent_id} a traité la commande: {action}")
    252:     
    253:     def report_status(self):
    254:         """Reporte le statut de l'agent"""
    255:         status = {
    256:             "agent_id": self.agent_id,
    257:             "running": self.running,
    258:             "timestamp": datetime.now().isoformat()
    259:         }
    260:         
    261:         message = Message(
    262:             sender=self.agent_id,
    263:             receivers=["monitor"],
    264:             message_type="agent.status",
    265:             content=status
    266:         )
    267:         
    268:         self.message_bus.publish(message)
    269:     
    270:     async def stop(self):
    271:         """Arrête l'agent"""
    272:         self.running = False
    273:         logger.info(f"Agent {self.agent_id} arrêté")
    274: 
    275: async def main():
    276:     """Fonction principale de démonstration"""
    277:     # Initialiser le bus de messages
    278:     message_bus = ZeroMQMessageBus(host="127.0.0.1", pub_port=5555)
    279:     message_bus.start()
    280:     
    281:     # Créer un agent de démonstration
    282:     demo_agent = DemoAgent(agent_id="demo_agent_1", message_bus=message_bus)
    283:     
    284:     try:
    285:         # Démarrer l'agent
    286:         await demo_agent.start()
    287:     except KeyboardInterrupt:
    288:         print("\\nArrêt demandé...")
    289:     finally:
    290:         # Nettoyage
    291:         await demo_agent.stop()
    292:         message_bus.stop()
    293: 
    294: if __name__ == "__main__":
    295:     asyncio.run(main())
    296: '''
    297:     
    298:     demo_path = Path("examples") / "demo_agent.py"
    299:     demo_path.parent.mkdir(exist_ok=True)
    300:     
    301:     with open(demo_path, 'w') as f:
    302:         f.write(example_code)
    303:     
    304:     print(f"🎯 Exemple d'agent créé: {demo_path}")
    305:     return demo_path
    306: 
    307: def main():
    308:     """Fonction principale"""
    309:     print("=" * 60)
    310:     print("🚀 DÉPLOIEMENT DES AGENTS POOLSYNC - VERSION CORRIGÉE")
    311:     print("=" * 60)
    312:     
    313:     try:
    314:         # 1. Créer la structure
    315:         print("\n1. 📁 Création de la structure des répertoires...")
    316:         created = create_directory_structure()
    317:         print(f"   ✅ {created} fichiers/dossiers créés")
    318:         
    319:         # 2. Installer les dépendances
    320:         print("\n2. 📦 Installation des dépendances...")
    321:         install_dependencies()
    322:         
    323:         # 3. Configuration
    324:         print("\n3. ⚙️  Configuration du système...")
    325:         create_main_config()
    326:         setup_logging()
    327:         
    328:         # 4. Exemples
    329:         print("\n4. 🔧 Création des exemples...")
    330:         create_example_agent()
    331:         
    332:         # 5. Fichier README
    333:         readme = """# 🏗️ PoolSync Agents System
    334: 
    335: ## Architecture Multi-Agents Corrigée
    336: 
    337: Ce système implémente une architecture multi-agents pour le développement DeFi.
    338: 
    339: ### Structure Principale