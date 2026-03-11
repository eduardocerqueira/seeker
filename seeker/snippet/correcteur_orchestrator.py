#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: # correcteur_orchestrator.py
      2: import os
      3: import sys
      4: 
      5: print("🔧 Correction de l'orchestrateur")
      6: print("=" * 50)
      7: 
      8: # Chemin du fichier orchestrator
      9: orchestrator_path = os.path.join("orchestrator", "orchestrator.py")
     10: backup_path = orchestrator_path + ".backup"
     11: 
     12: # Lire le fichier
     13: with open(orchestrator_path, 'r', encoding='utf-8') as f:
     14:     content = f.read()
     15: 
     16: print(f"📄 Lecture de {orchestrator_path}")
     17: 
     18: # Sauvegarder
     19: with open(backup_path, 'w', encoding='utf-8') as f:
     20:     f.write(content)
     21: print(f"💾 Backup créé: {backup_path}")
     22: 
     23: # Trouver la méthode initialize_agents
     24: if "async def initialize_agents(self):" in content:
     25:     print("✅ Méthode initialize_agents trouvée")
     26:     
     27:     # Nouvelle version corrigée de la méthode
     28:     new_initialize_method = '''    async def initialize_agents(self):
     29:         """Initialise tous les agents du pipeline"""
     30:         if self.initialized:
     31:             return
     32:         
     33:         self.logger.info("Initialisation des agents...")
     34:         
     35:         # Définir le chemin du projet pour les imports
     36:         import os
     37:         import sys
     38:         project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
     39:         if project_root not in sys.path:
     40:             sys.path.insert(0, project_root)
     41:         
     42:         # Dynamiquement importer les agents basés sur la config
     43:         agents_to_load = self.config.get("agents", {})
     44:         
     45:         for agent_name, agent_config in agents_to_load.items():
     46:             if agent_config.get("enabled", True):
     47:                 try:
     48:                     # Construction du chemin d'import
     49:                     module_path = agent_config.get("module", f"agents.{agent_name}.agent")
     50:                     agent_class_name = agent_config.get("class", f"{agent_name.capitalize()}Agent")
     51:                     
     52:                     # Import dynamique simplifié
     53:                     module_parts = module_path.split('.')
     54:                     
     55:                     try:
     56:                         # Essayer d'importer directement
     57:                         exec(f"from {module_path} import {agent_class_name}")
     58:                         agent_class = eval(agent_class_name)
     59:                     except:
     60:                         # Méthode alternative avec importlib
     61:                         import importlib
     62:                         module_name = '.'.join(module_parts[:-1])
     63:                         class_name = module_parts[-1]
     64:                         
     65:                         if module_name:
     66:                             module = importlib.import_module(module_name)
     67:                         else:
     68:                             module = importlib.import_module(agent_name)
     69:                         
     70:                         agent_class = getattr(module, agent_class_name)
     71:                     
     72:                     # Instanciation
     73:                     config_path = agent_config.get("config_path", "")
     74:                     agent_instance = agent_class(config_path)
     75:                     self.agents[agent_name] = agent_instance
     76:                     
     77:                     self.logger.info(f"✅ Agent {agent_name} initialisé")
     78:                     
     79:                 except Exception as e:
     80:                     self.logger.error(f"❌ Erreur lors de l'initialisation de {agent_name}: {e}")
     81:                     import traceback
     82:                     self.logger.error(traceback.format_exc())
     83:         
     84:         self.initialized = True
     85:         self.logger.info(f"✅ {len(self.agents)} agents initialisés")'''
     86:     
     87:     # Remplacer l'ancienne méthode
     88:     import re
     89:     pattern = r'async def initialize_agents\(self\):(?s:.*?)(?=\n    async def|\n    def|\n\n|$)'
     90:     
     91:     # Essayer de remplacer
     92:     new_content, count = re.subn(pattern, new_initialize_method, content)
     93:     
     94:     if count > 0:
     95:         print("✅ Méthode initialize_agents remplacée")
     96:         
     97:         # Écrire le fichier corrigé
     98:         with open(orchestrator_path, 'w', encoding='utf-8') as f:
     99:             f.write(new_content)
    100:         
    101:         print(f"✅ {orchestrator_path} corrigé")
    102:         
    103:         # Afficher un extrait de la correction
    104:         print("\n📋 Extrait de la correction:")
    105:         print("-" * 40)
    106:         lines = new_initialize_method.split('\n')[:15]
    107:         for line in lines:
    108:             print(line)
    109:         print("...")
    110:         print("-" * 40)
    111:         
    112:     else:
    113:         print("❌ Impossible de trouver/replacer la méthode")
    114:         # Essayer une autre méthode
    115:         lines = content.split('\n')
    116:         new_lines = []
    117:         in_method = False
    118:         
    119:         for line in lines:
    120:             if line.strip() == "async def initialize_agents(self):":
    121:                 in_method = True
    122:                 new_lines.append(line)
    123:                 # Ajouter notre nouvelle méthode
    124:                 new_lines.extend(new_initialize_method.split('\n')[1:])
    125:             elif in_method and line.startswith("    async def") and line != "    async def initialize_agents(self):":
    126:                 in_method = False
    127:                 new_lines.append(line)
    128:             elif not in_method:
    129:                 new_lines.append(line)
    130:         
    131:         with open(orchestrator_path, 'w', encoding='utf-8') as f:
    132:             f.write('\n'.join(new_lines))
    133:         
    134:         print("✅ Correction appliquée (méthode alternative)")
    135:         
    136: else:
    137:     print("❌ Méthode initialize_agents non trouvée")
    138: 
    139: # Vérifier aussi le début du fichier pour ajouter les imports nécessaires
    140: print("\n🔍 Vérification des imports...")
    141: 
    142: if "import importlib" not in content:
    143:     print("⚠️  importlib manquant, ajout...")
    144:     
    145:     # Ajouter importlib après les autres imports
    146:     lines = content.split('\n')
    147:     new_lines = []
    148:     
    149:     for i, line in enumerate(lines):
    150:         new_lines.append(line)
    151:         if line.startswith("import ") and "importlib" not in line and i+1 < len(lines) and not lines[i+1].startswith("import "):
    152:             new_lines.append("import importlib")
    153:     
    154:     with open(orchestrator_path, 'w', encoding='utf-8') as f:
    155:         f.write('\n'.join(new_lines))
    156:     
    157:     print("✅ importlib ajouté")
    158: 
    159: print("\n" + "=" * 50)
    160: print("🎯 Testez maintenant:")
    161: print("python orchestrator/orchestrator.py --test")
    162: print("\n🔧 Si ça ne marche toujours pas, essayez:")
    163: print("python test_simple.py")