#date: 2026-03-16T17:43:40Z
#url: https://api.github.com/gists/befd3d7c917b9b1c1c82a6930ce45c5a
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script de correction automatique des agents
      4: """
      5: import os
      6: import re
      7: 
      8: def fix_agent_file(filepath, agent_class_name):
      9:     """Corrige un fichier agent.py"""
     10:     with open(filepath, 'r', encoding='utf-8') as f:
     11:         content = f.read()
     12:     
     13:     # 1. Corriger les imports
     14:     old_import = """import os
     15: import sys
     16: sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
     17: 
     18: from base_agent import BaseAgent"""
     19:     
     20:     new_import = """import os
     21: import sys
     22: 
     23: # Correction du chemin pour importer base_agent
     24: current_dir = os.path.dirname(os.path.abspath(__file__))
     25: agents_root = os.path.dirname(current_dir)  # Remonte à agents/
     26: 
     27: # Ajouter agents/ au path si pas déjà présent
     28: if agents_root not in sys.path:
     29:     sys.path.insert(0, agents_root)
     30: 
     31: from base_agent.agent import BaseAgent"""
     32:     
     33:     if old_import in content:
     34:         content = content.replace(old_import, new_import)
     35:         print(f"  ✅ Imports corrigés dans {os.path.basename(filepath)}")
     36:     
     37:     # 2. Corriger le constructeur pour passer le nom de l'agent
     38:     constructor_pattern = r'def __init__\(self, config_path: str = None\):'
     39:     replacement = f'def __init__(self, config_path: str = None):\n        super().__init__(config_path, "{agent_class_name}")'
     40:     
     41:     if re.search(constructor_pattern, content):
     42:         content = re.sub(constructor_pattern, replacement, content)
     43:         print(f"  ✅ Constructeur corrigé pour {agent_class_name}")
     44:     
     45:     # Sauvegarder
     46:     with open(filepath, 'w', encoding='utf-8') as f:
     47:         f.write(content)
     48:     
     49:     return content
     50: 
     51: def main():
     52:     print("🔧 CORRECTION AUTOMATIQUE DES AGENTS")
     53:     print("=" * 50)
     54:     
     55:     # Agents à corriger
     56:     agents = [
     57:         ("agents/architect/agent.py", "ArchitectAgent"),
     58:         ("agents/coder/agent.py", "CoderAgent"),
     59:         ("agents/smart_contract/agent.py", "SmartContractAgent"),
     60:         ("agents/frontend_web3/agent.py", "FrontendWeb3Agent"),
     61:         ("agents/tester/agent.py", "TesterAgent")
     62:     ]
     63:     
     64:     # 1. Créer agents/base_agent/__init__.py
     65:     init_file = "agents/base_agent/__init__.py"
     66:     if not os.path.exists(init_file):
     67:         os.makedirs(os.path.dirname(init_file), exist_ok=True)
     68:         with open(init_file, 'w') as f:
     69:             f.write('from .agent import BaseAgent, AgentUtils\n\n__all__ = ["BaseAgent", "AgentUtils"]')
     70:         print(f"✅ Créé: {init_file}")
     71:     
     72:     # 2. Corriger base_agent/agent.py
     73:     base_agent_file = "agents/base_agent/agent.py"
     74:     if os.path.exists(base_agent_file):
     75:         with open(base_agent_file, 'r') as f:
     76:             content = f.read()
     77:         
     78:         # Corriger l'ordre d'initialisation
     79:         old_init = """def __init__(self, config_path: Optional[str] = None, agent_name: Optional[str] = None):
     80:     self.config = {}
     81:     self.name = agent_name or self.__class__.__name__
     82:     self.agent_type = self._get_agent_type()
     83:     self.logger = logging.getLogger(self.name)"""
     84:         
     85:         new_init = """def __init__(self, config_path: Optional[str] = None, agent_name: Optional[str] = None):
     86:     # D'ABORD déterminer le nom
     87:     if agent_name:
     88:         self.name = agent_name
     89:     else:
     90:         self.name = self.__class__.__name__
     91:     
     92:     # MAINTENANT initialiser le logger
     93:     self.logger = logging.getLogger(self.name)
     94:     
     95:     # Puis le reste
     96:     self.config = {}
     97:     self.agent_type = self._get_agent_type()"""
     98:         
     99:         if old_init in content:
    100:             content = content.replace(old_init, new_init)
    101:             with open(base_agent_file, 'w') as f:
    102:                 f.write(content)
    103:             print(f"✅ BaseAgent corrigé")
    104:     
    105:     # 3. Corriger les 5 agents principaux
    106:     for filepath, class_name in agents:
    107:         if os.path.exists(filepath):
    108:             fix_agent_file(filepath, class_name)
    109:         else:
    110:             print(f"⚠️  Fichier non trouvé: {filepath}")
    111:     
    112:     print("\n" + "=" * 50)
    113:     print("✅ CORRECTIONS APPLIQUÉES")
    114:     print("\nProchaines étapes:")
    115:     print("1. Testez avec: python orchestrator/orchestrator.py --test")
    116:     print("2. Si erreur persistante, vérifiez le chemin Python")
    117:     print("3. Exécutez depuis SmartContractDevPipeline/")
    118: 
    119: if __name__ == "__main__":
    120:     main()