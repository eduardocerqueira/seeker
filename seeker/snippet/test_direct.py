#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import sys
      2: import os
      3: import importlib.util
      4: 
      5: print("="*60)
      6: print("🚀 IMPORT DIRECT - Contournement du bug")
      7: print("="*60)
      8: 
      9: # Chemin absolu vers architect.py
     10: architect_path = os.path.abspath("agents/architect/architect.py")
     11: print(f"Chemin: {architect_path}")
     12: 
     13: if not os.path.exists(architect_path):
     14:     print("❌ Fichier non trouvé!")
     15:     exit(1)
     16: 
     17: # Import direct avec importlib
     18: try:
     19:     spec = importlib.util.spec_from_file_location("ArchitectAgent", architect_path)
     20:     architect_module = importlib.util.module_from_spec(spec)
     21:     
     22:     # Exécuter le module
     23:     spec.loader.exec_module(architect_module)
     24:     
     25:     # Récupérer la classe
     26:     ArchitectAgent = getattr(architect_module, "ArchitectAgent")
     27:     print("✅ Classe ArchitectAgent chargée directement")
     28:     
     29:     # Créer une config simple
     30:     class SimpleConfig:
     31:         def __init__(self, **kwargs):
     32:             self.__dict__.update(kwargs)
     33:     
     34:     config = SimpleConfig(
     35:         name="DirectArchitect",
     36:         capabilities=["DESIGN"],
     37:         description="Test direct"
     38:     )
     39:     
     40:     # Instancier
     41:     agent = ArchitectAgent(config)
     42:     print(f"✅ Agent instancié: {agent.name}")
     43:     print(f"   - Capabilités: {len(agent.capabilities)}")
     44:     
     45: except Exception as e:
     46:     print(f"❌ Erreur: {e}")
     47:     import traceback
     48:     traceback.print_exc()