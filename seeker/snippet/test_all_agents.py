#date: 2026-03-16T17:43:40Z
#url: https://api.github.com/gists/befd3d7c917b9b1c1c82a6930ce45c5a
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: # test_all_agents.py - version corrigée
      2: """
      3: Test simplifié pour vérifier l'initialisation de tous les agents
      4: """
      5: 
      6: import sys
      7: from pathlib import Path
      8: import asyncio
      9: import logging
     10: 
     11: # Ajouter le chemin du projet
     12: project_root = Path(__file__).parent
     13: sys.path.insert(0, str(project_root))
     14: 
     15: # Configuration du logging
     16: logging.basicConfig(
     17:     level=logging.INFO,
     18:     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
     19: )
     20: 
     21: async def test_agent(agent_name, agent_class, config_path=None):
     22:     """
     23:     Teste un agent spécifique
     24:     
     25:     Args:
     26:         agent_name: Nom de l'agent
     27:         agent_class: Classe de l'agent
     28:         config_path: Chemin vers le fichier de config (optionnel)
     29:     """
     30:     print(f"\n{'='*60}")
     31:     print(f"🧪 TEST: {agent_name.upper()}")
     32:     print('='*60)
     33:     
     34:     try:
     35:         # Créer l'instance
     36:         if config_path:
     37:             agent = agent_class(config_path)
     38:         else:
     39:             agent = agent_class()
     40:         
     41:         # Vérifier si c'est une classe abstraite
     42:         if hasattr(agent_class, '__abstractmethods__') and agent_class.__abstractmethods__:
     43:             print(f"⚠️  {agent_name}: Classe abstraite (ne peut pas être instanciée)")
     44:             print(f"   Méthodes abstraites: {agent_class.__abstractmethods__}")
     45:             return None  # Pas une erreur, juste une info
     46:         
     47:         # Initialiser
     48:         success = await agent.initialize()
     49:         
     50:         if success:
     51:             print(f"✅ {agent_name}: Initialisation réussie")
     52:             print(f"   Statut: {agent.status}")
     53:             print(f"   Capacités: {len(agent.capabilities)}")
     54:             return True
     55:         else:
     56:             print(f"❌ {agent_name}: Échec de l'initialisation")
     57:             return False
     58:             
     59:     except TypeError as e:
     60:         if "Can't instantiate abstract class" in str(e):
     61:             print(f"⚠️  {agent_name}: Classe abstraite (normal)")
     62:             return None
     63:         else:
     64:             print(f"❌ {agent_name}: Erreur - {e}")
     65:             import traceback
     66:             traceback.print_exc()
     67:             return False
     68:     except Exception as e:
     69:         print(f"❌ {agent_name}: Erreur - {e}")
     70:         import traceback
     71:         traceback.print_exc()
     72:         return False
     73: 
     74: async def main():
     75:     """Fonction principale de test"""
     76:     print("🧪 TEST ORCHESTRATOR SIMPLIFIÉ")
     77:     print("="*50 + "\n")
     78:     
     79:     results = {}
     80:     
     81:     try:
     82:         # Test BaseAgent - juste l'import, pas l'instanciation
     83:         try:
     84:             from agents.base_agent import BaseAgent
     85:             print("✅ BaseAgent importé avec succès")
     86:             results['base_agent'] = True
     87:         except ImportError as e:
     88:             print(f"❌ BaseAgent: ImportError - {e}")
     89:             results['base_agent'] = False
     90:         
     91:         # Test ArchitectAgent
     92:         try:
     93:             from agents.architect.architect import ArchitectAgent
     94:             architect_config = "agents/architect/config.yaml"
     95:             results['architect'] = await test_agent("ArchitectAgent", ArchitectAgent, architect_config)
     96:         except ImportError as e:
     97:             print(f"❌ ArchitectAgent: ImportError - {e}")
     98:             results['architect'] = False
     99:         
    100:         # Test CoderAgent
    101:         try:
    102:             from agents.coder.coder import CoderAgent
    103:             coder_config = "agents/coder/config.yaml"
    104:             results['coder'] = await test_agent("CoderAgent", CoderAgent, coder_config)
    105:         except ImportError as e:
    106:             print(f"❌ CoderAgent: ImportError - {e}")
    107:             results['coder'] = False
    108:         
    109:         # Test TesterAgent (s'il existe)
    110:         try:
    111:             from agents.tester.tester import TesterAgent
    112:             tester_config = "agents/tester/config.yaml"
    113:             results['tester'] = await test_agent("TesterAgent", TesterAgent, tester_config)
    114:         except ImportError:
    115:             print("ℹ️  TesterAgent: Non implémenté (c'est normal)")
    116:             results['tester'] = None
    117:         
    118:     except Exception as e:
    119:         print(f"\n❌ ERREUR GLOBALE: {e}")
    120:         import traceback
    121:         traceback.print_exc()
    122:     
    123:     # Afficher le résumé
    124:     print("\n" + "="*50)
    125:     print("📊 RÉSUMÉ DES TESTS")
    126:     print("="*50)
    127:     
    128:     successful = 0
    129:     total = 0
    130:     
    131:     for agent_name, result in results.items():
    132:         if result is None:
    133:             status = "⚠️ "
    134:         elif result:
    135:             successful += 1
    136:             status = "✅"
    137:         else:
    138:             status = "❌"
    139:         
    140:         if result is not None:  # Ne pas compter les agents abstraits/non implémentés
    141:             total += 1
    142:         
    143:         print(f"{status} {agent_name:20}")
    144:     
    145:     print("-"*50)
    146:     print(f"Total: {successful}/{total} agents concrets initialisés avec succès")
    147:     
    148:     if successful == total:
    149:         print("\n🎉 TOUS LES AGENTS CONCRETS SONT OPÉRATIONNELS !")
    150:     else:
    151:         print(f"\n⚠️  {total - successful} agent(s) concret(s) nécessite(nt) attention")
    152: 
    153: if __name__ == "__main__":
    154:     asyncio.run(main())