#date: 2026-03-11T17:32:37Z
#url: https://api.github.com/gists/ff4f91fd2487b9d23843e8c09adeb659
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: # diagnose_communication_v2.py
      2: import sys
      3: from pathlib import Path
      4: 
      5: # Ajouter le chemin du projet
      6: project_root = Path(__file__).parent
      7: if str(project_root) not in sys.path:
      8:     sys.path.insert(0, str(project_root))
      9: 
     10: from agents.communication.agent import CommunicationAgent
     11: import yaml
     12: import importlib
     13: 
     14: async def diagnose():
     15:     print("="*60)
     16:     print("🔍 DIAGNOSTIC V2 - AGENT COMMUNICATION")
     17:     print("="*60)
     18:     
     19:     # 1. Vérifier la configuration
     20:     print("\n📁 1. Configuration")
     21:     config_path = Path("agents/communication/config.yaml")
     22:     with open(config_path, 'r', encoding='utf-8') as f:
     23:         config = yaml.safe_load(f)
     24:     
     25:     subagents = config.get('subAgents', [])
     26:     print(f"   • Sous-agents configurés: {len(subagents)}")
     27:     
     28:     # 2. Créer l'agent
     29:     print("\n🤖 2. Création de l'agent")
     30:     agent = CommunicationAgent()
     31:     
     32:     # 3. Tenter d'initialiser les sous-agents manuellement
     33:     print("\n🔄 3. Initialisation manuelle des sous-agents")
     34:     await agent._initialize_sub_agents()
     35:     print(f"   • Sous-agents chargés: {len(agent._sub_agents)}")
     36:     
     37:     # 4. Test d'import avec le bon chemin
     38:     print("\n📦 4. Test d'import avec le bon chemin")
     39:     for sa in subagents:
     40:         agent_id = sa['id']
     41:         module_path = f"agents.communication.sous_agents.{agent_id}.agent"
     42:         try:
     43:             module = importlib.import_module(module_path)
     44:             print(f"   ✅ {agent_id}: module importé")
     45:             
     46:             class_name = ''.join(p.capitalize() for p in agent_id.split('_')) + 'SubAgent'
     47:             agent_class = getattr(module, class_name, None)
     48:             if agent_class:
     49:                 print(f"      ✅ Classe {class_name} trouvée")
     50:                 
     51:                 # Tenter d'instancier
     52:                 instance = agent_class()
     53:                 print(f"      ✅ Instance créée")
     54:             else:
     55:                 print(f"      ❌ Classe {class_name} non trouvée")
     56:         except Exception as e:
     57:             print(f"   ❌ {agent_id}: {e}")
     58: 
     59: if __name__ == "__main__":
     60:     import asyncio
     61:     asyncio.run(diagnose())