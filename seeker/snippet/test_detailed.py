#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import sys
      2: import os
      3: import traceback
      4: sys.path.insert(0, '.')
      5: 
      6: print("="*60)
      7: print("🧪 TEST DÉTAILLÉ ARCHITECT AGENT")
      8: print("="*60)
      9: 
     10: try:
     11:     # 1. Import avec plus de logs
     12:     print("1. Import ArchitectAgent...")
     13:     from agents.architect.architect import ArchitectAgent
     14:     print("   ✅ Import réussi")
     15:     
     16:     # 2. Import AgentConfiguration
     17:     print("2. Import AgentConfiguration...")
     18:     from agents.base_agent.base_agent import AgentConfiguration
     19:     print("   ✅ Import réussi")
     20:     
     21:     # 3. Création configuration
     22:     print("3. Création configuration...")
     23:     config = AgentConfiguration(
     24:         name="TestArchitect",
     25:         capabilities=["DESIGN_SYSTEM_ARCHITECTURE"],
     26:         description="Agent de test"
     27:     )
     28:     print("   ✅ Configuration créée")
     29:     
     30:     # 4. Instanciation avec try-catch détaillé
     31:     print("4. Instanciation de l'agent...")
     32:     try:
     33:         agent = ArchitectAgent(config=config)
     34:         print(f"   ✅ Instanciation réussie: {agent.__class__.__name__}")
     35:         
     36:         # Vérifier les attributs
     37:         print(f"   - Nom: {getattr(agent, 'name', 'NON DÉFINI')}")
     38:         print(f"   - Capabilités: {len(getattr(agent, 'capabilities', []))}")
     39:         print(f"   - Statut: {getattr(agent, 'status', 'NON DÉFINI')}")
     40:         
     41:     except Exception as inst_error:
     42:         print(f"   ❌ Erreur instanciation: {inst_error}")
     43:         print("   Stack trace:")
     44:         traceback.print_exc()
     45:     
     46: except ImportError as e:
     47:     print(f"❌ ImportError: {e}")
     48:     traceback.print_exc()
     49: except Exception as e:
     50:     print(f"❌ Autre erreur: {e}")
     51:     traceback.print_exc()
     52: 
     53: print("\n" + "="*60)
     54: print("FIN DU TEST")
     55: print("="*60)