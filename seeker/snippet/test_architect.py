#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import sys
      2: import os
      3: sys.path.insert(0, '.')
      4: 
      5: print("="*60)
      6: print("🧪 TEST ARCHITECT AGENT")
      7: print("="*60)
      8: 
      9: try:
     10:     # Import
     11:     from agents.architect.architect import ArchitectAgent
     12:     print("✅ Import ArchitectAgent: RÉUSSI")
     13:     
     14:     from agents.base_agent.base_agent import AgentConfiguration
     15:     print("✅ Import AgentConfiguration: RÉUSSI")
     16:     
     17:     # Configuration
     18:     config = AgentConfiguration(
     19:         name="TestArchitect",
     20:         capabilities=["DESIGN_SYSTEM_ARCHITECTURE"],
     21:         description="Test"
     22:     )
     23:     
     24:     # Instanciation
     25:     agent = ArchitectAgent(config=config)
     26:     print(f"✅ Instanciation: RÉUSSI ({agent.__class__.__name__})")
     27:     print(f"   - Nom: {agent.name}")
     28:     print(f"   - Capabilités: {len(agent.capabilities)}")
     29:     
     30:     # Test tâche
     31:     if hasattr(agent, 'execute_task'):
     32:         result = agent.execute_task("validate_config")
     33:         print(f"✅ Tâche exécutée: {result.get('status', 'N/A')}")
     34:     
     35:     print("\n" + "="*60)
     36:     print("🎉 TOUS LES TESTS RÉUSSIS !")
     37:     print("="*60)
     38:     
     39: except Exception as e:
     40:     print(f"\n❌ ERREUR: {e}")
     41:     import traceback
     42:     traceback.print_exc()