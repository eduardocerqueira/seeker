#date: 2026-03-11T17:32:37Z
#url: https://api.github.com/gists/ff4f91fd2487b9d23843e8c09adeb659
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Test final simplifié du pipeline
      4: """
      5: import os
      6: import sys
      7: import asyncio
      8: 
      9: print("🧪 TEST FINAL SIMPLIFIÉ")
     10: print("=" * 60)
     11: 
     12: async def test_simple():
     13:     """Test simple"""
     14:     
     15:     # Configuration
     16:     project_root = os.path.abspath(".")
     17:     if project_root not in sys.path:
     18:         sys.path.insert(0, project_root)
     19:     
     20:     print(f"📁 Projet: {project_root}")
     21:     
     22:     print("
     23: 1. Test d'import de l'orchestrateur...")
     24:     try:
     25:         from orchestrator.orchestrator import Orchestrator
     26:         print("✅ Orchestrateur importé")
     27:     except Exception as e:
     28:         print(f"❌ Erreur: {e}")
     29:         return False
     30:     
     31:     print("
     32: 2. Création de l'orchestrateur...")
     33:     try:
     34:         orchestrator = Orchestrator()
     35:         print("✅ Orchestrateur créé")
     36:     except Exception as e:
     37:         print(f"❌ Erreur: {e}")
     38:         return False
     39:     
     40:     print("
     41: 3. Initialisation des agents...")
     42:     try:
     43:         await orchestrator.initialize_agents()
     44:         print(f"✅ Agents initialisés: {len(orchestrator.agents)}")
     45:     except Exception as e:
     46:         print(f"❌ Erreur: {e}")
     47:         return False
     48:     
     49:     print("
     50: 4. Test de santé...")
     51:     try:
     52:         health = await orchestrator.health_check()
     53:         print(f"✅ Santé vérifiée")
     54:         print(f"   Orchestrateur: {health.get('orchestrator', 'N/A')}")
     55:         print(f"   Agents: {len(health.get('agents', {}))}")
     56:     except Exception as e:
     57:         print(f"❌ Erreur: {e}")
     58:         return False
     59:     
     60:     return True
     61: 
     62: async def main():
     63:     """Fonction principale"""
     64:     success = await test_simple()
     65:     
     66:     print("
     67: " + "=" * 60)
     68:     
     69:     if success:
     70:         print("🎉 TEST RÉUSSI !")
     71:         print("
     72: Votre pipeline est fonctionnel.")
     73:         print("
     74: Pour utiliser l'orchestrateur:")
     75:         print("python orchestrator/orchestrator.py --test")
     76:     else:
     77:         print("❌ TEST ÉCHOUÉ")
     78:         print("
     79: Prochaines étapes:")
     80:         print("1. Vérifiez la structure des dossiers")
     81:         print("2. Vérifiez que les fichiers existent:")
     82:         print("   - base_agent.py")
     83:         print("   - agents/*/agent.py")
     84:         print("   - orchestrator/orchestrator.py")
     85: 
     86: if __name__ == "__main__":
     87:     asyncio.run(main())