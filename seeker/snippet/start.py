#date: 2026-03-11T17:32:37Z
#url: https://api.github.com/gists/ff4f91fd2487b9d23843e8c09adeb659
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script de démarrage du pipeline
      4: """
      5: import subprocess
      6: import sys
      7: 
      8: print("🚀 DÉMARRAGE SMART CONTRACT PIPELINE")
      9: print("=" * 60)
     10: 
     11: def run_command(cmd, description):
     12:     """Exécute une commande"""
     13:     print(f"
     14: {description}...")
     15:     print(f"Commande: {cmd}")
     16:     
     17:     try:
     18:         result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
     19:         if result.returncode == 0:
     20:             print(f"✅ Succès")
     21:             if result.stdout:
     22:                 print(f"Sortie: {result.stdout[:200]}...")
     23:             return True
     24:         else:
     25:             print(f"❌ Échec (code: {result.returncode})")
     26:             if result.stderr:
     27:                 print(f"Erreur: {result.stderr[:200]}...")
     28:             return False
     29:     except Exception as e:
     30:         print(f"❌ Exception: {e}")
     31:         return False
     32: 
     33: # 1. Tester l'orchestrateur
     34: print("
     35: 1. Test de l'orchestrateur...")
     36: success = run_command(
     37:     f'"{sys.executable}" orchestrator/orchestrator.py --test',
     38:     "Test de santé de l'orchestrateur"
     39: )
     40: 
     41: if success:
     42:     print("
     43: " + "=" * 60)
     44:     print("🎉 PIPELINE OPÉRATIONNEL !")
     45:     print("=" * 60)
     46:     
     47:     print("
     48: Commandes disponibles:")
     49:     print("• Test de santé:    python orchestrator/orchestrator.py --test")
     50:     print("• Workflow test:    python orchestrator/orchestrator.py --workflow test")
     51:     print("• Mode interactif:  python orchestrator/orchestrator.py")
     52:     
     53:     print("
     54: Structure déployée:")
     55:     print("• 5 agents principaux (architect, coder, smart_contract, frontend_web3, tester)")
     56:     print("• 17 sous-agents spécialisés")
     57:     print("• Orchestrateur central")
     58:     
     59: else:
     60:     print("
     61: " + "=" * 60)
     62:     print("⚠️  PROBLÈME DÉTECTÉ")
     63:     print("=" * 60)
     64:     
     65:     print("
     66: Solutions:")
     67:     print("1. Vérifiez les dépendances: pip install PyYAML aiohttp")
     68:     print("2. Testez avec: python test_simple.py")
     69:     print("3. Recréez la structure: python deploy_pipeline.py --force")
     70:     
     71:     print("
     72: Test simple:")
     73:     run_command(f'"{sys.executable}" test_simple.py', "Test simple")
     74: 
     75: print("
     76: " + "=" * 60)