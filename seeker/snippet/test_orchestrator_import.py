#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Test d'import simple pour l'orchestrator
      4: """
      5: 
      6: import sys
      7: import os
      8: import traceback
      9: 
     10: print("\n" + "="*70)
     11: print("🔍 TEST D'IMPORT DE L'ORCHESTRATOR")
     12: print("="*70)
     13: 
     14: # Ajouter le chemin
     15: sys.path.insert(0, os.path.dirname(__file__))
     16: 
     17: try:
     18:     print("🔄 Tentative d'import: agents.orchestrator.agent")
     19:     module = __import__('agents.orchestrator.agent', fromlist=['OrchestratorAgent'])
     20:     
     21:     if hasattr(module, 'OrchestratorAgent'):
     22:         print(f"✅ Classe OrchestratorAgent trouvée")
     23:         agent_class = getattr(module, 'OrchestratorAgent')
     24:         
     25:         # Tester l'instanciation
     26:         try:
     27:             agent = agent_class()
     28:             print(f"✅ Agent instancié avec succès")
     29:         except Exception as e:
     30:             print(f"❌ Erreur instanciation: {e}")
     31:             traceback.print_exc()
     32:     else:
     33:         print(f"❌ Classe OrchestratorAgent non trouvée")
     34:         # Lister les classes disponibles
     35:         classes = [attr for attr in dir(module) if attr.endswith('Agent')]
     36:         print(f"   Classes trouvées: {classes}")
     37:         
     38: except ImportError as e:
     39:     print(f"❌ Erreur import: {e}")
     40:     traceback.print_exc()
     41:     
     42:     # Vérifier si le fichier existe
     43:     agent_path = os.path.join("agents", "orchestrator", "agent.py")
     44:     if os.path.exists(agent_path):
     45:         print(f"✅ Fichier trouvé: {agent_path}")
     46:         print(f"📄 Taille: {os.path.getsize(agent_path)} octets")
     47:     else:
     48:         print(f"❌ Fichier non trouvé: {agent_path}")
     49: 
     50: print("="*70)