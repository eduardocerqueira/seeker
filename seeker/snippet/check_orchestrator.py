#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script de diagnostic pour l'orchestrator
      4: """
      5: 
      6: import os
      7: import sys
      8: import importlib
      9: 
     10: print("\n" + "="*70)
     11: print("🔍 DIAGNOSTIC DE L'ORCHESTRATOR")
     12: print("="*70)
     13: 
     14: # Vérifier le fichier
     15: orchestrator_path = os.path.join("agents", "orchestrator", "agent.py")
     16: if os.path.exists(orchestrator_path):
     17:     print(f"✅ Fichier trouvé: {orchestrator_path}")
     18:     print(f"📄 Taille: {os.path.getsize(orchestrator_path)} octets")
     19: else:
     20:     print(f"❌ Fichier introuvable: {orchestrator_path}")
     21: 
     22: # Essayer d'importer
     23: try:
     24:     print("\n🔄 Tentative d'import...")
     25:     module = importlib.import_module("agents.orchestrator.agent")
     26:     print(f"✅ Module importé avec succès")
     27:     
     28:     # Lister les classes
     29:     classes = [attr for attr in dir(module) if attr.endswith('Agent')]
     30:     print(f"📋 Classes trouvées: {classes}")
     31:     
     32:     if 'OrchestratorAgent' in classes:
     33:         AgentClass = getattr(module, 'OrchestratorAgent')
     34:         print(f"✅ Classe OrchestratorAgent trouvée")
     35:         
     36:         # Tester l'instanciation
     37:         try:
     38:             agent = AgentClass()
     39:             print(f"✅ Agent instancié avec succès")
     40:         except Exception as e:
     41:             print(f"❌ Erreur instanciation: {e}")
     42:     else:
     43:         print(f"❌ Classe OrchestratorAgent non trouvée")
     44:         
     45: except Exception as e:
     46:     print(f"❌ Erreur import: {e}")
     47: 
     48: print("\n" + "="*70)