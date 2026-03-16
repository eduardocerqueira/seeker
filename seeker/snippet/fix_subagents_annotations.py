#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: # fix_subagents_annotations.py
      2: """
      3: Script pour corriger les annotations de type dans les sous-agents communication
      4: """
      5: 
      6: import re
      7: from pathlib import Path
      8: 
      9: # Chemin vers les sous-agents
     10: subagents_path = Path("agents/communication/sous_agents")
     11: 
     12: # Parcourir tous les sous-agents
     13: for agent_dir in subagents_path.iterdir():
     14:     if agent_dir.is_dir():
     15:         agent_file = agent_dir / "agent.py"
     16:         if agent_file.exists():
     17:             print(f"🔧 Correction de {agent_file}")
     18:             
     19:             # Lire le contenu
     20:             with open(agent_file, 'r', encoding='utf-8') as f:
     21:                 content = f.read()
     22:             
     23:             # Remplacer les annotations problématiques
     24:             # Version originale avec l'erreur
     25:             original = "self._task_metrics: Dict[str, Dict] = {}"
     26:             # Version corrigée
     27:             corrected = "self._task_metrics = {}"
     28:             
     29:             if original in content:
     30:                 content = content.replace(original, corrected)
     31:                 print(f"  ✅ Annotation corrigée dans {agent_file}")
     32:                 
     33:                 # Écrire le fichier corrigé
     34:                 with open(agent_file, 'w', encoding='utf-8') as f:
     35:                     f.write(content)
     36:             else:
     37:                 print(f"  ℹ️ Annotation non trouvée dans {agent_file}")
     38: 
     39: print("\n✅ Corrections terminées !")