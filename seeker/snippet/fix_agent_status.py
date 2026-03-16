#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script pour ajouter AgentStatus dans les imports
      4: """
      5: 
      6: import os
      7: import re
      8: from pathlib import Path
      9: 
     10: ROOT_DIR = Path("D:/Web3Projects/SmartContractDevPipeline")
     11: AGENTS_DIR = ROOT_DIR / "agents"
     12: 
     13: def fix_agent_file(file_path: Path):
     14:     """Ajoute AgentStatus dans les imports."""
     15:     print(f"\n📝 Traitement de: {file_path.relative_to(ROOT_DIR)}")
     16:     
     17:     with open(file_path, 'r', encoding='utf-8') as f:
     18:         content = f.read()
     19:     
     20:     # Remplacer l'import de BaseAgent seul par BaseAgent, AgentStatus
     21:     content = content.replace(
     22:         'from agents.base_agent.base_agent import BaseAgent',
     23:         'from agents.base_agent.base_agent import BaseAgent, AgentStatus'
     24:     )
     25:     
     26:     # Sauvegarder
     27:     with open(file_path, 'w', encoding='utf-8') as f:
     28:         f.write(content)
     29:     
     30:     print("  ✅ AgentStatus ajouté")
     31: 
     32: def main():
     33:     print("\n" + "="*70)
     34:     print("🔧 CORRECTION DES IMPORTS AGENTSTATUS")
     35:     print("="*70)
     36:     
     37:     # Trouver tous les fichiers agent.py
     38:     agent_files = list(AGENTS_DIR.rglob("agent.py"))
     39:     print(f"📊 {len(agent_files)} fichiers trouvés\n")
     40:     
     41:     for file_path in agent_files:
     42:         fix_agent_file(file_path)
     43:     
     44:     print("\n" + "="*70)
     45:     print("✅ CORRECTION TERMINÉE")
     46:     print("="*70)
     47: 
     48: if __name__ == "__main__":
     49:     main()