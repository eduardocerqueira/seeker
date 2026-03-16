#date: 2026-03-16T17:43:40Z
#url: https://api.github.com/gists/befd3d7c917b9b1c1c82a6930ce45c5a
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script pour corriger précisément l'indentation des fichiers agent.py
      4: """
      5: 
      6: import re
      7: from pathlib import Path
      8: 
      9: SOUS_AGENTS_PATH = Path("agents/communication/sous_agents")
     10: 
     11: SUB_AGENTS = [
     12:     "circuit_breaker",
     13:     "dead_letter_analyzer",
     14:     "message_router",
     15:     "performance_optimizer",
     16:     "pubsub_manager",
     17:     "queue_manager",
     18:     "security_validator"
     19: ]
     20: 
     21: def fix_indentation(content: str) -> str:
     22:     """Corrige l'indentation dans le contenu du fichier"""
     23:     
     24:     # Pattern pour trouver la section handlers avec la mauvaise indentation
     25:     pattern = r'(handlers = \{.*?\n)(.*?)(?=\n        \})'
     26:     
     27:     def replacer(match):
     28:         start = match.group(1)
     29:         middle = match.group(2)
     30:         
     31:         # Remplacer l'indentation incorrecte
     32:         lines = middle.split('\n')
     33:         corrected_lines = []
     34:         for line in lines:
     35:             if line.strip():
     36:                 # Compter les espaces actuels
     37:                 current_indent = len(line) - len(line.lstrip())
     38:                 # Si l'indentation est de 4, la passer à 8
     39:                 if current_indent == 4:
     40:                     line = '        ' + line.lstrip()
     41:                 corrected_lines.append(line)
     42:         
     43:         return start + '\n'.join(corrected_lines)
     44:     
     45:     # Appliquer la correction
     46:     content = re.sub(pattern, replacer, content, flags=re.DOTALL)
     47:     
     48:     return content
     49: 
     50: def fix_file(file_path: Path) -> bool:
     51:     """Corrige un fichier"""
     52:     print(f"\n🔧 Correction de {file_path}")
     53:     
     54:     with open(file_path, 'r', encoding='utf-8') as f:
     55:         content = f.read()
     56:     
     57:     # Sauvegarder l'original
     58:     backup = file_path.with_suffix('.py.bak2')
     59:     with open(backup, 'w', encoding='utf-8') as f:
     60:         f.write(content)
     61:     print(f"  ✅ Backup créé: {backup}")
     62:     
     63:     # Corriger l'indentation
     64:     new_content = fix_indentation(content)
     65:     
     66:     if new_content != content:
     67:         with open(file_path, 'w', encoding='utf-8') as f:
     68:             f.write(new_content)
     69:         print(f"  ✅ Indentation corrigée")
     70:         return True
     71:     else:
     72:         print(f"  ℹ️ Aucune modification nécessaire")
     73:         return False
     74: 
     75: def main():
     76:     print("=" * 60)
     77:     print("🔧 CORRECTION PRÉCISE DE L'INDENTATION")
     78:     print("=" * 60)
     79:     
     80:     fixed_count = 0
     81:     
     82:     for agent_id in SUB_AGENTS:
     83:         agent_file = SOUS_AGENTS_PATH / agent_id / "agent.py"
     84:         if agent_file.exists():
     85:             if fix_file(agent_file):
     86:                 fixed_count += 1
     87:     
     88:     print("\n" + "=" * 60)
     89:     print(f"✅ Correction terminée: {fixed_count} fichiers corrigés")
     90:     print("=" * 60)
     91:     print("\n📋 Prochaines étapes:")
     92:     print("   1. Relancez python diagnose_communication.py")
     93:     print("   2. Les sous-agents devraient maintenant s'importer")
     94: 
     95: if __name__ == "__main__":
     96:     main()