#date: 2026-03-11T17:32:37Z
#url: https://api.github.com/gists/ff4f91fd2487b9d23843e8c09adeb659
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Script de correction des sous-agents communication
      4: Corrige les erreurs d'annotation de type et autres problèmes
      5: """
      6: 
      7: import re
      8: from pathlib import Path
      9: 
     10: # Configuration
     11: SOUS_AGENTS_PATH = Path("agents/communication/sous_agents")
     12: 
     13: # Liste des sous-agents à corriger
     14: SUB_AGENTS = [
     15:     "queue_manager",
     16:     "pubsub_manager", 
     17:     "circuit_breaker",
     18:     "message_router",
     19:     "dead_letter_analyzer",
     20:     "performance_optimizer",
     21:     "security_validator"
     22: ]
     23: 
     24: def fix_agent_file(agent_file: Path) -> bool:
     25:     """Corrige les erreurs dans un fichier agent.py"""
     26:     print(f"\n🔧 Analyse de {agent_file}")
     27:     
     28:     with open(agent_file, 'r', encoding='utf-8') as f:
     29:         content = f.read()
     30:     
     31:     modified = False
     32:     
     33:     # 1. Corriger l'annotation de type problématique
     34:     # Remplacer "self._task_metrics: Dict[str, Dict] = {}" par "self._task_metrics = {}"
     35:     pattern1 = r'self\._task_metrics: Dict\[str, Dict\] = \{\}'
     36:     replacement1 = 'self._task_metrics = {}'
     37:     
     38:     if re.search(pattern1, content):
     39:         content = re.sub(pattern1, replacement1, content)
     40:         print(f"  ✅ Annotation de type corrigée")
     41:         modified = True
     42:     
     43:     # 2. Vérifier que le logger est correctement défini
     44:     if "logger = logging.getLogger(__name__)" not in content:
     45:         print(f"  ⚠️ Logger manquant - à vérifier")
     46:     
     47:     # 3. Vérifier que la méthode __init__ a bien les doubles accolades
     48:     # (c'est déjà bon normalement)
     49:     
     50:     if modified:
     51:         # Sauvegarder le fichier corrigé
     52:         with open(agent_file, 'w', encoding='utf-8') as f:
     53:             f.write(content)
     54:         print(f"  ✅ Fichier mis à jour")
     55:         return True
     56:     else:
     57:         print(f"  ℹ️ Aucune correction nécessaire")
     58:         return False
     59: 
     60: def main():
     61:     print("=" * 60)
     62:     print("🔧 CORRECTION DES SOUS-AGENTS COMMUNICATION")
     63:     print("=" * 60)
     64:     
     65:     if not SOUS_AGENTS_PATH.exists():
     66:         print(f"❌ Dossier {SOUS_AGENTS_PATH} introuvable")
     67:         return
     68:     
     69:     fixed_count = 0
     70:     error_count = 0
     71:     
     72:     for agent_id in SUB_AGENTS:
     73:         agent_dir = SOUS_AGENTS_PATH / agent_id
     74:         agent_file = agent_dir / "agent.py"
     75:         
     76:         if not agent_file.exists():
     77:             print(f"\n⚠️ Fichier non trouvé: {agent_file}")
     78:             error_count += 1
     79:             continue
     80:         
     81:         try:
     82:             if fix_agent_file(agent_file):
     83:                 fixed_count += 1
     84:         except Exception as e:
     85:             print(f"❌ Erreur lors de la correction de {agent_id}: {e}")
     86:             error_count += 1
     87:     
     88:     print("\n" + "=" * 60)
     89:     print(f"✅ CORRECTION TERMINÉE")
     90:     print(f"   • Fichiers corrigés: {fixed_count}")
     91:     print(f"   • Erreurs: {error_count}")
     92:     print("=" * 60)
     93: 
     94: if __name__ == "__main__":
     95:     main()