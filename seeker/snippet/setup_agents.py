#date: 2026-03-11T17:32:37Z
#url: https://api.github.com/gists/ff4f91fd2487b9d23843e8c09adeb659
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: # setup_agents.py
      2: """
      3: Script de configuration initiale pour les agents
      4: """
      5: 
      6: import os
      7: import sys
      8: from pathlib import Path
      9: 
     10: def create_init_files():
     11:     """Crée les fichiers __init__.py manquants"""
     12:     project_root = Path(__file__).parent
     13:     
     14:     # Structure des agents
     15:     agents_structure = {
     16:         "agents": [
     17:             "base_agent",
     18:             "architect", 
     19:             "coder",
     20:             "tester",
     21:             "smart_contract",
     22:             "registry",
     23:             "formal_verification",
     24:             "fuzzing_simulation",
     25:             "documenter",
     26:             "frontend_web3",
     27:             "communication",
     28:             "storage",
     29:             "monitoring",
     30:             "learning",
     31:             "workflow"
     32:         ]
     33:     }
     34:     
     35:     # Contenu minimal des __init__.py
     36:     init_content = '''"""
     37: {agent_name} Agent Package
     38: """
     39: '''
     40: 
     41:     for agent_dir in agents_structure["agents"]:
     42:         agent_path = project_root / "agents" / agent_dir
     43:         init_file = agent_path / "__init__.py"
     44:         
     45:         # Créer le répertoire s'il n'existe pas
     46:         agent_path.mkdir(parents=True, exist_ok=True)
     47:         
     48:         # Créer le __init__.py s'il n'existe pas
     49:         if not init_file.exists():
     50:             content = init_content.format(agent_name=agent_dir.replace('_', ' ').title())
     51:             init_file.write_text(content)
     52:             print(f"✅ Créé: {init_file.relative_to(project_root)}")
     53:     
     54:     # Créer le __init__.py racine des agents
     55:     agents_root = project_root / "agents" / "__init__.py"
     56:     if not agents_root.exists():
     57:         agents_root.write_text('''"""
     58: Agents Package - Tous les agents du système
     59: """
     60: 
     61: __version__ = "2.2.0"
     62: ''')
     63:         print(f"✅ Créé: {agents_root.relative_to(project_root)}")
     64: 
     65: def check_structure():
     66:     """Vérifie la structure des agents"""
     67:     project_root = Path(__file__).parent
     68:     required = [
     69:         "agents/base_agent/base_agent.py",
     70:         "agents/base_agent/config.yaml",
     71:         "agents/architect/architect.py", 
     72:         "agents/architect/config.yaml",
     73:         "agents/coder/coder.py",
     74:         "agents/coder/config.yaml"
     75:     ]
     76:     
     77:     print("🔍 Vérification de la structure...")
     78:     
     79:     for path in required:
     80:         full_path = project_root / path
     81:         if full_path.exists():
     82:             print(f"✅ {path}")
     83:         else:
     84:             print(f"❌ {path} - MANQUANT")
     85:     
     86:     # Vérifier les __init__.py
     87:     print("\n🔍 Vérification des packages...")
     88:     for agent_dir in ["base_agent", "architect", "coder"]:
     89:         init_file = project_root / "agents" / agent_dir / "__init__.py"
     90:         if init_file.exists():
     91:             print(f"✅ agents/{agent_dir}/__init__.py")
     92:         else:
     93:             print(f"❌ agents/{agent_dir}/__init__.py - MANQUANT")
     94: 
     95: def main():
     96:     """Fonction principale"""
     97:     print("🛠️  CONFIGURATION DES AGENTS")
     98:     print("="*40)
     99:     
    100:     create_init_files()
    101:     print()
    102:     check_structure()
    103:     
    104:     print("\n" + "="*40)
    105:     print("📋 PROCHAINES ÉTAPES:")
    106:     print("1. Exécuter: python setup_agents.py")
    107:     print("2. Exécuter: python test_all_agents.py")
    108:     print("3. Vérifier que tous les agents s'initialisent")
    109:     print("="*40)
    110: 
    111: if __name__ == "__main__":
    112:     main()