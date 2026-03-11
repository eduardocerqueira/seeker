#date: 2026-03-11T17:32:37Z
#url: https://api.github.com/gists/ff4f91fd2487b9d23843e8c09adeb659
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import os
      2: import yaml
      3: from pathlib import Path
      4: 
      5: def find_sub_agents(agent_path):
      6:     """Trouve tous les sous-agents dans le dossier sous_agents/ d'un agent"""
      7:     sub_agents = []
      8:     sous_agents_dir = agent_path / 'sous_agents'
      9:     
     10:     if not sous_agents_dir.exists():
     11:         return sub_agents
     12:     
     13:     for sub_dir in sous_agents_dir.iterdir():
     14:         if sub_dir.is_dir():
     15:             config_file = sub_dir / 'config.yaml'
     16:             if config_file.exists():
     17:                 # Lire la config du sous-agent pour obtenir son nom
     18:                 try:
     19:                     with open(config_file, 'r', encoding='utf-8') as f:
     20:                         sub_config = yaml.safe_load(f)
     21:                     
     22:                     # Extraire le nom du sous-agent
     23:                     sub_name = None
     24:                     if sub_config and 'sub_agent' in sub_config:
     25:                         sub_name = sub_config['sub_agent'].get('name')
     26:                     elif sub_config and 'agent' in sub_config:
     27:                         sub_name = sub_config['agent'].get('name')
     28:                     
     29:                     sub_agents.append({
     30:                         'id': sub_dir.name,
     31:                         'name': sub_name or sub_dir.name.replace('_', ' ').title(),
     32:                         'display_name': sub_name or sub_dir.name.replace('_', ' ').title(),
     33:                         'config_path': f"agents/{agent_path.name}/sous_agents/{sub_dir.name}/config.yaml",
     34:                         'dependencies': [agent_path.name]
     35:                     })
     36:                     print(f"    ✅ Sous-agent trouvé: {sub_dir.name}")
     37:                 except Exception as e:
     38:                     print(f"    ⚠ Erreur lecture {sub_dir.name}: {e}")
     39:     
     40:     return sub_agents
     41: 
     42: def update_agent_config(agent_path):
     43:     """Met à jour le fichier config.yaml d'un agent pour inclure ses sous-agents"""
     44:     config_file = agent_path / 'config.yaml'
     45:     
     46:     if not config_file.exists():
     47:         print(f"  ⚠ Fichier config.yaml non trouvé dans {agent_path}")
     48:         return
     49:     
     50:     print(f"\n📝 Traitement de {agent_path.name}...")
     51:     
     52:     # Lire la config existante
     53:     with open(config_file, 'r', encoding='utf-8') as f:
     54:         config = yaml.safe_load(f)
     55:     
     56:     # Chercher les sous-agents
     57:     sub_agents = find_sub_agents(agent_path)
     58:     
     59:     if not sub_agents:
     60:         print(f"  ℹ Aucun sous-agent trouvé pour {agent_path.name}")
     61:         return
     62:     
     63:     print(f"  📦 {len(sub_agents)} sous-agent(s) trouvé(s)")
     64:     
     65:     # S'assurer que la structure agent existe
     66:     if 'agent' not in config:
     67:         config = {'agent': config}
     68:     
     69:     # Ajouter ou mettre à jour la section subAgents
     70:     if 'subAgents' not in config['agent']:
     71:         config['agent']['subAgents'] = []
     72:     
     73:     # Ajouter les nouveaux sous-agents (éviter les doublons)
     74:     existing_ids = {sub['id'] for sub in config['agent']['subAgents']}
     75:     
     76:     for sub in sub_agents:
     77:         if sub['id'] not in existing_ids:
     78:             config['agent']['subAgents'].append(sub)
     79:             print(f"    ✅ Ajouté: {sub['name']}")
     80:     
     81:     # Sauvegarder le fichier
     82:     backup_file = config_file.with_suffix('.yaml.backup')
     83:     if not backup_file.exists():
     84:         os.rename(config_file, backup_file)
     85:         print(f"  ✅ Backup créé: {backup_file.name}")
     86:     
     87:     with open(config_file, 'w', encoding='utf-8') as f:
     88:         yaml.dump(config, f, allow_unicode=True, sort_keys=False, indent=2)
     89:     
     90:     print(f"  ✅ Fichier mis à jour: {config_file.name}")
     91: 
     92: def main():
     93:     # Chemin vers le dossier agents
     94:     agents_dir = Path(__file__).parent / 'agents'
     95:     
     96:     if not agents_dir.exists():
     97:         print(f"❌ Dossier agents/ non trouvé: {agents_dir}")
     98:         return
     99:     
    100:     print("🚀 Démarrage de la mise à jour des configurations agents...")
    101:     
    102:     # Parcourir tous les sous-dossiers de agents/
    103:     for agent_dir in agents_dir.iterdir():
    104:         if agent_dir.is_dir() and not agent_dir.name.startswith('_'):
    105:             # Ignorer certains dossiers si nécessaire
    106:             if agent_dir.name in ['__pycache__', 'sous_agents']:
    107:                 continue
    108:             
    109:             update_agent_config(agent_dir)
    110:     
    111:     print("\n✨ Mise à jour terminée !")
    112:     print("Les fichiers config.yaml ont maintenant leurs sous-agents déclarés.")
    113: 
    114: if __name__ == "__main__":
    115:     main()