#date: 2026-03-11T17:32:37Z
#url: https://api.github.com/gists/ff4f91fd2487b9d23843e8c09adeb659
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import os
      2: import yaml
      3: from pathlib import Path
      4: 
      5: def convert_orchestrator_config(file_path):
      6:     """Convertit le fichier orchestrator/config.yaml au format standard agent"""
      7:     
      8:     print(f"🔄 Conversion de {file_path}...")
      9:     
     10:     # Lire le fichier existant
     11:     with open(file_path, 'r', encoding='utf-8') as f:
     12:         data = yaml.safe_load(f)
     13:     
     14:     # Créer la nouvelle structure standard
     15:     standard_format = {
     16:         'agent': {
     17:             'name': 'Orchestrateur',
     18:             'display_name': 'Orchestrateur de Workflows',
     19:             'version': data.get('orchestrator', {}).get('version', '1.0.0'),
     20:             'description': 'Orchestration globale des workflows et sprints du pipeline',
     21:             'agent_type': 'concrete',
     22:             'enabled': True,
     23:             'instantiate': True,
     24:             'dependencies': ['base_agent'],
     25:             'initialization_order': 15,
     26:             'parent': 'base_agent',
     27:             'purpose': 'Orchestration des workflows complexes entre agents',
     28:             'specialization': 'workflow_orchestration',
     29:             'mandatory': True,
     30:             
     31:             # Capacités standards de l'orchestrateur
     32:             'capabilities': [
     33:                 {
     34:                     'name': 'WORKFLOW_ORCHESTRATION',
     35:                     'description': 'Orchestration de workflows complexes'
     36:                 },
     37:                 {
     38:                     'name': 'SPRINT_MANAGEMENT',
     39:                     'description': 'Gestion de sprints et planification'
     40:                 },
     41:                 {
     42:                     'name': 'TASK_SCHEDULING',
     43:                     'description': 'Planification et ordonnancement des tâches'
     44:                 },
     45:                 {
     46:                     'name': 'RESOURCE_ALLOCATION',
     47:                     'description': 'Allocation et gestion des ressources'
     48:                 },
     49:                 {
     50:                     'name': 'DEADLINE_MANAGEMENT',
     51:                     'description': 'Gestion des délais et échéances'
     52:                 },
     53:                 {
     54:                     'name': 'PARALLEL_EXECUTION',
     55:                     'description': 'Exécution parallèle de workflows'
     56:                 },
     57:                 {
     58:                     'name': 'DEPENDENCY_RESOLUTION',
     59:                     'description': 'Résolution des dépendances entre tâches'
     60:                 },
     61:                 {
     62:                     'name': 'STATE_TRACKING',
     63:                     'description': 'Suivi d\'état des workflows'
     64:                 }
     65:             ]
     66:         },
     67:         
     68:         # Conserver les workflows comme sous-agents
     69:         'subAgents': []
     70:     }
     71:     
     72:     # Convertir les workflows en sous-agents
     73:     if 'workflow' in data:
     74:         for wf_name, wf_config in data['workflow'].items():
     75:             # Créer un sous-agent pour chaque workflow
     76:             sub_agent = {
     77:                 'id': f"workflow_{wf_name}",
     78:                 'name': wf_config.get('name', wf_name),
     79:                 'display_name': wf_config.get('name', wf_name),
     80:                 'description': wf_config.get('description', f'Workflow {wf_name}'),
     81:                 'version': '1.0.0',
     82:                 'enabled': True,
     83:                 'config_path': None,  # Pas de fichier de config séparé
     84:                 'dependencies': [],
     85:                 
     86:                 # Capacités basées sur les étapes du workflow
     87:                 'capabilities': [
     88:                     {
     89:                         'name': f"STEP_{step.get('id', f'step_{i}')}".upper(),
     90:                         'description': f"Étape: {step.get('task', 'unknown')}"
     91:                     }
     92:                     for i, step in enumerate(wf_config.get('steps', []))
     93:                 ]
     94:             }
     95:             standard_format['subAgents'].append(sub_agent)
     96:     
     97:     # Conserver aussi la configuration des autres agents si présente
     98:     if 'agents' in data:
     99:         standard_format['managed_agents'] = data['agents']
    100:     
    101:     # Sauvegarder le fichier converti
    102:     backup_path = file_path.with_suffix('.yaml.backup')
    103:     os.rename(file_path, backup_path)
    104:     print(f"  ✅ Backup créé: {backup_path}")
    105:     
    106:     with open(file_path, 'w', encoding='utf-8') as f:
    107:         yaml.dump(standard_format, f, allow_unicode=True, sort_keys=False, indent=2)
    108:     
    109:     print(f"  ✅ Fichier converti avec succès: {file_path}")
    110:     print(f"  📊 {len(standard_format['subAgents'])} workflows convertis en sous-agents")
    111:     
    112:     return True
    113: 
    114: def main():
    115:     # Chemin vers le fichier orchestrator/config.yaml
    116:     orchestrator_config = Path(__file__).parent / 'orchestrator' / 'config.yaml'
    117:     
    118:     if not orchestrator_config.exists():
    119:         print(f"❌ Fichier non trouvé: {orchestrator_config}")
    120:         return
    121:     
    122:     convert_orchestrator_config(orchestrator_config)
    123:     
    124:     print("\n✨ Conversion terminée !")
    125:     print("Le fichier orchestrator/config.yaml est maintenant au format standard.")
    126:     print("Les workflows sont maintenant des sous-agents de l'orchestrateur.")
    127: 
    128: if __name__ == "__main__":
    129:     main()