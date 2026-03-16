#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: import os
      2: import yaml
      3: from pathlib import Path
      4: 
      5: def unify_yaml_format(file_path):
      6:     """Convertit un fichier YAML au format unifié (name + description)"""
      7:     with open(file_path, 'r', encoding='utf-8') as f:
      8:         data = yaml.safe_load(f)
      9:     
     10:     modified = False
     11:     
     12:     # Traiter les capacités
     13:     if 'capabilities' in data:
     14:         caps = data['capabilities']
     15:         if caps and isinstance(caps, list):
     16:             new_caps = []
     17:             for cap in caps:
     18:                 if isinstance(cap, str):
     19:                     new_caps.append({
     20:                         'name': cap,
     21:                         'description': f'Capacité {cap.lower().replace("_", " ")}'
     22:                     })
     23:                     modified = True
     24:                 else:
     25:                     new_caps.append(cap)
     26:             if modified:
     27:                 data['capabilities'] = new_caps
     28:     
     29:     # Sauvegarder si modifié
     30:     if modified:
     31:         with open(file_path, 'w', encoding='utf-8') as f:
     32:             yaml.dump(data, f, allow_unicode=True, sort_keys=False)
     33:         print(f"✅ Converti: {file_path}")
     34:     
     35:     return modified
     36: 
     37: # Parcourir tous les fichiers config.yaml
     38: for config_file in Path('.').rglob('config.yaml'):
     39:     unify_yaml_format(config_file)