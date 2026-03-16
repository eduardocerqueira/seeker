#date: 2026-03-16T17:43:43Z
#url: https://api.github.com/gists/22c1df6a4eb98627d9b2224c280b0bc5
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: # correcteur_requirements.py
      2: import os
      3: 
      4: print("🔧 Correction de requirements.txt")
      5: print("=" * 40)
      6: 
      7: requirements_path = "requirements.txt"
      8: 
      9: # Nouvelles dépendances corrigées
     10: new_requirements = '''# Dépendances du projet SmartContractDevPipeline
     11: # NOTE: python>=3.9 n'est pas une dépendance pip, c'est une exigence système
     12: 
     13: # Core
     14: PyYAML>=6.0
     15: pydantic>=2.5.0
     16: python-dotenv>=1.0.0
     17: 
     18: # Async
     19: aiohttp>=3.9.0
     20: 
     21: # Web3 & Blockchain
     22: web3>=6.0.0
     23: eth-account>=0.11.0
     24: eth-typing>=3.0.0
     25: 
     26: # Development (optionnel)
     27: black>=23.0.0
     28: pytest>=7.0.0
     29: pytest-asyncio>=0.21.0
     30: 
     31: # API (optionnel)
     32: fastapi>=0.104.0
     33: uvicorn>=0.24.0
     34: httpx>=0.25.0
     35: 
     36: # Utils (optionnel)
     37: jinja2>=3.1.0
     38: rich>=13.0.0
     39: '''
     40: 
     41: # Vérifier si le fichier existe
     42: if os.path.exists(requirements_path):
     43:     with open(requirements_path, 'r', encoding='utf-8') as f:
     44:         old_content = f.read()
     45:     
     46:     print("📄 Ancien contenu de requirements.txt:")
     47:     print("-" * 40)
     48:     print(old_content[:200] + "..." if len(old_content) > 200 else old_content)
     49:     print("-" * 40)
     50:     
     51:     # Sauvegarder l'ancienne version
     52:     backup_path = requirements_path + ".backup"
     53:     with open(backup_path, 'w', encoding='utf-8') as f:
     54:         f.write(old_content)
     55:     
     56:     print(f"💾 Backup créé: {backup_path}")
     57: 
     58: # Écrire la nouvelle version
     59: with open(requirements_path, 'w', encoding='utf-8') as f:
     60:     f.write(new_requirements)
     61: 
     62: print("✅ requirements.txt corrigé")
     63: print("\n📋 Nouvelles dépendances:")
     64: print("-" * 40)
     65: print(new_requirements)
     66: print("-" * 40)
     67: 
     68: print("\n🎯 Installation des dépendances principales:")
     69: print("pip install PyYAML aiohttp pydantic python-dotenv web3")
     70: 
     71: print("\n🎯 Installation complète (optionnel):")
     72: print("pip install -r requirements.txt")