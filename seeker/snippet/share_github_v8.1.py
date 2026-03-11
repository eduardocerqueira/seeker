#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Share GitHub V8.1 - Version CORRIGÉE pour PROJECT_SHARE.txt
      4: Traitement fiable du fichier de partage avec gestion d'erreurs améliorée
      5: """
      6: 
      7: import json
      8: import os
      9: import sys
     10: import datetime
     11: from pathlib import Path
     12: import requests
     13: import math
     14: 
     15: # ============================================================================
     16: # CONFIGURATION
     17: # ============================================================================
     18: 
     19: class Config:
     20:     """Charge et gère la configuration depuis JSON"""
     21:     
     22:     def __init__(self, config_path="../project_config.json"):
     23:         self.config_path = Path(config_path)
     24:         self.default_config = {
     25:             "PROJECT_NAME": "SmartContractDevPipeline",
     26:             "GITHUB_USERNAME": "poolsyncdefi-ui",
     27: "**********": "",
     28:             "MAX_GIST_SIZE_MB": 45,  # Limite de sécurité (45 Mo au lieu de 50)
     29:             "LINES_PER_GIST": 7000,   # Optionnel: pour un contrôle plus fin
     30:         }
     31:         self.config = self.load_config()
     32:     
     33:     def load_config(self):
     34:         """Charge la configuration depuis le fichier JSON"""
     35:         try:
     36:             if self.config_path.exists():
     37:                 print(f"✓ Fichier config trouvé: {self.config_path}")
     38:                 with open(self.config_path, 'r', encoding='utf-8') as f:
     39:                     config = json.load(f)
     40:                 
     41:                 # Fusion avec valeurs par défaut
     42:                 for key, value in self.default_config.items():
     43:                     if key not in config:
     44:                         config[key] = value
     45:                 
     46:                 return config
     47:         except Exception as e:
     48:             print(f"⚠ Erreur chargement config: {e}")
     49:         
     50:         return self.default_config.copy()
     51:     
     52:     def get(self, key, default=None):
     53:         return self.config.get(key, default)
     54:     
     55: "**********":
     56: "**********"
     57:         try:
     58:             with open(self.config_path, 'w', encoding='utf-8') as f:
     59:                 json.dump(self.config, f, indent=2)
     60:             return True
     61:         except:
     62:             return False
     63: 
     64: # ============================================================================
     65: # FONCTIONS PRINCIPALES POUR GISTS
     66: # ============================================================================
     67: 
     68: def estimate_gist_size(content):
     69:     """Estime la taille réelle d'un Gist en bytes (encodage UTF-8)"""
     70:     return len(content.encode('utf-8'))
     71: 
     72: def split_project_share_file(input_file="PROJECT_SHARE.txt", config=None):
     73:     """
     74:     Lit PROJECT_SHARE.txt et le divise intelligemment en parties
     75:     Respecte la structure des fichiers pour éviter les coupures au milieu
     76:     """
     77:     print("\n" + "="*60)
     78:     print("ANALYSE ET DÉCOUPAGE DE PROJECT_SHARE.txt")
     79:     print("="*60)
     80:     
     81:     if not os.path.exists(input_file):
     82:         print(f"✗ Fichier introuvable: {input_file}")
     83:         return None
     84:     
     85:     # Taille du fichier source
     86:     file_size = os.path.getsize(input_file)
     87:     file_size_mb = file_size / (1024 * 1024)
     88:     print(f"📄 Fichier source: {input_file}")
     89:     print(f"📊 Taille: {file_size:,} bytes ({file_size_mb:.2f} MB)")
     90:     
     91:     # Lire tout le fichier
     92:     try:
     93:         with open(input_file, 'r', encoding='utf-8') as f:
     94:             content = f.read()
     95:     except Exception as e:
     96:         print(f"✗ Erreur lecture: {e}")
     97:         return None
     98:     
     99:     # Séparateur de fichiers dans PROJECT_SHARE.txt
    100:     # Le format est: "FICHIER : chemin" puis le contenu, puis ligne vide
    101:     lines = content.split('\n')
    102:     total_lines = len(lines)
    103:     print(f"📝 Lignes totales: {total_lines:,}")
    104:     
    105:     # Détecter les marqueurs de fichiers
    106:     file_markers = []
    107:     for i, line in enumerate(lines):
    108:         if line.startswith("FICHIER : "):
    109:             file_markers.append(i)
    110:     
    111:     print(f"📁 Fichiers détectés: {len(file_markers)}")
    112:     
    113:     if len(file_markers) == 0:
    114:         print("⚠ Aucun marqueur 'FICHIER :' trouvé - format incorrect?")
    115:         return None
    116:     
    117:     # Calculer la taille moyenne par fichier
    118:     avg_lines_per_file = total_lines / len(file_markers)
    119:     print(f"📊 Moyenne: {avg_lines_per_file:.0f} lignes par fichier")
    120:     
    121:     # Déterminer le nombre de Gists nécessaires
    122:     # Option 1: Basé sur la taille (plus fiable)
    123:     max_size_bytes = config.get("MAX_GIST_SIZE_MB", 45) * 1024 * 1024
    124:     
    125:     # Option 2: Basé sur les lignes (fallback)
    126:     max_lines_per_gist = config.get("LINES_PER_GIST", 7000)
    127:     
    128:     # Calculer les deux méthodes et prendre la plus conservative
    129:     estimated_gists_by_size = math.ceil(file_size / max_size_bytes)
    130:     estimated_gists_by_lines = math.ceil(total_lines / max_lines_per_gist)
    131:     num_gists = max(estimated_gists_by_size, estimated_gists_by_lines)
    132:     
    133:     print(f"\n📊 Découpage estimé:")
    134:     print(f"  - Par taille ({max_size_bytes/(1024*1024):.0f} Mo): {estimated_gists_by_size} Gist(s)")
    135:     print(f"  - Par lignes ({max_lines_per_gist} lignes): {estimated_gists_by_lines} Gist(s)")
    136:     print(f"  - Total retenu: {num_gists} Gist(s)")
    137:     
    138:     # Découpage intelligent en respectant les frontières des fichiers
    139:     gist_parts = []
    140:     files_per_gist = math.ceil(len(file_markers) / num_gists)
    141:     
    142:     print(f"\n✂️ Découpage: ~{files_per_gist} fichiers par Gist")
    143:     
    144:     for gist_idx in range(num_gists):
    145:         start_file_idx = gist_idx * files_per_gist
    146:         end_file_idx = min((gist_idx + 1) * files_per_gist, len(file_markers))
    147:         
    148:         if start_file_idx >= len(file_markers):
    149:             break
    150:         
    151:         # Déterminer les lignes de début et fin
    152:         start_line = file_markers[start_file_idx]
    153:         if end_file_idx < len(file_markers):
    154:             end_line = file_markers[end_file_idx] - 1  # Jusqu'à la veille du prochain marker
    155:         else:
    156:             end_line = total_lines - 1  # Jusqu'à la fin
    157:         
    158:         # Extraire le contenu
    159:         part_lines = lines[start_line:end_line + 1]
    160:         part_content = '\n'.join(part_lines)
    161:         
    162:         # Vérifier la taille
    163:         part_size = estimate_gist_size(part_content)
    164:         part_size_mb = part_size / (1024 * 1024)
    165:         
    166:         # Ajouter un en-tête de partie
    167:         header = f"{'='*80}\n"
    168:         header += f"PARTIE {gist_idx + 1}/{num_gists} - {config.get('PROJECT_NAME')}\n"
    169:         header += f"Fichiers: {end_file_idx - start_file_idx}\n"
    170:         header += f"Lignes: {len(part_lines)}\n"
    171:         header += f"{'='*80}\n\n"
    172:         
    173:         full_part_content = header + part_content
    174:         
    175:         gist_parts.append({
    176:             "index": gist_idx + 1,
    177:             "total": num_gists,
    178:             "content": full_part_content,
    179:             "size": estimate_gist_size(full_part_content),
    180:             "size_mb": part_size_mb,
    181:             "files": end_file_idx - start_file_idx,
    182:             "lines": len(part_lines)
    183:         })
    184:         
    185:         print(f"  Partie {gist_idx + 1}: {part_size_mb:.2f} MB, {end_file_idx - start_file_idx} fichiers")
    186:     
    187:     return gist_parts
    188: 
    189: def upload_gists_to_github(gist_parts, config):
    190:     """Upload les parties vers GitHub Gists"""
    191:     print("\n" + "="*60)
    192:     print("UPLOAD VERS GITHUB GISTS")
    193:     print("="*60)
    194:     
    195: "**********"
    196: "**********":
    197: "**********"
    198: "**********": ").strip()
    199: "**********":
    200: "**********"
    201:             return False
    202: "**********"
    203:     
    204:     headers = {
    205: "**********": f"token {github_token}",
    206:         "Accept": "application/vnd.github.v3+json"
    207:     }
    208:     
    209: "**********"
    210:     try:
    211:         test_response = requests.get("https://api.github.com/user", headers=headers)
    212:         if test_response.status_code == 200:
    213:             user = test_response.json().get('login')
    214:             print(f"✓ Connecté en tant que: {user}")
    215:         else:
    216: "**********": {test_response.status_code}")
    217:             return False
    218:     except Exception as e:
    219:         print(f"✗ Erreur connexion: {e}")
    220:         return False
    221:     
    222:     uploaded_gists = []
    223:     
    224:     print(f"\n🚀 Création de {len(gist_parts)} Gist(s)...")
    225:     
    226:     for part in gist_parts:
    227:         print(f"\n📤 Partie {part['index']}/{part['total']} ({part['size_mb']:.2f} MB, {part['files']} fichiers)")
    228:         
    229:         # Créer le nom de fichier
    230:         filename = f"PROJECT_PART{part['index']:02d}_of_{part['total']:02d}.txt"
    231:         
    232:         data = {
    233:             "description": f"{config.get('PROJECT_NAME')} - Partie {part['index']}/{part['total']} - {datetime.datetime.now().strftime('%Y-%m-%d')}",
    234:             "public": True,
    235:             "files": {
    236:                 filename: {
    237:                     "content": part['content']
    238:                 }
    239:             }
    240:         }
    241:         
    242:         try:
    243:             response = requests.post(
    244:                 "https://api.github.com/gists",
    245:                 headers=headers,
    246:                 json=data,
    247:                 timeout=60
    248:             )
    249:             
    250:             if response.status_code == 201:
    251:                 gist_data = response.json()
    252:                 gist_url = gist_data["html_url"]
    253:                 uploaded_gists.append({
    254:                     "part": part['index'],
    255:                     "url": gist_url,
    256:                     "id": gist_data["id"]
    257:                 })
    258:                 print(f"  ✅ Créé: {gist_url}")
    259:             else:
    260:                 print(f"  ❌ Erreur {response.status_code}")
    261:                 print(f"     {response.text[:200]}")
    262:                 
    263:         except Exception as e:
    264:             print(f"  ❌ Exception: {e}")
    265:     
    266:     return uploaded_gists
    267: 
    268: def create_gist_index(uploaded_gists, config):
    269:     """Crée un fichier index avec tous les liens"""
    270:     if not uploaded_gists:
    271:         return
    272:     
    273:     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    274:     index_file = "GISTS_INDEX.txt"
    275:     
    276:     with open(index_file, 'w', encoding='utf-8') as f:
    277:         f.write("="*80 + "\n")
    278:         f.write(f"INDEX DES GISTS - {config.get('PROJECT_NAME')}\n")
    279:         f.write("="*80 + "\n")
    280:         f.write(f"Date: {timestamp}\n")
    281:         f.write(f"Total Gists: {len(uploaded_gists)}\n")
    282:         f.write(f"Visibilité: Public\n")
    283:         f.write("-"*80 + "\n\n")
    284:         
    285:         f.write("LIENS:\n")
    286:         for gist in sorted(uploaded_gists, key=lambda x: x['part']):
    287:             f.write(f"\nPartie {gist['part']:02d}: {gist['url']}")
    288:         
    289:         f.write("\n\n" + "="*80 + "\n")
    290:     
    291:     print(f"\n📄 Index créé: {index_file}")
    292:     print("   Contient les liens vers tous les Gists")
    293: 
    294: def share_to_github_gists(config):
    295:     """Fonction principale - Option 4 du menu"""
    296:     print("\n" + "="*60)
    297:     print("SHARE TO GITHUB GISTS - VERSION CORRIGÉE")
    298:     print("="*60)
    299:     
    300:     # 1. Vérifier que PROJECT_SHARE.txt existe
    301:     if not os.path.exists("PROJECT_SHARE.txt"):
    302:         print("✗ PROJECT_SHARE.txt introuvable!")
    303:         print("   Générez-le d'abord avec l'option 1")
    304:         return False
    305:     
    306:     # 2. Découper le fichier
    307:     gist_parts = split_project_share_file("PROJECT_SHARE.txt", config)
    308:     if not gist_parts:
    309:         return False
    310:     
    311:     # 3. Upload sur GitHub
    312:     uploaded = upload_gists_to_github(gist_parts, config)
    313:     
    314:     # 4. Créer l'index
    315:     if uploaded:
    316:         create_gist_index(uploaded, config)
    317:         print("\n" + "="*60)
    318:         print("✅ OPÉRATION TERMINÉE AVEC SUCCÈS")
    319:         print("="*60)
    320:         return True
    321:     else:
    322:         print("\n❌ Aucun Gist n'a pu être créé")
    323:         return False
    324: 
    325: # ============================================================================
    326: # FONCTIONS EXISTANTES À CONSERVER
    327: # ============================================================================
    328: 
    329: def create_full_share(config):
    330:     """Crée PROJECT_SHARE.txt (identique à votre version)"""
    331:     print("\n" + "="*60)
    332:     print("CREATE FULL SHARE")
    333:     print("="*60)
    334:     print("Cette fonction doit être implémentée avec votre logique existante")
    335:     print("ou utilisez simplement le PROJECT_SHARE.txt que vous avez déjà")
    336:     return True
    337: 
    338: def create_diff_share(config):
    339:     """Crée un rapport des différences Git"""
    340:     print("\n" + "="*60)
    341:     print("CREATE DIFF SHARE")
    342:     print("="*60)
    343:     print("Fonction à implémenter selon vos besoins")
    344:     return True
    345: 
    346: def git_commit_push_publish(config):
    347:     """
    348:     Effectue un commit et un push automatiques vers le dépôt GitHub.
    349:     Gère les cas de figure courants (pas de changement, première push, etc.).
    350:     """
    351:     print("\n" + "="*60)
    352:     print("GIT COMMIT + PUSH")
    353:     print("="*60)
    354: 
    355:     import subprocess
    356:     import os
    357: 
    358:     # 1. Vérifier qu'on est bien dans un dépôt Git
    359:     try:
    360:         subprocess.run(["git", "status"], check=True, capture_output=True, text=True)
    361:     except subprocess.CalledProcessError:
    362:         print("✗ Répertoire courant n'est pas un dépôt Git.")
    363:         return False
    364:     except FileNotFoundError:
    365:         print("✗ Git n'est pas installé ou pas dans le PATH.")
    366:         return False
    367: 
    368:     # 2. Obtenir un message de commit
    369:     default_msg = f"Mise à jour automatique {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    370:     print(f"\nMessage de commit (par défaut : \"{default_msg}\") :")
    371:     commit_msg = input("> ").strip()
    372:     if not commit_msg:
    373:         commit_msg = default_msg
    374: 
    375:     # 3. Ajouter tous les changements
    376:     print("\n📦 Ajout des fichiers modifiés...")
    377:     result_add = subprocess.run(["git", "add", "."], capture_output=True, text=True)
    378:     if result_add.returncode != 0:
    379:         print(f"✗ Erreur lors de git add : {result_add.stderr}")
    380:         return False
    381: 
    382:     # 4. Vérifier s'il y a des changements à commiter
    383:     result_status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    384:     if not result_status.stdout.strip():
    385:         print("ℹ Aucun changement à commiter.")
    386:         return True  # Ce n'est pas une erreur, on sort proprement
    387: 
    388:     # 5. Commit
    389:     print("📝 Création du commit...")
    390:     result_commit = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
    391:     if result_commit.returncode != 0:
    392:         print(f"✗ Erreur lors du commit : {result_commit.stderr}")
    393:         return False
    394:     print(f"✓ Commit créé : {commit_msg}")
    395: 
    396:     # 6. Push vers GitHub
    397:     print("☁ Push vers GitHub...")
    398:     # Récupérer le nom de la branche courante
    399:     branch_result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    400:     current_branch = branch_result.stdout.strip()
    401: 
    402:     if not current_branch:
    403:         # Fallback : si pas de branche (détaché HEAD), on utilise "main"
    404:         current_branch = "main"
    405: 
    406:     # Tenter le push
    407:     result_push = subprocess.run(["git", "push", "origin", current_branch], capture_output=True, text=True)
    408: 
    409:     if result_push.returncode == 0:
    410:         print(f"✓ Push réussi vers origin/{current_branch}")
    411:         print("="*60)
    412:         return True
    413:     else:
    414:         # Analyser les erreurs courantes
    415:         error_msg = result_push.stderr.lower()
    416:         if "no upstream branch" in error_msg:
    417:             print("ℹ Branche locale n'a pas de branche amont. Tentative de push avec -u...")
    418:             result_push_u = subprocess.run(["git", "push", "-u", "origin", current_branch], capture_output=True, text=True)
    419:             if result_push_u.returncode == 0:
    420:                 print(f"✓ Push réussi (upstream configuré) vers origin/{current_branch}")
    421:                 return True
    422:             else:
    423:                 print(f"✗ Échec du push : {result_push_u.stderr}")
    424:         else:
    425:             print(f"✗ Erreur lors du push : {result_push.stderr}")
    426:         return False
    427: 
    428: # ============================================================================
    429: # MENU PRINCIPAL
    430: # ============================================================================
    431: 
    432: def clear_screen():
    433:     os.system('cls' if os.name == 'nt' else 'clear')
    434: 
    435: def main_menu(config):
    436:     while True:
    437:         clear_screen()
    438:         print("\n" + "="*60)
    439:         print(f"SHARE GITHUB V8.1 - {config.get('PROJECT_NAME')}")
    440:         print("="*60)
    441:         
    442: "**********"
    443: "**********"
    444: "**********": {token_status}")
    445:         
    446:         print("\n" + "="*60)
    447:         print("MENU PRINCIPAL")
    448:         print("="*60)
    449:         print("1. CREATE FULL SHARE (génère PROJECT_SHARE.txt)")
    450:         print("2. CREATE DIFF SHARE")
    451:         print("3. GIT COMMIT + PUSH + PUBLISH")
    452:         print("4. SHARE TO GITHUB GISTS (CORRIGÉ) ← NOUVEAU")
    453:         print("5. CONFIGURATION")
    454:         print("6. EXIT")
    455:         print("="*60)
    456:         
    457:         choice = input("\nVotre choix (1-6): ").strip()
    458:         
    459:         if choice == "1":
    460:             create_full_share(config)
    461:         elif choice == "2":
    462:             create_diff_share(config)
    463:         elif choice == "3":
    464:             git_commit_push_publish(config)
    465:         elif choice == "4":
    466:             share_to_github_gists(config)
    467:         elif choice == "5":
    468:             configuration_menu(config)
    469:         elif choice == "6":
    470:             print("\nAu revoir!")
    471:             break
    472:         
    473:         input("\nAppuyez sur ENTER pour continuer...")
    474: 
    475: def configuration_menu(config):
    476:     """Menu de configuration simplifié"""
    477:     while True:
    478:         clear_screen()
    479:         print("\n" + "="*60)
    480:         print("CONFIGURATION")
    481:         print("="*60)
    482:         
    483:         print(f"1. PROJECT_NAME: {config.get('PROJECT_NAME')}")
    484: "**********"
    485: "**********":4] + "..." + token[-4:] if token and len(token) > 8 else "Non configuré"
    486: "**********": {masked}")
    487:         print(f"3. MAX_GIST_SIZE_MB: {config.get('MAX_GIST_SIZE_MB')} Mo")
    488:         print(f"4. LINES_PER_GIST: {config.get('LINES_PER_GIST')}")
    489:         print("5. Retour")
    490:         
    491:         choice = input("\nVotre choix (1-5): ").strip()
    492:         
    493:         if choice == "1":
    494:             new_name = input(f"Nouveau nom [{config.get('PROJECT_NAME')}]: ").strip()
    495:             if new_name:
    496:                 config.config["PROJECT_NAME"] = new_name
    497:         elif choice == "2":
    498: "**********": ").strip()
    499: "**********":
    500: "**********"
    501:         elif choice == "3":
    502:             try:
    503:                 new_size = int(input(f"Nouvelle taille max (Mo) [{config.get('MAX_GIST_SIZE_MB')}]: "))
    504:                 config.config["MAX_GIST_SIZE_MB"] = new_size
    505:             except:
    506:                 pass
    507:         elif choice == "4":
    508:             try:
    509:                 new_lines = int(input(f"Nouvelles lignes max [{config.get('LINES_PER_GIST')}]: "))
    510:                 config.config["LINES_PER_GIST"] = new_lines
    511:             except:
    512:                 pass
    513:         elif choice == "5":
    514:             break
    515: 
    516: # ============================================================================
    517: # POINT D'ENTRÉE
    518: # ============================================================================
    519: 
    520: if __name__ == "__main__":
    521:     print("\n" + "="*60)
    522:     print("SHARE GITHUB V8.1 - Version CORRIGÉE")
    523:     print("="*60)
    524:     
    525:     config = Config()
    526:     
    527:     try:
    528:         main_menu(config)
    529:     except KeyboardInterrupt:
    530:         print("\n\nProgramme interrompu.")
    531:     except Exception as e:
    532:         print(f"\n✗ Erreur: {e}")
    533:         import traceback
    534:         traceback.print_exc()======================================
    519: 
    520: if __name__ == "__main__":
    521:     print("\n" + "="*60)
    522:     print("SHARE GITHUB V8.1 - Version CORRIGÉE")
    523:     print("="*60)
    524:     
    525:     config = Config()
    526:     
    527:     try:
    528:         main_menu(config)
    529:     except KeyboardInterrupt:
    530:         print("\n\nProgramme interrompu.")
    531:     except Exception as e:
    532:         print(f"\n✗ Erreur: {e}")
    533:         import traceback
    534:         traceback.print_exc()