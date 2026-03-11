#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Share GitHub V9.12 - Version corrigée
      4: "**********"
      5: - Affichage synthétique optimisé
      6: """
      7: 
      8: import json
      9: import os
     10: import sys
     11: import datetime
     12: import fnmatch
     13: import subprocess
     14: from pathlib import Path
     15: import requests
     16: 
     17: # ============================================================================
     18: # CONFIGURATION
     19: # ============================================================================
     20: 
     21: class Config:
     22:     def __init__(self):
     23:         current_dir = Path.cwd()
     24:         
     25:         if current_dir.name == "SmartContractDevPipeline":
     26:             self.project_root = current_dir
     27:             self.parent_dir = current_dir.parent
     28:         else:
     29:             self.project_root = current_dir / "SmartContractDevPipeline"
     30:             self.parent_dir = current_dir
     31:         
     32:         # Le fichier de config est dans le parent (D:\Web3Projects\)
     33:         self.config_path = self.parent_dir / "project_config.json"
     34:         print(f"📁 Chargement config: {self.config_path}")
     35:         self.config = self.load_config()
     36:     
     37:     def load_config(self):
     38:         """Charge la configuration depuis project_config.json"""
     39:         if self.config_path.exists():
     40:             try:
     41:                 with open(self.config_path, 'r', encoding='utf-8') as f:
     42:                     config = json.load(f)
     43:                     print(f"✓ Config chargée: {config.get('PROJECT_NAME')}")
     44:                     
     45: "**********": voir si le token est présent (masqué)
     46: "**********"
     47: "**********":
     48: "**********": {token[:4]}...{token[-4:]}")
     49:                     else:
     50: "**********"
     51:                     
     52:                     return config
     53:             except Exception as e:
     54:                 print(f"✗ Erreur chargement: {e}")
     55:         
     56:         # Configuration par défaut
     57:         print("⚠ Utilisation config par défaut")
     58:         return {
     59:             "PROJECT_NAME": "SmartContractDevPipeline",
     60: "**********": "",
     61:             "PROJECT_PATH": str(self.project_root),
     62:             "MAX_GIST_SIZE_MB": 45,
     63:             "MAX_FILES_PER_GIST": 100,
     64:             "MAX_FILE_SIZE_MB": 8,
     65:             "EXCLUDE_PATTERNS": [],
     66:             "EXCLUDE_DIRS": [],
     67:             "INCLUDE_PATTERNS": ["*"]
     68:         }
     69:     
     70:     def get(self, key, default=None):
     71:         return self.config.get(key, default)
     72:     
     73: "**********":
     74: "**********"
     75:         try:
     76:             with open(self.config_path, 'w', encoding='utf-8') as f:
     77:                 json.dump(self.config, f, indent=2)
     78: "**********"
     79:             return True
     80:         except Exception as e:
     81:             print(f"✗ Erreur sauvegarde: {e}")
     82:             return False
     83: 
     84: 
     85: # ============================================================================
     86: # FILTRAGE
     87: # ============================================================================
     88: 
     89: def should_exclude(path, config):
     90:     path_str = str(path)
     91:     path_parts = path.parts
     92:     
     93:     exclude_dirs = config.get("EXCLUDE_DIRS", [])
     94:     for part in path_parts:
     95:         if part in exclude_dirs:
     96:             return True
     97:     
     98:     exclude_patterns = config.get("EXCLUDE_PATTERNS", [])
     99:     for pattern in exclude_patterns:
    100:         if pattern.endswith('/*'):
    101:             dir_pattern = pattern[:-2]
    102:             if f"{dir_pattern}{os.sep}" in path_str or f"{dir_pattern}/" in path_str:
    103:                 return True
    104:         elif fnmatch.fnmatch(path.name, pattern):
    105:             return True
    106:         elif fnmatch.fnmatch(path_str, pattern):
    107:             return True
    108:     
    109:     return False
    110: 
    111: def should_include(path, config):
    112:     if path.is_dir():
    113:         return False
    114:     
    115:     include_patterns = config.get("INCLUDE_PATTERNS", ["*"])
    116:     for pattern in include_patterns:
    117:         if fnmatch.fnmatch(path.name, pattern):
    118:             return True
    119:     
    120:     return False
    121: 
    122: def collect_files(project_path, config):
    123:     all_files = []
    124:     project_root = Path(project_path)
    125:     
    126:     if not project_root.exists():
    127:         return []
    128:     
    129:     for root, dirs, files in os.walk(project_root):
    130:         dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d, config)]
    131:         
    132:         for file in files:
    133:             file_path = Path(root) / file
    134:             
    135:             if should_exclude(file_path, config):
    136:                 continue
    137:             
    138:             if should_include(file_path, config):
    139:                 all_files.append(file_path)
    140:     
    141:     return sorted(all_files)
    142: 
    143: 
    144: # ============================================================================
    145: # LECTURE DE FICHIERS
    146: # ============================================================================
    147: 
    148: def read_file_with_line_numbers(file_path):
    149:     encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    150:     
    151:     for encoding in encodings:
    152:         try:
    153:             with open(file_path, 'r', encoding=encoding) as f:
    154:                 lines = f.readlines()
    155:             
    156:             numbered_lines = []
    157:             for i, line in enumerate(lines, 1):
    158:                 line = line.rstrip('\n\r')
    159:                 numbered_lines.append(f"   {i:4d}: {line}")
    160:             
    161:             return numbered_lines
    162:         except:
    163:             continue
    164:     
    165:     return [f"   [FICHIER BINAIRE]"]
    166: 
    167: 
    168: # ============================================================================
    169: # GÉNÉRATION DE PROJECT_SHARE.txt
    170: # ============================================================================
    171: 
    172: def create_full_share(config):
    173:     print("\n" + "="*50)
    174:     print("📄 GÉNÉRATION PROJECT_SHARE.txt")
    175:     
    176:     project_path = config.get("PROJECT_PATH")
    177:     if not project_path:
    178:         print("✗ PROJECT_PATH non défini")
    179:         return False
    180:     
    181:     project_root = Path(project_path)
    182:     if not project_root.exists():
    183:         print(f"✗ Chemin introuvable")
    184:         return False
    185:     
    186:     files = collect_files(project_path, config)
    187:     
    188:     if not files:
    189:         print("✗ Aucun fichier")
    190:         return False
    191:     
    192:     output_file = project_root / "PROJECT_SHARE.txt"
    193:     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    194:     
    195:     with open(output_file, 'w', encoding='utf-8') as out_f:
    196:         out_f.write(f"PROJECT: {config.get('PROJECT_NAME')}\n")
    197:         out_f.write(f"DATE: {timestamp}\n")
    198:         out_f.write(f"FILES: {len(files)}\n")
    199:         out_f.write("=" * 80 + "\n\n")
    200:         
    201:         for file_path in files:
    202:             out_f.write(f"FICHIER : {file_path}\n")
    203:             
    204:             numbered_lines = read_file_with_line_numbers(file_path)
    205:             for line in numbered_lines:
    206:                 out_f.write(line + '\n')
    207:             
    208:             out_f.write('\n')
    209:     
    210:     if output_file.exists():
    211:         file_size = output_file.stat().st_size
    212:         file_size_mb = file_size / (1024 * 1024)
    213:         print(f"✅ {len(files)} fichiers - {file_size_mb:.2f} MB")
    214:         return True
    215:     else:
    216:         print("❌ Échec")
    217:         return False
    218: 
    219: 
    220: # ============================================================================
    221: # PRÉPARATION DES FICHIERS
    222: # ============================================================================
    223: 
    224: def split_large_file(content, filename, max_size_bytes):
    225:     """Fractionne un fichier en parties de taille max_size_bytes"""
    226:     lines = content.split('\n')
    227:     total_size = len(content.encode('utf-8'))
    228:     
    229:     if total_size <= max_size_bytes:
    230:         return [(filename, content)]
    231:     
    232:     nb_parts = (total_size + max_size_bytes - 1) // max_size_bytes
    233:     size_per_part = total_size / nb_parts
    234:     
    235:     parts = []
    236:     current_part = []
    237:     current_size = 0
    238:     part_num = 1
    239:     
    240:     for line in lines:
    241:         line_size = len(line.encode('utf-8')) + 1
    242:         if current_size + line_size > size_per_part * 1.2 and current_part:
    243:             part_content = '\n'.join(current_part)
    244:             base, ext = os.path.splitext(filename)
    245:             part_filename = f"{base}_part{part_num:02d}{ext}"
    246:             parts.append((part_filename, part_content))
    247:             
    248:             current_part = [line]
    249:             current_size = line_size
    250:             part_num += 1
    251:         else:
    252:             current_part.append(line)
    253:             current_size += line_size
    254:     
    255:     if current_part:
    256:         part_content = '\n'.join(current_part)
    257:         base, ext = os.path.splitext(filename)
    258:         part_filename = f"{base}_part{part_num:02d}{ext}"
    259:         parts.append((part_filename, part_content))
    260:     
    261:     return parts
    262: 
    263: def estimate_content_size(content):
    264:     return len(content.encode('utf-8'))
    265: 
    266: def prepare_files_for_gists(project_root, config):
    267:     """Prépare les fichiers pour les Gists"""
    268:     all_files = collect_files(project_root, config)
    269:     all_files = [f for f in all_files if f.name not in ["PROJECT_SHARE.txt", "GISTS_INDEX.txt"]]
    270:     
    271:     max_file_size = config.get("MAX_FILE_SIZE_MB", 8) * 1024 * 1024
    272:     
    273:     prepared = []
    274:     total_split = 0
    275:     
    276:     for file_path in all_files:
    277:         numbered_lines = read_file_with_line_numbers(file_path)
    278:         content = '\n'.join(numbered_lines)
    279:         file_size = estimate_content_size(content)
    280:         
    281:         if file_size > max_file_size:
    282:             parts = split_large_file(content, file_path.name, max_file_size)
    283:             total_split += 1
    284:             for part_filename, part_content in parts:
    285:                 prepared.append({
    286:                     "path": str(file_path),
    287:                     "name": part_filename,
    288:                     "content": part_content,
    289:                     "size": estimate_content_size(part_content),
    290:                     "original": file_path.name
    291:                 })
    292:         else:
    293:             prepared.append({
    294:                 "path": str(file_path),
    295:                 "name": file_path.name,
    296:                 "content": content,
    297:                 "size": file_size,
    298:                 "original": None
    299:             })
    300:     
    301:     # Trier par taille décroissante
    302:     prepared.sort(key=lambda x: x["size"], reverse=True)
    303:     
    304:     return prepared, total_split
    305: 
    306: def pack_gists(files, config):
    307:     """Packaging optimisé des fichiers dans les Gists"""
    308:     max_size = config.get("MAX_GIST_SIZE_MB", 45) * 1024 * 1024
    309:     max_files = config.get("MAX_FILES_PER_GIST", 100)
    310:     
    311:     gists = []
    312:     
    313:     # Séparer les fichiers par taille
    314:     large = [f for f in files if f["size"] > 1024 * 1024]  # > 1 MB
    315:     medium = [f for f in files if 100 * 1024 < f["size"] <= 1024 * 1024]  # 100 KB - 1 MB
    316:     small = [f for f in files if f["size"] <= 100 * 1024]  # < 100 KB
    317:     
    318:     # Placer les gros fichiers d'abord
    319:     for file in large + medium:
    320:         placed = False
    321:         # Chercher le Gist le plus rempli qui peut accueillir ce fichier
    322:         best_gist = None
    323:         best_fill = -1
    324:         
    325:         for gist in gists:
    326:             if (gist["size"] + file["size"] <= max_size and 
    327:                 len(gist["files"]) < max_files):
    328:                 fill = gist["size"] / max_size
    329:                 if fill > best_fill:
    330:                     best_fill = fill
    331:                     best_gist = gist
    332:         
    333:         if best_gist:
    334:             best_gist["files"].append(file)
    335:             best_gist["size"] += file["size"]
    336:         else:
    337:             gists.append({
    338:                 "files": [file],
    339:                 "size": file["size"]
    340:             })
    341:     
    342:     # Remplir avec les petits fichiers
    343:     for file in small:
    344:         # Chercher le Gist avec le plus d'espace libre mais pas trop vide
    345:         candidates = []
    346:         for gist in gists:
    347:             if (gist["size"] + file["size"] <= max_size and 
    348:                 len(gist["files"]) < max_files):
    349:                 free_space = max_size - gist["size"]
    350:                 candidates.append((free_space, gist))
    351:         
    352:         if candidates:
    353:             # Prendre le Gist avec le moins d'espace libre (pour maximiser remplissage)
    354:             candidates.sort()
    355:             gist = candidates[0][1]
    356:             gist["files"].append(file)
    357:             gist["size"] += file["size"]
    358:         else:
    359:             gists.append({
    360:                 "files": [file],
    361:                 "size": file["size"]
    362:             })
    363:     
    364:     return gists
    365: 
    366: 
    367: # ============================================================================
    368: # UPLOAD GISTS
    369: # ============================================================================
    370: 
    371: def create_gist_files(gist_data, part_idx, total_parts, project_name):
    372:     """Crée le dictionnaire des fichiers pour un Gist"""
    373:     files_dict = {}
    374:     
    375:     # README synthétique
    376:     readme = f"""# {project_name} - Partie {part_idx}/{total_parts}
    377: 📊 {len(gist_data['files'])} fichiers - {gist_data['size']/(1024*1024):.2f} MB
    378: 
    379: 📁 Fichiers:
    380: """
    381:     for f in gist_data["files"]:
    382:         if f["original"]:
    383:             readme += f"  • {f['name']} (partie de {f['original']})\n"
    384:         else:
    385:             readme += f"  • {f['name']}\n"
    386:     
    387:     files_dict["README.txt"] = {"content": readme}
    388:     
    389:     # Ajouter les fichiers
    390:     for f in gist_data["files"]:
    391:         files_dict[f["name"]] = {"content": f["content"]}
    392:     
    393:     return files_dict
    394: 
    395: def upload_to_github(gists, config, project_root):
    396:     """Upload les Gists vers GitHub"""
    397:     print("\n" + "="*50)
    398:     print("📤 UPLOAD GISTS")
    399:     
    400: "**********"
    401: "**********"
    402:     
    403: "**********":
    404: "**********"
    405: "**********": ").strip()
    406: "**********":
    407: "**********"
    408:         else:
    409:             return None
    410:     
    411: "**********"
    412: "**********": {github_token[:4]}...{github_token[-4:]}")
    413:     
    414: "**********": f"token {github_token}"}
    415:     
    416: "**********"
    417:     try:
    418: "**********"
    419:         test = requests.get("https://api.github.com/user", headers=headers)
    420:         
    421:         if test.status_code == 200:
    422:             user = test.json()
    423:             print(f"✓ Connecté: {user.get('login')}")
    424:         else:
    425: "**********": {test.status_code}")
    426:             print(f"  {test.text[:200]}")
    427:             
    428: "**********"
    429: "**********": ").strip()
    430: "**********":
    431: "**********": f"token {github_token}"}
    432: "**********"
    433:                 # Revérifier
    434:                 test = requests.get("https://api.github.com/user", headers=headers)
    435:                 if test.status_code != 200:
    436: "**********"
    437:                     return None
    438:             else:
    439:                 return None
    440:     except Exception as e:
    441:         print(f"✗ Erreur connexion: {e}")
    442:         return None
    443:     
    444:     uploaded = []
    445:     
    446:     for idx, gist in enumerate(gists, 1):
    447:         size_mb = gist["size"] / (1024 * 1024)
    448:         fill_pct = (gist["size"] / (config.get("MAX_GIST_SIZE_MB", 45) * 1024 * 1024)) * 100
    449:         
    450:         print(f"\n  [{idx}/{len(gists)}] ", end="")
    451:         print(f"{len(gist['files']):3d} fichiers - {size_mb:5.2f} MB ({fill_pct:3.0f}%)")
    452:         
    453:         files_dict = create_gist_files(gist, idx, len(gists), config.get('PROJECT_NAME'))
    454:         
    455:         data = {
    456:             "description": f"{config.get('PROJECT_NAME')} - Partie {idx}/{len(gists)}",
    457:             "public": True,
    458:             "files": files_dict
    459:         }
    460:         
    461:         try:
    462:             print("    ⏳ Upload...", end="", flush=True)
    463:             resp = requests.post("https://api.github.com/gists", headers=headers, json=data, timeout=300)
    464:             
    465:             if resp.status_code == 201:
    466:                 url = resp.json()["html_url"]
    467:                 print(f"\r    ✅ {url}")
    468:                 uploaded.append({
    469:                     "part": idx,
    470:                     "url": url,
    471:                     "size": size_mb,
    472:                     "files": len(gist["files"])
    473:                 })
    474:             else:
    475:                 print(f"\r    ❌ Erreur {resp.status_code}")
    476:                 print(f"    {resp.text[:200]}")
    477:                 if resp.status_code == 401:
    478: "**********"
    479:                     break
    480:         except Exception as e:
    481:             print(f"\r    ❌ Exception: {e}")
    482:     
    483:     return uploaded
    484: 
    485: def save_index(uploaded, config, project_root):
    486:     if not uploaded:
    487:         return
    488:     
    489:     index_file = project_root / "GISTS_INDEX.txt"
    490:     total_size = sum(g["size"] for g in uploaded)
    491:     total_files = sum(g["files"] for g in uploaded)
    492:     
    493:     with open(index_file, 'w', encoding='utf-8') as f:
    494:         f.write(f"{config.get('PROJECT_NAME')}\n")
    495:         f.write(f"Date: {datetime.datetime.now()}\n")
    496:         f.write(f"Gists: {len(uploaded)} | Fichiers: {total_files} | Taille: {total_size:.2f} MB\n\n")
    497:         
    498:         for g in sorted(uploaded, key=lambda x: x['part']):
    499:             f.write(f"Partie {g['part']:02d}: {g['url']} ({g['files']} fichiers, {g['size']:.2f} MB)\n")
    500:     
    501:     print(f"\n📄 Index: {index_file.name}")
    502: 
    503: def share_to_gists(config, project_root):
    504:     print("\n" + "="*50)
    505:     print("📊 SHARE TO GISTS")
    506:     
    507:     # Préparation
    508:     files, nb_split = prepare_files_for_gists(project_root, config)
    509:     
    510:     # Packaging optimisé
    511:     gists = pack_gists(files, config)
    512:     
    513:     # Stats
    514:     total_files = len(files)
    515:     total_size = sum(f["size"] for f in files) / (1024 * 1024)
    516:     
    517:     print(f"\n📊 {total_files} fichiers ({total_size:.2f} MB) → {len(gists)} Gists")
    518:     if nb_split:
    519:         print(f"   ({nb_split} fichiers fractionnés)")
    520:     
    521:     # Aperçu compact
    522:     max_size_mb = config.get("MAX_GIST_SIZE_MB", 45)
    523:     for i, g in enumerate(gists, 1):
    524:         fill = (g["size"] / (max_size_mb * 1024 * 1024)) * 100
    525:         print(f"   G{i:02d}: {len(g['files']):3d} fichiers, {g['size']/(1024*1024):5.2f} MB ({fill:3.0f}%)")
    526:     
    527:     # Upload
    528:     uploaded = upload_to_github(gists, config, project_root)
    529:     
    530:     if uploaded:
    531:         save_index(uploaded, config, project_root)
    532:         print("\n✅ Terminé")
    533:         return True
    534:     
    535:     return False
    536: 
    537: 
    538: # ============================================================================
    539: # GIT COMMIT + PUSH
    540: # ============================================================================
    541: 
    542: def git_commit_push(config, project_root):
    543:     print("\n" + "="*50)
    544:     print("📦 GIT COMMIT + PUSH")
    545:     
    546:     try:
    547:         status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=project_root)
    548:         
    549:         if not status.stdout.strip():
    550:             print("ℹ Aucun changement")
    551:             return True
    552:         
    553:         files = len(status.stdout.strip().split('\n'))
    554:         print(f"📝 {files} fichier(s)")
    555:         
    556:         msg = input("Message (défaut): ").strip()
    557:         if not msg:
    558:             msg = f"Màj {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    559:         
    560:         subprocess.run(["git", "add", "."], cwd=project_root, check=True)
    561:         subprocess.run(["git", "commit", "-m", msg], cwd=project_root, check=True)
    562:         
    563:         branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=project_root)
    564:         branch = branch.stdout.strip() or "main"
    565:         
    566:         result = subprocess.run(["git", "push", "origin", branch], capture_output=True, text=True, cwd=project_root)
    567:         
    568:         if result.returncode == 0:
    569:             print("✓ Push OK")
    570:             return True
    571:         else:
    572:             print("✗ Push échec")
    573:             return False
    574:     except Exception as e:
    575:         print(f"✗ Erreur: {e}")
    576:         return False
    577: 
    578: 
    579: # ============================================================================
    580: # MENU
    581: # ============================================================================
    582: 
    583: def clear():
    584:     os.system('cls' if os.name == 'nt' else 'clear')
    585: 
    586: def main():
    587:     config = Config()
    588:     
    589:     while True:
    590:         clear()
    591:         print("\n" + "="*50)
    592:         print(f"🚀 SHARE GITHUB V9.12 - {config.get('PROJECT_NAME')}")
    593:         print("="*50)
    594:         print("1. 📄 Générer PROJECT_SHARE.txt")
    595:         print("2. 📦 Git commit + push")
    596:         print("3. 📤 Partager vers Gists")
    597:         print("4. ❌ Quitter")
    598:         print("-"*50)
    599:         
    600:         choice = input("Choix: ").strip()
    601:         
    602:         if choice == "1":
    603:             create_full_share(config)
    604:         elif choice == "2":
    605:             git_commit_push(config, config.project_root)
    606:         elif choice == "3":
    607:             share_to_gists(config, config.project_root)
    608:         elif choice == "4":
    609:             print("\n👋 Au revoir!")
    610:             break
    611:         
    612:         input("\nAppuyez sur ENTER...")
    613: 
    614: if __name__ == "__main__":
    615:     try:
    616:         main()
    617:     except KeyboardInterrupt:
    618:         print("\n\n👋 Interrompu")
    619:     except Exception as e:
    620:         print(f"\n✗ Erreur: {e}")print("2. 📦 Git commit + push")
    596:         print("3. 📤 Partager vers Gists")
    597:         print("4. ❌ Quitter")
    598:         print("-"*50)
    599:         
    600:         choice = input("Choix: ").strip()
    601:         
    602:         if choice == "1":
    603:             create_full_share(config)
    604:         elif choice == "2":
    605:             git_commit_push(config, config.project_root)
    606:         elif choice == "3":
    607:             share_to_gists(config, config.project_root)
    608:         elif choice == "4":
    609:             print("\n👋 Au revoir!")
    610:             break
    611:         
    612:         input("\nAppuyez sur ENTER...")
    613: 
    614: if __name__ == "__main__":
    615:     try:
    616:         main()
    617:     except KeyboardInterrupt:
    618:         print("\n\n👋 Interrompu")
    619:     except Exception as e:
    620:         print(f"\n✗ Erreur: {e}")