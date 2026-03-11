#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: #!/usr/bin/env python3
      2: """
      3: Share GitHub V9.8 - Version ULTRA SIMPLIFIÉE
      4: Génération de PROJECT_SHARE.txt et upload vers GitHub Gists
      5: Affichage minimal
      6: """
      7: 
      8: import json
      9: import os
     10: import sys
     11: import datetime
     12: import math
     13: import fnmatch
     14: import subprocess
     15: from pathlib import Path
     16: import requests
     17: 
     18: # ============================================================================
     19: # CONFIGURATION
     20: # ============================================================================
     21: 
     22: class Config:
     23:     def __init__(self):
     24:         current_dir = Path.cwd()
     25:         
     26:         # Déterminer les chemins
     27:         if current_dir.name == "SmartContractDevPipeline":
     28:             self.project_root = current_dir
     29:             self.parent_dir = current_dir.parent
     30:         else:
     31:             self.project_root = current_dir / "SmartContractDevPipeline"
     32:             self.parent_dir = current_dir
     33:         
     34:         # Chercher la config dans le dossier parent
     35:         self.config_path = self.parent_dir / "project_config.json"
     36:         self.config = self.load_config()
     37:     
     38:     def load_config(self):
     39:         """Charge la configuration depuis project_config.json"""
     40:         if self.config_path.exists():
     41:             try:
     42:                 with open(self.config_path, 'r', encoding='utf-8') as f:
     43:                     config = json.load(f)
     44:                 return config
     45:             except Exception:
     46:                 pass
     47:         
     48:         # Configuration par défaut minimale
     49:         return {
     50:             "PROJECT_NAME": "SmartContractDevPipeline",
     51: "**********": "",
     52:             "PROJECT_PATH": str(self.project_root),
     53:             "MAX_GIST_SIZE_MB": 45,
     54:             "EXCLUDE_PATTERNS": [
     55:                 "node_modules/*", ".git/*", "__pycache__/*", ".venv/*", "venv/*", "env/*",
     56:                 "dist/*", "build/*", ".next/*", ".nuxt/*", "target/*", ".gradle/*",
     57:                 ".vscode/*", ".idea/*", ".vs/*", ".pytest_cache/*", ".mypy_cache/*",
     58:                 ".coverage/*", ".tox/*", "logs/*", ".cache/*", ".tmp/*", "temp/*",
     59:                 ".yarn/*", ".npm/*", "bower_components/*", "jspm_packages/*",
     60:                 ".serverless/*", ".webpack/*", ".angular/*", ".vuepress/*", ".docusaurus/*",
     61:                 ".sapper/*", ".gatsby/*", ".gridsome/*", ".quasar/*", ".ionic/*", ".expo/*",
     62:                 ".react-native/*", ".meteor/*", ".electron/*", ".github/*", ".gitlab/*",
     63:                 ".circleci/*", ".travis/*", ".appveyor/*", ".DS_Store", "Thumbs.db",
     64:                 "desktop.ini", "Icon", "$RECYCLE.BIN/*", "System Volume Information/*",
     65:                 ".Spotlight-V100/*", ".Trashes/*", ".TemporaryItems/*", ".fseventsd/*",
     66:                 ".DocumentRevisions-V100/*", "package-lock.json", "yarn.lock",
     67:                 "pnpm-lock.yaml", "shrinkwrap.json", "*.log", "*.tmp", "*.cache",
     68:                 "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dylib", "*.dll", "*.exe",
     69:                 "*.class", "*.jar", "*.o", "*.obj", ".env*", "*.key", "*.pem",
     70:                 "*.crt", "*.cert", "*.p12", "*.pfx", "*.keystore",
     71:                 "PROJECT_SHARE.txt"
     72:             ],
     73:             "EXCLUDE_DIRS": [
     74:                 "node_modules", ".git", ".vscode", ".idea", ".vs", "__pycache__",
     75:                 ".pytest_cache", ".mypy_cache", ".coverage", ".tox", "venv",
     76:                 ".venv", "env", "virtualenv", "env.bak", "dist", "build", "out",
     77:                 ".next", ".nuxt", "target", ".gradle", ".settings", "coverage",
     78:                 ".nyc_output", "logs", ".cache", ".tmp", "temp", "tmp", ".yarn",
     79:                 ".npm", "bower_components", "jspm_packages", ".serverless",
     80:                 ".webpack", ".angular", ".vuepress", ".docusaurus", ".sapper",
     81:                 ".gatsby", ".gridsome", ".quasar", ".ionic", ".expo",
     82:                 ".react-native", ".meteor", ".electron", ".github", ".gitlab",
     83:                 ".circleci", ".travis", ".appveyor", "$RECYCLE.BIN",
     84:                 "System Volume Information", ".Spotlight-V100", ".Trashes",
     85:                 ".TemporaryItems", ".fseventsd", ".DocumentRevisions-V100",
     86:                 ".gists", "gists", "github_gists", "test_gists", "gist_test",
     87:                 ".gist_cache", ".github_cache"
     88:             ],
     89:             "INCLUDE_PATTERNS": [
     90:                 "*.py", "*.js", "*.ts", "*.sol", "*.rs", "*.go", "*.java",
     91:                 "*.cpp", "*.h", "*.hpp", "*.c", "*.cs", "*.rb", "*.php",
     92:                 "*.swift", "*.kt", "*.kts", "*.scala", "*.md", "*.txt",
     93:                 "*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg",
     94:                 "*.conf", "*.html", "*.css", "*.scss", "*.sass", "*.less",
     95:                 "*.mermaid", "*.svg", "*.sh", "*.bat", "*.ps1", "*.dockerfile",
     96:                 "Dockerfile*", "docker-compose*", "Makefile", "Gemfile",
     97:                 "Cargo.toml", "go.mod", "package.json", "requirements.txt",
     98:                 "pyproject.toml", "setup.py", "*.sql", "*.graphql", "*.gql"
     99:             ]
    100:         }
    101:     
    102:     def get(self, key, default=None):
    103:         return self.config.get(key, default)
    104:     
    105: "**********":
    106: "**********"
    107:         try:
    108:             with open(self.config_path, 'w', encoding='utf-8') as f:
    109:                 json.dump(self.config, f, indent=2)
    110:             return True
    111:         except:
    112:             return False
    113: 
    114: 
    115: # ============================================================================
    116: # FILTRAGE
    117: # ============================================================================
    118: 
    119: def should_exclude(path, config):
    120:     path_str = str(path)
    121:     path_parts = path.parts
    122:     
    123:     exclude_dirs = config.get("EXCLUDE_DIRS", [])
    124:     for part in path_parts:
    125:         if part in exclude_dirs:
    126:             return True
    127:     
    128:     exclude_patterns = config.get("EXCLUDE_PATTERNS", [])
    129:     for pattern in exclude_patterns:
    130:         if pattern.endswith('/*'):
    131:             dir_pattern = pattern[:-2]
    132:             if f"{dir_pattern}{os.sep}" in path_str or f"{dir_pattern}/" in path_str:
    133:                 return True
    134:         elif fnmatch.fnmatch(path.name, pattern):
    135:             return True
    136:         elif fnmatch.fnmatch(path_str, pattern):
    137:             return True
    138:     
    139:     return False
    140: 
    141: def should_include(path, config):
    142:     if path.is_dir():
    143:         return False
    144:     
    145:     include_patterns = config.get("INCLUDE_PATTERNS", ["*"])
    146:     for pattern in include_patterns:
    147:         if fnmatch.fnmatch(path.name, pattern):
    148:             return True
    149:     
    150:     return False
    151: 
    152: def collect_files(project_path, config):
    153:     all_files = []
    154:     project_root = Path(project_path)
    155:     
    156:     if not project_root.exists():
    157:         return []
    158:     
    159:     for root, dirs, files in os.walk(project_root):
    160:         dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d, config)]
    161:         
    162:         for file in files:
    163:             file_path = Path(root) / file
    164:             
    165:             if should_exclude(file_path, config):
    166:                 continue
    167:             
    168:             if should_include(file_path, config):
    169:                 all_files.append(file_path)
    170:     
    171:     return sorted(all_files)
    172: 
    173: 
    174: # ============================================================================
    175: # LECTURE DE FICHIERS
    176: # ============================================================================
    177: 
    178: def read_file_with_line_numbers(file_path):
    179:     encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    180:     
    181:     for encoding in encodings:
    182:         try:
    183:             with open(file_path, 'r', encoding=encoding) as f:
    184:                 lines = f.readlines()
    185:             
    186:             numbered_lines = []
    187:             for i, line in enumerate(lines, 1):
    188:                 line = line.rstrip('\n\r')
    189:                 numbered_lines.append(f"   {i:4d}: {line}")
    190:             
    191:             return numbered_lines
    192:         except:
    193:             continue
    194:     
    195:     return [f"   [FICHIER BINAIRE]"]
    196: 
    197: 
    198: # ============================================================================
    199: # GÉNÉRATION DE PROJECT_SHARE.txt
    200: # ============================================================================
    201: 
    202: def create_full_share(config):
    203:     print("\n" + "="*50)
    204:     print("GÉNÉRATION PROJECT_SHARE.txt")
    205:     
    206:     project_path = config.get("PROJECT_PATH")
    207:     if not project_path:
    208:         print("✗ PROJECT_PATH non défini")
    209:         return False
    210:     
    211:     project_root = Path(project_path)
    212:     if not project_root.exists():
    213:         print(f"✗ Chemin introuvable")
    214:         return False
    215:     
    216:     files = collect_files(project_path, config)
    217:     
    218:     if not files:
    219:         print("✗ Aucun fichier")
    220:         return False
    221:     
    222:     output_file = project_root / "PROJECT_SHARE.txt"
    223:     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    224:     
    225:     with open(output_file, 'w', encoding='utf-8') as out_f:
    226:         out_f.write(f"PROJECT: {config.get('PROJECT_NAME')}\n")
    227:         out_f.write(f"DATE: {timestamp}\n")
    228:         out_f.write(f"FILES: {len(files)}\n")
    229:         out_f.write("=" * 80 + "\n\n")
    230:         
    231:         for file_path in files:
    232:             out_f.write(f"FICHIER : {file_path}\n")
    233:             
    234:             numbered_lines = read_file_with_line_numbers(file_path)
    235:             for line in numbered_lines:
    236:                 out_f.write(line + '\n')
    237:             
    238:             out_f.write('\n')
    239:     
    240:     if output_file.exists():
    241:         file_size = output_file.stat().st_size
    242:         file_size_mb = file_size / (1024 * 1024)
    243:         print(f"✅ {len(files)} fichiers - {file_size_mb:.2f} MB")
    244:         return True
    245:     else:
    246:         print("❌ Échec")
    247:         return False
    248: 
    249: 
    250: # ============================================================================
    251: # FONCTIONS POUR GISTS
    252: # ============================================================================
    253: 
    254: def estimate_gist_size(content):
    255:     return len(content.encode('utf-8'))
    256: 
    257: def split_files_into_gists(project_root, config):
    258:     all_files = collect_files(project_root, config)
    259:     all_files = [f for f in all_files if f.name != "PROJECT_SHARE.txt"]
    260:     
    261:     max_size_bytes = config.get("MAX_GIST_SIZE_MB", 45) * 1024 * 1024
    262:     
    263:     file_sizes = []
    264:     for file_path in all_files:
    265:         content = '\n'.join(read_file_with_line_numbers(file_path))
    266:         size = estimate_gist_size(content)
    267:         file_sizes.append({
    268:             "path": str(file_path),
    269:             "name": file_path.name,
    270:             "size": size,
    271:             "content": content
    272:         })
    273:     
    274:     file_sizes.sort(key=lambda x: x["size"], reverse=True)
    275:     
    276:     gist_parts = []
    277:     current_gist = {"index": 1, "files": [], "size": 0}
    278:     
    279:     for file_info in file_sizes:
    280:         if current_gist["size"] + file_info["size"] > max_size_bytes and current_gist["files"]:
    281:             gist_parts.append(current_gist)
    282:             current_gist = {"index": len(gist_parts) + 1, "files": [], "size": 0}
    283:         
    284:         current_gist["files"].append(file_info)
    285:         current_gist["size"] += file_info["size"]
    286:     
    287:     if current_gist["files"]:
    288:         gist_parts.append(current_gist)
    289:     
    290:     return gist_parts
    291: 
    292: def create_gist_content(gist_data, part_index, total_parts, project_name):
    293:     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    294:     
    295:     content = []
    296:     content.append("=" * 80)
    297:     content.append(f"PROJECT_SHARE - PARTIE {part_index}/{total_parts}")
    298:     content.append(f"Projet: {project_name}")
    299:     content.append(f"Date: {timestamp}")
    300:     content.append(f"Fichiers: {len(gist_data['files'])}")
    301:     content.append("=" * 80)
    302:     content.append("")
    303:     
    304:     for file_info in gist_data["files"]:
    305:         content.append("")
    306:         content.append(f"FICHIER : {file_info['path']}")
    307:         content.extend(file_info["content"].split('\n'))
    308:         content.append("")
    309:     
    310:     return '\n'.join(content)
    311: 
    312: def upload_gists_to_github(gist_parts, config, project_root):
    313:     print("\n" + "="*50)
    314:     print("UPLOAD GISTS")
    315:     
    316: "**********"
    317: "**********":
    318: "**********": ").strip()
    319: "**********":
    320: "**********"
    321:             return None
    322: "**********"
    323:     
    324: "**********": f"token {github_token}"}
    325:     
    326:     try:
    327:         test_response = requests.get("https://api.github.com/user", headers=headers)
    328:         if test_response.status_code != 200:
    329: "**********"
    330:             return None
    331:     except:
    332:         print("✗ Erreur connexion")
    333:         return None
    334:     
    335:     uploaded_gists = []
    336:     
    337:     for i, gist_data in enumerate(gist_parts):
    338:         part_index = i + 1
    339:         total_parts = len(gist_parts)
    340:         size_mb = gist_data["size"] / (1024 * 1024)
    341:         
    342:         content = create_gist_content(gist_data, part_index, total_parts, config.get('PROJECT_NAME'))
    343:         filename = f"PROJECT_PART{part_index:02d}_of_{total_parts:02d}.txt"
    344:         
    345:         data = {
    346:             "description": f"{config.get('PROJECT_NAME')} - Partie {part_index}/{total_parts}",
    347:             "public": True,
    348:             "files": {filename: {"content": content}}
    349:         }
    350:         
    351:         try:
    352:             response = requests.post("https://api.github.com/gists", headers=headers, json=data, timeout=120)
    353:             
    354:             if response.status_code == 201:
    355:                 gist_response = response.json()
    356:                 gist_url = gist_response["html_url"]
    357:                 
    358:                 print(f"  ✅ {gist_url} ({size_mb:.2f} MB)")
    359:                 
    360:                 uploaded_gists.append({
    361:                     "part": part_index,
    362:                     "url": gist_url,
    363:                     "size_mb": round(size_mb, 2)
    364:                 })
    365:                 
    366:             else:
    367:                 print(f"  ❌ Erreur part {part_index}")
    368:         except Exception as e:
    369:             print(f"  ❌ Exception part {part_index}")
    370:     
    371:     return uploaded_gists
    372: 
    373: def save_gist_index(uploaded_gists, config, project_root):
    374:     if not uploaded_gists:
    375:         return
    376:     
    377:     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    378:     index_file = project_root / "GISTS_INDEX.txt"
    379:     
    380:     with open(index_file, 'w', encoding='utf-8') as f:
    381:         f.write(f"GISTS - {config.get('PROJECT_NAME')}\n")
    382:         f.write(f"Date: {timestamp}\n")
    383:         f.write(f"Total: {len(uploaded_gists)}\n")
    384:         f.write("-"*40 + "\n\n")
    385:         
    386:         for gist in sorted(uploaded_gists, key=lambda x: x['part']):
    387:             f.write(f"Partie {gist['part']:02d}: {gist['url']}\n")
    388:     
    389:     print(f"\n📄 Index: {index_file.name}")
    390: 
    391: def share_to_github_gists(config, project_root):
    392:     print("\n" + "="*50)
    393:     print("SHARE TO GISTS")
    394:     
    395:     gist_parts = split_files_into_gists(str(project_root), config)
    396:     
    397:     if not gist_parts:
    398:         print("✗ Aucun fichier")
    399:         return False
    400:     
    401:     total_files = sum(len(p['files']) for p in gist_parts)
    402:     total_size = sum(p['size'] for p in gist_parts) / (1024 * 1024)
    403:     
    404:     print(f"📊 {total_files} fichiers - {total_size:.2f} MB - {len(gist_parts)} Gist(s)")
    405:     
    406:     uploaded = upload_gists_to_github(gist_parts, config, project_root)
    407:     
    408:     if uploaded:
    409:         save_gist_index(uploaded, config, project_root)
    410:         print("\n✅ Terminé")
    411:         return True
    412:     else:
    413:         print("\n❌ Échec")
    414:         return False
    415: 
    416: 
    417: # ============================================================================
    418: # GIT COMMIT + PUSH
    419: # ============================================================================
    420: 
    421: def git_commit_push_publish(config, project_root):
    422:     print("\n" + "="*50)
    423:     print("GIT COMMIT + PUSH")
    424:     
    425:     try:
    426:         status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=project_root)
    427:         
    428:         if not status.stdout.strip():
    429:             print("ℹ Aucun changement")
    430:             return True
    431:         
    432:         files_changed = len(status.stdout.strip().split('\n'))
    433:         print(f"📝 {files_changed} fichier(s)")
    434:         
    435:         default_msg = f"Màj {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    436:         commit_msg = input(f"Message (défaut): ").strip() or default_msg
    437:         
    438:         subprocess.run(["git", "add", "."], cwd=project_root, check=True)
    439:         subprocess.run(["git", "commit", "-m", commit_msg], cwd=project_root, check=True)
    440:         
    441:         branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=project_root)
    442:         current_branch = branch.stdout.strip() or "main"
    443:         
    444:         result = subprocess.run(["git", "push", "origin", current_branch], capture_output=True, text=True, cwd=project_root)
    445:         
    446:         if result.returncode == 0:
    447:             print("✓ Push OK")
    448:             return True
    449:         else:
    450:             print("✗ Push échec")
    451:             return False
    452:             
    453:     except Exception:
    454:         print("✗ Erreur Git")
    455:         return False
    456: 
    457: 
    458: # ============================================================================
    459: # MENU PRINCIPAL
    460: # ============================================================================
    461: 
    462: def clear_screen():
    463:     os.system('cls' if os.name == 'nt' else 'clear')
    464: 
    465: def main_menu(config, project_root):
    466:     while True:
    467:         clear_screen()
    468:         print("\n" + "="*50)
    469:         print(f"SHARE GITHUB V9.8 - {config.get('PROJECT_NAME')}")
    470:         print("="*50)
    471:         print("1. GÉNÉRER PROJECT_SHARE.txt")
    472:         print("2. GIT COMMIT + PUSH")
    473:         print("3. SHARE TO GITHUB GISTS")
    474:         print("4. EXIT")
    475:         print("-"*50)
    476:         
    477:         choice = input("Choix: ").strip()
    478:         
    479:         if choice == "1":
    480:             create_full_share(config)
    481:         elif choice == "2":
    482:             git_commit_push_publish(config, project_root)
    483:         elif choice == "3":
    484:             share_to_github_gists(config, project_root)
    485:         elif choice == "4":
    486:             print("\nAu revoir!")
    487:             break
    488:         
    489:         input("\nAppuyez sur ENTER...")
    490: 
    491: 
    492: # ============================================================================
    493: # POINT D'ENTRÉE
    494: # ============================================================================
    495: 
    496: if __name__ == "__main__":
    497:     config = Config()
    498:     
    499:     try:
    500:         main_menu(config, config.project_root)
    501:     except KeyboardInterrupt:
    502:         print("\n\nInterrompu.")
    503:     except Exception as e:
    504:         print(f"\n✗ Erreur: {e}")
    498:     
    499:     try:
    500:         main_menu(config, config.project_root)
    501:     except KeyboardInterrupt:
    502:         print("\n\nInterrompu.")
    503:     except Exception as e:
    504:         print(f"\n✗ Erreur: {e}")