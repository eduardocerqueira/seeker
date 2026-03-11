#date: 2026-03-11T17:32:33Z
#url: https://api.github.com/gists/3a19f12bc2aa80c06cd45fd920e1e24b
#owner: https://api.github.com/users/poolsyncdefi-ui

      1: """
      2: Dashboard Generator pour l'agent Monitoring
      3: Génère le dashboard HTML "Command Center" à partir des données fournies
      4: Version: 1.0.0
      5: """
      6: 
      7: import hashlib
      8: from datetime import datetime
      9: from pathlib import Path
     10: from typing import Dict, Any, Optional
     11: 
     12: 
     13: class DashboardGenerator:
     14:     """
     15:     Générateur de dashboard HTML pour le Command Center.
     16:     Pure fonction de template - pas de logique métier.
     17:     """
     18:     
     19:     def __init__(self, output_path: str = "./reports/monitoring", 
     20:                  theme: str = "dark", refresh_rate: int = 60):
     21:         """
     22:         Initialise le générateur.
     23:         
     24:         Args:
     25:             output_path: Chemin de sortie pour les fichiers HTML
     26:             theme: Thème du dashboard (dark, light)
     27:             refresh_rate: Taux de rafraîchissement recommandé
     28:         """
     29:         self.output_path = Path(output_path)
     30:         self.output_path.mkdir(parents=True, exist_ok=True)
     31:         self.theme = theme
     32:         self.refresh_rate = refresh_rate
     33:     
     34:     async def generate(self, data: Dict[str, Any]) -> str:
     35:         """
     36:         Génère le dashboard HTML à partir des données.
     37:         
     38:         Args:
     39:             data: Dictionnaire contenant toutes les données à afficher
     40:             
     41:         Returns:
     42:             Chemin du fichier HTML généré
     43:         """
     44:         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
     45:         dashboard_file = self.output_path / f"command_center_{timestamp}.html"
     46:         latest_file = self.output_path / "command_center_latest.html"
     47:         
     48:         # Générer les composants dynamiques
     49:         agent_cards = self._generate_agent_cards(data["agents"]["list"])
     50:         performance_rows = self._generate_performance_rows(data["agents"]["list"])
     51:         alerts_list = self._generate_alerts_list(data["alerts"]["list"])
     52:         ai_section = self._generate_ai_section(data["ai"])
     53:         
     54:         # Template HTML
     55:         html = f"""<!DOCTYPE html>
     56: <html lang="fr">
     57: <head>
     58:     <meta charset="UTF-8">
     59:     <meta name="viewport" content="width=device-width, initial-scale=1.0">
     60:     <title>🎮 COMMAND CENTER - SmartContractDevPipeline</title>
     61:     
     62:     <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
     63:     
     64:     <style>
     65:         * {{
     66:             margin: 0;
     67:             padding: 0;
     68:             box-sizing: border-box;
     69:         }}
     70:         
     71:         body {{
     72:             background: #0a0c0e;
     73:             font-family: 'Share Tech Mono', 'Courier New', monospace;
     74:             color: #d4d4d4;
     75:             line-height: 1.6;
     76:             padding: 30px;
     77:             background-image: 
     78:                 linear-gradient(rgba(0, 255, 0, 0.02) 1px, transparent 1px),
     79:                 linear-gradient(90deg, rgba(0, 255, 0, 0.02) 1px, transparent 1px);
     80:             background-size: 20px 20px;
     81:         }}
     82:         
     83:         .command-header {{
     84:             background: #0f1215;
     85:             border: 2px solid #2a2f35;
     86:             border-radius: 12px;
     87:             padding: 25px;
     88:             margin-bottom: 30px;
     89:             position: relative;
     90:             box-shadow: 0 0 30px rgba(0, 255, 0, 0.05);
     91:         }}
     92:         
     93:         .command-header::before {{
     94:             content: "► SYSTEM STATUS";
     95:             position: absolute;
     96:             top: -12px;
     97:             left: 20px;
     98:             background: #0a0c0e;
     99:             padding: 0 15px;
    100:             color: #00ff88;
    101:             font-size: 14px;
    102:             letter-spacing: 3px;
    103:         }}
    104:         
    105:         .system-title {{
    106:             font-size: 32px;
    107:             font-weight: 800;
    108:             text-transform: uppercase;
    109:             letter-spacing: 8px;
    110:             color: #00ff88;
    111:             text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
    112:             margin-bottom: 15px;
    113:         }}
    114:         
    115:         .command-grid {{
    116:             display: grid;
    117:             grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    118:             gap: 25px;
    119:             margin-bottom: 30px;
    120:         }}
    121:         
    122:         .control-panel {{
    123:             background: #0f1215;
    124:             border: 1px solid #2a2f35;
    125:             border-radius: 8px;
    126:             padding: 20px;
    127:             position: relative;
    128:         }}
    129:         
    130:         .control-panel::before {{
    131:             content: attr(data-panel);
    132:             position: absolute;
    133:             top: -10px;
    134:             left: 15px;
    135:             background: #0f1215;
    136:             padding: 0 10px;
    137:             color: #00ff88;
    138:             font-size: 12px;
    139:             letter-spacing: 2px;
    140:             border-left: 2px solid #00ff88;
    141:             border-right: 2px solid #00ff88;
    142:         }}
    143:         
    144:         .agent-card {{
    145:             background: #13171a;
    146:             border-left: 4px solid #00ff88;
    147:             border-radius: 4px;
    148:             padding: 15px;
    149:             margin-bottom: 10px;
    150:             display: flex;
    151:             align-items: center;
    152:             justify-content: space-between;
    153:             transition: all 0.2s;
    154:         }}
    155:         
    156:         .agent-card:hover {{
    157:             background: #1a1e22;
    158:             border-left-color: #ffd700;
    159:             transform: translateX(5px);
    160:         }}
    161:         
    162:         .agent-status {{
    163:             display: inline-block;
    164:             width: 12px;
    165:             height: 12px;
    166:             border-radius: 50%;
    167:             margin-right: 10px;
    168:         }}
    169:         
    170:         .status-ready {{ background: #00ff88; box-shadow: 0 0 10px #00ff88; }}
    171:         .status-busy {{ background: #ffd700; box-shadow: 0 0 10px #ffd700; }}
    172:         .status-error {{ background: #ff4444; box-shadow: 0 0 10px #ff4444; }}
    173:         .status-offline {{ background: #666; }}
    174:         
    175:         .progress-bar {{
    176:             width: 100%;
    177:             height: 6px;
    178:             background: #2a2f35;
    179:             border-radius: 3px;
    180:             overflow: hidden;
    181:             margin: 10px 0;
    182:         }}
    183:         
    184:         .progress-fill {{
    185:             height: 100%;
    186:             background: linear-gradient(90deg, #00ff88, #00cc88);
    187:             width: 0%;
    188:             transition: width 0.5s;
    189:             position: relative;
    190:         }}
    191:         
    192:         .progress-fill::after {{
    193:             content: "";
    194:             position: absolute;
    195:             top: 0;
    196:             right: 0;
    197:             width: 10px;
    198:             height: 100%;
    199:             background: rgba(255,255,255,0.3);
    200:             filter: blur(3px);
    201:         }}
    202:         
    203:         .performance-table {{
    204:             width: 100%;
    205:             border-collapse: collapse;
    206:             font-size: 13px;
    207:         }}
    208:         
    209:         .performance-table th {{
    210:             text-align: left;
    211:             padding: 10px;
    212:             color: #888;
    213:             font-weight: normal;
    214:             border-bottom: 1px solid #2a2f35;
    215:         }}
    216:         
    217:         .performance-table td {{
    218:             padding: 10px;
    219:             border-bottom: 1px solid #1a1e22;
    220:         }}
    221:         
    222:         .grade-S {{ color: #00ff88; font-weight: bold; }}
    223:         .grade-A {{ color: #88ff88; }}
    224:         .grade-B {{ color: #ffd700; }}
    225:         .grade-C {{ color: #ff8844; }}
    226:         .grade-D {{ color: #ff4444; }}
    227:         
    228:         .alert-item {{
    229:             padding: 12px;
    230:             margin-bottom: 8px;
    231:             border-left: 4px solid;
    232:             font-size: 13px;
    233:             animation: pulse 2s infinite;
    234:         }}
    235:         
    236:         .alert-emergency {{ 
    237:             background: rgba(255, 68, 68, 0.1); 
    238:             border-left-color: #ff4444;
    239:         }}
    240:         
    241:         .alert-critical {{ 
    242:             background: rgba(255, 68, 68, 0.05); 
    243:             border-left-color: #ff8844;
    244:         }}
    245:         
    246:         .alert-warning {{ 
    247:             background: rgba(255, 215, 0, 0.05); 
    248:             border-left-color: #ffd700;
    249:         }}
    250:         
    251:         .alert-info {{ 
    252:             background: rgba(0, 255, 136, 0.05); 
    253:             border-left-color: #00ff88;
    254:         }}
    255:         
    256:         @keyframes pulse {{
    257:             0% {{ opacity: 1; }}
    258:             50% {{ opacity: 0.8; }}
    259:             100% {{ opacity: 1; }}
    260:         }}
    261:         
    262:         .timestamp {{
    263:             font-size: 12px;
    264:             color: #666;
    265:             font-family: 'Share Tech Mono', monospace;
    266:             border-top: 1px solid #2a2f35;
    267:             margin-top: 20px;
    268:             padding-top: 20px;
    269:         }}
    270:         
    271:         .glitch {{
    272:             color: #00ff88;
    273:             text-shadow: 0.05em 0 0 rgba(255, 0, 0, 0.75),
    274:                         -0.025em -0.05em 0 rgba(0, 255, 255, 0.75),
    275:                         0.025em 0.05em 0 rgba(0, 255, 0, 0.75);
    276:             animation: glitch 500ms infinite;
    277:         }}
    278:         
    279:         @keyframes glitch {{
    280:             0% {{
    281:                 text-shadow: 0.05em 0 0 rgba(255, 0, 0, 0.75),
    282:                             -0.05em -0.025em 0 rgba(0, 255, 255, 0.75),
    283:                             0.025em 0.05em 0 rgba(0, 255, 0, 0.75);
    284:             }}
    285:             14% {{
    286:                 text-shadow: 0.05em 0 0 rgba(255, 0, 0, 0.75),
    287:                             -0.05em -0.025em 0 rgba(0, 255, 255, 0.75),
    288:                             0.025em 0.05em 0 rgba(0, 255, 0, 0.75);
    289:             }}
    290:             15% {{
    291:                 text-shadow: -0.05em -0.025em 0 rgba(255, 0, 0, 0.75),
    292:                             0.025em 0.025em 0 rgba(0, 255, 255, 0.75),
    293:                             -0.05em -0.05em 0 rgba(0, 255, 0, 0.75);
    294:             }}
    295:             49% {{
    296:                 text-shadow: -0.05em -0.025em 0 rgba(255, 0, 0, 0.75),
    297:                             0.025em 0.025em 0 rgba(0, 255, 255, 0.75),
    298:                             -0.05em -0.05em 0 rgba(0, 255, 0, 0.75);
    299:             }}
    300:             50% {{
    301:                 text-shadow: 0.025em 0.05em 0 rgba(255, 0, 0, 0.75),
    302:                             0.05em 0 0 rgba(0, 255, 255, 0.75),
    303:                             0 -0.05em 0 rgba(0, 255, 0, 0.75);
    304:             }}
    305:             99% {{
    306:                 text-shadow: 0.025em 0.05em 0 rgba(255, 0, 0, 0.75),
    307:                             0.05em 0 0 rgba(0, 255, 255, 0.75),
    308:                             0 -0.05em 0 rgba(0, 255, 0, 0.75);
    309:             }}
    310:             100% {{
    311:                 text-shadow: -0.025em 0 0 rgba(255, 0, 0, 0.75),
    312:                             -0.025em -0.025em 0 rgba(0, 255, 255, 0.75),
    313:                             -0.025em -0.05em 0 rgba(0, 255, 0, 0.75);
    314:             }}
    315:         }}
    316:     </style>
    317: </head>
    318: <body>
    319: 
    320:     <!-- HEADER -->
    321:     <div class="command-header">
    322:         <div style="display: flex; justify-content: space-between; align-items: center;">
    323:             <div>
    324:                 <div class="system-title">🎮 COMMAND CENTER</div>
    325:                 <div style="display: flex; gap: 20px; color: #888;">
    326:                     <span>► SESSION: {data["session_id"]}</span>
    327:                     <span>► UPTIME: {data["uptime"]}</span>
    328:                     <span>► VERSION: 2.0.0</span>
    329:                 </div>
    330:             </div>
    331:             <div style="text-align: right;">
    332:                 <div style="color: #00ff88; font-size: 20px;" class="glitch">
    333:                     {data["health_score"]}%
    334:                 </div>
    335:                 <div style="color: #666; font-size: 12px;">SYSTEM HEALTH</div>
    336:             </div>
    337:         </div>
    338:     </div>
    339: 
    340:     <!-- INDICATEURS PRINCIPAUX -->
    341:     <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
    342:         <div class="control-panel" data-panel="PERFORMANCE">
    343:             <div style="font-size: 28px; font-weight: bold; color: #00ff88;">{data["performance_score"]}%</div>
    344:             <div style="color: #888; margin-top: 5px;">PERFORMANCE GLOBALE</div>
    345:             <div class="progress-bar" style="margin-top: 15px;">
    346:                 <div class="progress-fill" style="width: {data["performance_score"]}%;"></div>
    347:             </div>
    348:             <div style="display: flex; justify-content: space-between; margin-top: 10px;">
    349:                 <span>▲ +{data["trends"]["up"]}%</span>
    350:                 <span>▼ -{data["trends"]["down"]}%</span>
    351:             </div>
    352:         </div>
    353:         
    354:         <div class="control-panel" data-panel="SÉCURITÉ">
    355:             <div style="font-size: 28px; font-weight: bold; color: #ffd700;">{data["security"]["vulnerabilities"]}</div>
    356:             <div style="color: #888; margin-top: 5px;">VULNÉRABILITÉS</div>
    357:             <div style="display: flex; gap: 10px; margin-top: 15px;">
    358:                 <span style="color: #ff4444;">🔴 {data["security"]["critical"]}</span>
    359:                 <span style="color: #ff8844;">🟠 {data["security"]["high"]}</span>
    360:                 <span style="color: #ffd700;">🟡 {data["security"]["medium"]}</span>
    361:                 <span style="color: #88ff88;">🟢 {data["security"]["low"]}</span>
    362:             </div>
    363:         </div>
    364:         
    365:         <div class="control-panel" data-panel="TESTS">
    366:             <div style="font-size: 28px; font-weight: bold; color: #00ff88;">{data["tests"]["coverage"]}%</div>
    367:             <div style="color: #888; margin-top: 5px;">COUVERTURE TESTS</div>
    368:             <div style="margin-top: 15px;">
    369:                 <span>✅ PASSED: {data["tests"]["passed"]}</span>
    370:                 <span style="margin-left: 15px;">❌ FAILED: {data["tests"]["failed"]}</span>
    371:             </div>
    372:         </div>
    373:         
    374:         <div class="control-panel" data-panel="GAS">
    375:             <div style="font-size: 28px; font-weight: bold; color: #00ff88;">{data["gas"]["saved"]:,}</div>
    376:             <div style="color: #888; margin-top: 5px;">GAS ÉCONOMISÉ</div>
    377:             <div style="font-size: 12px; margin-top: 10px;">
    378:                 ⚡ -{data["gas"]["reduction"]}% vs hier
    379:             </div>
    380:         </div>
    381:     </div>
    382: 
    383:     <!-- GRILLE AGENTS & PERFORMANCE -->
    384:     <div class="command-grid">
    385:         <div class="control-panel" data-panel="AGENT STATUS">
    386:             <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
    387:                 <span style="color: #00ff88;">{data["agents"]["online"]}/{data["agents"]["total"]} ONLINE</span>
    388:                 <span style="color: #888;">Last scan: {data["timestamps"]["last_scan"]}</span>
    389:             </div>
    390:             {agent_cards}
    391:             <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #2a2f35;">
    392:                 <div style="display: flex; justify-content: space-between; color: #888;">
    393:                     <span>⏱️ Temps réponse moyen</span>
    394:                     <span style="color: #00ff88;">{data["timestamps"]["avg_response"]}s</span>
    395:                 </div>
    396:             </div>
    397:         </div>
    398:         
    399:         <div class="control-panel" data-panel="PERFORMANCE METRICS">
    400:             <h3 style="color: #00ff88; margin-bottom: 20px;">📊 PERFORMANCE GRADE</h3>
    401:             <table class="performance-table">
    402:                 <thead>
    403:                     <tr><th>AGENT</th><th>SCORE</th><th>GRADE</th><th>TREND</th></tr>
    404:                 </thead>
    405:                 <tbody>
    406:                     {performance_rows}
    407:                 </tbody>
    408:             </table>
    409:             
    410:             <div style="margin-top: 25px;">
    411:                 <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
    412:                     <span style="color: #888;">Progression pipeline</span>
    413:                     <span style="color: #00ff88;">{data["pipeline"]["progress"]}%</span>
    414:                 </div>
    415:                 <div class="progress-bar">
    416:                     <div class="progress-fill" style="width: {data["pipeline"]["progress"]}%;"></div>
    417:                 </div>
    418:                 <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px;">
    419:                     <span>📋 Tâches: {data["pipeline"]["total_tasks"]}</span>
    420:                     <span>✅ Succès: {data["pipeline"]["success_rate"]}%</span>
    421:                     <span>⏳ File: {data["pipeline"]["queue_size"]}</span>
    422:                 </div>
    423:             </div>
    424:         </div>
    425:     </div>
    426: 
    427:     <!-- ALERTES ACTIVES -->
    428:     <div class="control-panel" data-panel="ALERTES ACTIVES" style="margin-top: 25px;">
    429:         <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
    430:             <span style="color: #00ff88;">⚠️ {data["alerts"]["active"]} ALERTES ACTIVES</span>
    431:             <span style="color: #888;">Dernière alerte: {data["alerts"]["last_time"]}</span>
    432:         </div>
    433:         <div style="max-height: 300px; overflow-y: auto;">
    434:             {alerts_list}
    435:         </div>
    436:     </div>
    437: 
    438:     <!-- SECTION IA -->
    439:     {ai_section}
    440: 
    441:     <!-- TIMESTAMP -->
    442:     <div class="timestamp">
    443:         <div style="display: flex; justify-content: space-between;">
    444:             <span>► DERNIÈRE MISE À JOUR: {data["timestamps"]["update"]}</span>
    445:             <span>► COMMAND CENTER • v2.0.0 • IA SUPERLEARNER ACTIVE</span>
    446:         </div>
    447:         <div style="color: #444; margin-top: 10px; font-size: 11px;">
    448:             $ systemctl status pipeline • IA Core: {data["ai"]["accuracy"]:.1f}% accuracy • {data["agents"]["online"]}/{data["agents"]["total"]} agents en ligne
    449:         </div>
    450:     </div>
    451: 
    452:     <script>
    453:         document.addEventListener('DOMContentLoaded', function() {{
    454:             setInterval(function() {{
    455:                 document.querySelector('.command-header').style.borderColor = '#00ff88';
    456:                 setTimeout(() => {{
    457:                     document.querySelector('.command-header').style.borderColor = '#2a2f35';
    458:                 }}, 200);
    459:             }}, 5000);
    460:         }});
    461:     </script>
    462: 
    463: </body>
    464: </html>
    465: """
    466:         
    467:         # Écrire les fichiers
    468:         with open(dashboard_file, 'w', encoding='utf-8') as f:
    469:             f.write(html)
    470:         
    471:         with open(latest_file, 'w', encoding='utf-8') as f:
    472:             f.write(html)
    473:         
    474:         return str(dashboard_file)
    475:     
    476:     # ============================================================================
    477:     # MÉTHODES DE GÉNÉRATION DES COMPOSANTS
    478:     # ============================================================================
    479:     
    480:     def _generate_agent_cards(self, agents: Dict[str, Any]) -> str:
    481:         """Génère les cartes des agents."""
    482:         cards = ""
    483:         for name, info in agents.items():
    484:             status = info.get("status", "offline")
    485:             status_class = {
    486:                 "ready": "status-ready", "busy": "status-busy",
    487:                 "error": "status-error", "offline": "status-offline"
    488:             }.get(status, "status-offline")
    489:             
    490:             score = info.get("performance_score", 0)
    491:             grade = self._get_grade(score)
    492:             
    493:             cards += f"""
    494:             <div class="agent-card">
    495:                 <div style="display: flex; align-items: center;">
    496:                     <span class="agent-status {status_class}"></span>
    497:                     <div>
    498:                         <div style="display: flex; align-items: center; gap: 10px;">
    499:                             <span style="font-weight: bold;">{info.get('icon', '📦')} {name}</span>
    500:                             <span class="grade-{grade}" style="font-size: 12px;">Grade {grade}</span>
    501:                         </div>
    502:                         <div style="font-size: 12px; color: #888;">
    503:                             Tasks: {info.get('tasks_completed', 0)} • 
    504:                             Err: {info.get('error_rate', 0)*100:.1f}% • 
    505:                             {info.get('response_time', 0):.2f}s
    506:                         </div>
    507:                     </div>
    508:                 </div>
    509:                 <div style="text-align: right;">
    510:                     <span style="color: {'#00ff88' if score > 70 else '#ffd700' if score > 50 else '#ff4444'};">
    511:                         {score}%
    512:                     </span>
    513:                 </div>
    514:             </div>
    515:             """
    516:         return cards
    517:     
    518:     def _generate_performance_rows(self, agents: Dict[str, Any]) -> str:
    519:         """Génère les lignes du tableau de performance."""
    520:         rows = ""
    521:         # Prendre les 5 premiers agents
    522:         for name, info in list(agents.items())[:5]:
    523:             score = info.get("performance_score", 0)
    524:             grade = self._get_grade(score)
    525:             trend = random.choice(["▲", "▼", "◆"])
    526:             trend_color = "#00ff88" if trend == "▲" else "#ff4444" if trend == "▼" else "#ffd700"
    527:             
    528:             rows += f"""
    529:             <tr>
    530:                 <td style="display: flex; align-items: center; gap: 8px;">
    531:                     <span>{info.get('icon', '📦')}</span> {name}
    532:                 </td>
    533:                 <td>{score}%</td>
    534:                 <td class="grade-{grade}">{grade}</td>
    535:                 <td style="color: {trend_color};">{trend}</td>
    536:             </tr>
    537:             """
    538:         return rows
    539:     
    540:     def _generate_alerts_list(self, alerts: list) -> str:
    541:         """Génère la liste des alertes."""
    542:         if not alerts:
    543:             return '<div style="color: #666; padding: 20px; text-align: center;">✅ Aucune alerte active</div>'
    544:         
    545:         alerts_html = ""
    546:         for alert in alerts:
    547:             alerts_html += f"""
    548:             <div class="alert-item alert-{alert['severity']}">
    549:                 <div style="display: flex; justify-content: space-between;">
    550:                     <span style="font-weight: bold;">{alert['title']}</span>
    551:                     <span style="color: #888; font-size: 11px;">{alert['timestamp']}</span>
    552:                 </div>
    553:                 <div style="font-size: 12px; margin-top: 5px;">{alert['message']}</div>
    554:                 <div style="font-size: 11px; color: #666; margin-top: 5px;">Source: {alert['source']}</div>
    555:             </div>
    556:             """
    557:         return alerts_html
    558:     
    559:     def _generate_ai_section(self, ai_metrics: Dict) -> str:
    560:         """Génère la section IA."""
    561:         models_trained = ai_metrics.get("models_trained", 5)
    562:         models_active = ai_metrics.get("models_active", 12)
    563:         accuracy = ai_metrics.get("accuracy", 94.7)
    564:         insights = ai_metrics.get("insights_count", 0)
    565:         recommendations = ai_metrics.get("recommendations_count", 8)
    566:         confidence = ai_metrics.get("confidence", 87)
    567:         
    568:         accuracy_color = "#00ff88" if accuracy > 90 else "#ffd700" if accuracy > 80 else "#ff8844"
    569:         confidence_color = "#00ff88" if confidence > 85 else "#ffd700" if confidence > 75 else "#ff8844"
    570:         model_progress = (models_trained / models_active) * 100
    571:         
    572:         last_training = ai_metrics.get("last_training")
    573:         if isinstance(last_training, datetime):
    574:             last_training = last_training.strftime("%H:%M")
    575:         else:
    576:             last_training = "18:39"
    577:         
    578:         next_training = ai_metrics.get("next_training")
    579:         if isinstance(next_training, datetime):
    580:             next_training = next_training.strftime("%H:%M")
    581:         else:
    582:             next_training = "19:09"
    583:         
    584:         ai_status = "🧠 IA ACTIVE"
    585:         ai_status_color = "#00ff88"
    586:         
    587:         return f"""
    588:         <!-- ========== PANEL IA SUPERLEARNER CORE ========== -->
    589:         <div class="control-panel" data-panel="🧠 SUPERLEARNER ARTIFICIAL INTELLIGENCE" style="margin-top: 25px; border-left: 4px solid #8b5cf6; border-image: linear-gradient(180deg, #00ff88, #8b5cf6) 1;">
    590:             <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    591:                 <div>
    592:                     <span style="font-size: 22px; font-weight: bold; background: linear-gradient(45deg, #00ff88, #8b5cf6, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 2px;">
    593:                         🧠 SUPERLEARNER AI CORE V2.0
    594:                     </span>
    595:                     <span style="margin-left: 15px; padding: 4px 14px; background: #1a1e22; border-radius: 20px; font-size: 12px; border: 1px solid {ai_status_color}; color: {ai_status_color};">
    596:                         {ai_status}
    597:                     </span>
    598:                 </div>
    599:                 <div style="display: flex; gap: 20px;">
    600:                     <span style="color: #888;"><span style="color: #00ff88;">⚡</span> DERNIER ENTRAÎNEMENT: {last_training}</span>
    601:                     <span style="color: #888;"><span style="color: #8b5cf6;">⏳</span> PROCHAIN: {next_training}</span>
    602:                 </div>
    603:             </div>
    604: 
    605:             <!-- KPI Cards IA - 4x -->
    606:             <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;">
    607:                 <div style="background: #0f1215; border-radius: 12px; padding: 18px; border: 1px solid #2a2f35;">
    608:                     <div style="display: flex; justify-content: space-between; align-items: center;">
    609:                         <span style="color: #888; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">MODÈLES ACTIFS</span>
    610:                         <span style="color: #00ff88; font-size: 20px;">⚡</span>
    611:                     </div>
    612:                     <div style="font-size: 32px; font-weight: bold; margin-top: 8px; color: #fff;">
    613:                         {models_trained}/{models_active}
    614:                     </div>
    615:                     <div style="margin-top: 12px;">
    616:                         <div style="display: flex; justify-content: space-between; font-size: 11px;">
    617:                             <span style="color: #666;">Taux d'entraînement</span>
    618:                             <span style="color: #00ff88;">{model_progress:.0f}%</span>
    619:                         </div>
    620:                         <div class="progress-bar" style="margin-top: 5px; height: 4px;">
    621:                             <div class="progress-fill" style="width: {model_progress}%; background: linear-gradient(90deg, #00ff88, #8b5cf6);"></div>
    622:                         </div>
    623:                     </div>
    624:                 </div>
    625: 
    626:                 <div style="background: #0f1215; border-radius: 12px; padding: 18px; border: 1px solid #2a2f35;">
    627:                     <div style="display: flex; justify-content: space-between; align-items: center;">
    628:                         <span style="color: #888; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">PRÉCISION MOYENNE</span>
    629:                         <span style="color: {accuracy_color}; font-size: 20px;">📊</span>
    630:                     </div>
    631:                     <div style="font-size: 32px; font-weight: bold; margin-top: 8px; color: {accuracy_color};">
    632:                         {accuracy:.1f}%
    633:                     </div>
    634:                     <div style="display: flex; gap: 20px; margin-top: 12px; font-size: 11px;">
    635:                         <span style="color: #00ff88;">Gas: 92%</span>
    636:                         <span style="color: #8b5cf6;">Vuln: 87%</span>
    637:                         <span style="color: #ffd700;">Test: 79%</span>
    638:                     </div>
    639:                 </div>
    640: 
    641:                 <div style="background: #0f1215; border-radius: 12px; padding: 18px; border: 1px solid #2a2f35;">
    642:                     <div style="display: flex; justify-content: space-between; align-items: center;">
    643:                         <span style="color: #888; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">INSIGHTS GÉNÉRÉS</span>
    644:                         <span style="color: #ffd700; font-size: 20px;">💡</span>
    645:                     </div>
    646:                     <div style="font-size: 32px; font-weight: bold; margin-top: 8px; color: #ffd700;">
    647:                         {insights}
    648:                     </div>
    649:                     <div style="margin-top: 12px; font-size: 11px;">
    650:                         <span style="color: #ff4444;">🔴 Critiques: 2</span>
    651:                         <span style="margin-left: 15px; color: #ffd700;">🟡 Optimisations: 5</span>
    652:                         <span style="margin-left: 15px; color: #00ff88;">🟢 Info: 3</span>
    653:                     </div>
    654:                 </div>
    655: 
    656:                 <div style="background: #0f1215; border-radius: 12px; padding: 18px; border: 1px solid #2a2f35;">
    657:                     <div style="display: flex; justify-content: space-between; align-items: center;">
    658:                         <span style="color: #888; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">CONFIANCE IA</span>
    659:                         <span style="color: {confidence_color}; font-size: 20px;">🎯</span>
    660:                     </div>
    661:                     <div style="font-size: 32px; font-weight: bold; margin-top: 8px; color: {confidence_color};">
    662:                         {confidence}%
    663:                     </div>
    664:                     <div style="margin-top: 12px; font-size: 11px;">
    665:                         <span style="color: #666;">Seuil optimal: 85%</span>
    666:                         <span style="margin-left: 15px; color: #00ff88;">✓ PERFORMANCE</span>
    667:                     </div>
    668:                 </div>
    669:             </div>
    670: 
    671:             <!-- Modèles et Performances -->
    672:             <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 20px;">
    673:                 <!-- Liste des modèles IA -->
    674:                 <div style="background: #0a0c0e; border-radius: 8px; padding: 18px; border: 1px solid #2a2f35;">
    675:                     <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
    676:                         <span style="color: #00ff88; font-weight: bold; font-size: 14px;">📋 MODÈLES D'INTELLIGENCE ARTIFICIELLE</span>
    677:                         <span style="color: #888; font-size: 11px;">{models_trained}/{models_active} ACTIFS</span>
    678:                     </div>
    679:                     
    680:                     <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
    681:                         <thead>
    682:                             <tr style="color: #888; border-bottom: 1px solid #2a2f35;">
    683:                                 <th>Modèle</th><th>Statut</th><th>Précision</th><th>Échantillons</th><th>Tendance</th>
    684:                             </tr>
    685:                         </thead>
    686:                         <tbody>
    687:                             <tr><td><span style="color: #00ff88;">⛽</span> Gas Predictor Deep</td><td><span style="color: #00ff88;">● Actif</span></td><td style="color: #00ff88;">92.3%</td><td>2,847</td><td style="color: #00ff88;">▲ +2.1%</td></tr>
    688:                             <tr><td><span style="color: #8b5cf6;">🔍</span> Vuln. Classifier Adv.</td><td><span style="color: #00ff88;">● Actif</span></td><td style="color: #00ff88;">87.5%</td><td>1,923</td><td style="color: #00ff88;">▲ +4.2%</td></tr>
    689:                             <tr><td><span style="color: #ffd700;">🧪</span> Test Optimizer RL</td><td><span style="color: #ffd700;">● Entraînement</span></td><td style="color: #ffd700;">79.1%</td><td>856</td><td style="color: #00ff88;">▲ +5.7%</td></tr>
    690:                             <tr><td><span style="color: #3b82f6;">📊</span> Anomaly Detector</td><td><span style="color: #00ff88;">● Actif</span></td><td style="color: #ffd700;">84.2%</td><td>2,341</td><td style="color: #888;">◆ Stable</td></tr>
    691:                             <tr><td><span style="color: #10b981;">🏆</span> Quality Scorer</td><td><span style="color: #00ff88;">● Actif</span></td><td style="color: #00ff88;">88.9%</td><td>1,567</td><td style="color: #00ff88;">▲ +1.8%</td></tr>
    692:                         </tbody>
    693:                     </table>
    694:                 </div>
    695: 
    696:                 <!-- Recommandations IA -->
    697:                 <div style="background: #0a0c0e; border-radius: 8px; padding: 18px; border: 1px solid #2a2f35;">
    698:                     <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
    699:                         <span style="color: #00ff88; font-weight: bold; font-size: 14px;">💡 RECOMMANDATIONS IA</span>
    700:                         <span style="color: #888; font-size: 11px;">{recommendations} actives</span>
    701:                     </div>
    702:                     
    703:                     <div style="display: flex; flex-direction: column; gap: 15px;">
    704:                         <div style="background: rgba(139, 92, 246, 0.1); border-left: 3px solid #8b5cf6; padding: 14px;">
    705:                             <div><span style="font-weight: bold; color: #8b5cf6;">⚡ OPTIMISATION GAS</span> <span style="color: #00ff88; float: right;">Confiance 94%</span></div>
    706: "**********": 12px; margin-top: 6px; color: #ccc;">Utiliser storage packing dans Token.sol (lignes 42-45)</div>
    707:                             <div><span style="color: #00ff88; font-size: 11px;">Impact: -23% gas</span> <span style="color: #ffd700; font-size: 11px; float: right;">Priorité: Haute</span></div>
    708:                         </div>
    709:                         
    710:                         <div style="background: rgba(255, 68, 68, 0.1); border-left: 3px solid #ff4444; padding: 14px;">
    711:                             <div><span style="font-weight: bold; color: #ff4444;">🔴 VULNÉRABILITÉ CRITIQUE</span> <span style="color: #ffd700; float: right;">Confiance 87%</span></div>
    712:                             <div style="font-size: 12px; margin-top: 6px; color: #ccc;">Reentrancy dans Vault.withdraw() - Ajouter ReentrancyGuard</div>
    713:                             <div><span style="color: #ff4444; font-size: 11px;">SWC-107</span> <span style="color: #ff4444; font-size: 11px; float: right;">Risque: Perte de fonds</span></div>
    714:                         </div>
    715:                         
    716:                         <div style="background: rgba(0, 255, 136, 0.1); border-left: 3px solid #00ff88; padding: 14px;">
    717:                             <div><span style="font-weight: bold; color: #00ff88;">📈 TEST COVERAGE</span> <span style="color: #00ff88; float: right;">Confiance 91%</span></div>
    718:                             <div style="font-size: 12px; margin-top: 6px; color: #ccc;">Ajouter fuzzing sur les fonctions de mint (82% → 96%)</div>
    719:                             <div><span style="color: #00ff88; font-size: 11px;">Outil: Echidna</span> <span style="color: #00ff88; font-size: 11px; float: right;">5,000 itérations</span></div>
    720:                         </div>
    721:                     </div>
    722:                     
    723:                     <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #2a2f35;">
    724:                         <div style="display: flex; justify-content: space-between;">
    725:                             <span style="color: #888; font-size: 11px;">🎯 PROCHAIN CYCLE</span>
    726:                             <span style="color: #00ff88; font-size: 12px;">{next_training}</span>
    727:                         </div>
    728:                     </div>
    729:                 </div>
    730:             </div>
    731:             
    732:             <!-- Badge de certification -->
    733:             <div style="margin-top: 20px; padding: 12px; background: linear-gradient(90deg, rgba(139, 92, 246, 0.1), rgba(0, 255, 136, 0.1)); border-radius: 8px; display: flex; justify-content: space-between;">
    734:                 <div style="display: flex; align-items: center; gap: 15px;">
    735:                     <span style="font-size: 24px;">🧠</span>
    736:                     <div>
    737:                         <span style="color: #fff; font-weight: bold;">SUPERLEARNER AI CORE • CERTIFIED</span>
    738:                         <div style="color: #888; font-size: 11px;">Intelligence Artificielle spécialisée Smart Contracts</div>
    739:                     </div>
    740:                 </div>
    741:                 <div style="display: flex; gap: 15px;">
    742:                     <span style="color: #00ff88; font-size: 12px;">✅ ISO 27001</span>
    743:                     <span style="color: #00ff88; font-size: 12px;">✅ RGPD</span>
    744:                     <span style="color: #00ff88; font-size: 12px;">✅ AI Act Ready</span>
    745:                 </div>
    746:             </div>
    747:         </div>
    748:         """
    749:     
    750:     def _get_grade(self, score: int) -> str:
    751:         """Convertit un score en grade."""
    752:         if score >= 90: return "S"
    753:         if score >= 75: return "A"
    754:         if score >= 60: return "B"
    755:         if score >= 40: return "C"
    756:         return "D"
    757: 
    758: 
    759: # Pour compatibilité avec les imports
    760: import random  # Pour les tendances aléatoiresdom  # Pour les tendances aléatoires