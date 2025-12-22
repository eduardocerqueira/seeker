#date: 2025-12-22T16:54:27Z
#url: https://api.github.com/gists/aaa13ff00b3f72dafb22aa7886649ab9
#owner: https://api.github.com/users/Schwarz402

import requests
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===== .env betöltés + alapkönyvtárak (zetbet) =====
import os as _env_os
from pathlib import Path as _env_Path

# === UNIFIED ROOT DIR (VPS + Termux SAFE) ===
from pathlib import Path
import os

def _pick_root():
    # VPS
    p = Path.home() / "zetbet"
    if p.exists():
        return p

    # Termux fallback
    for q in (
      # Path("/storage/emulated/0/Zetbet"),
      # Path("/storage/emulated/0/zetbet"),
    ):
        if q.exists():
            return q

    # last resort
    return Path.cwd()

ZETBET_ROOT = _pick_root()
LOG_DIR = ZETBET_ROOT / "logs"
CACHE_DIR = ZETBET_ROOT / "cache"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _env_load(p):
    d={}
    try:
        for line in _env_Path(p).read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k,v = line.split("=",1)
            d[k.strip()] = v.strip()
    except Exception:
        pass
    return d

_ENV_PATH = str(ZETBET_ROOT / ".env")
_ENV = _env_load(_ENV_PATH)

TIP_API_FOOTBALL_KEY = _env_os.environ.get("TIP_API_FOOTBALL_KEY", _ENV.get("TIP_API_FOOTBALL_KEY",""))



# ---- API-Football OU segédfüggvények (hozzáadva) ----
def _af_pick_ou_over_from_rows(rows, line_str):
    """
    API-Football /odds response -> Over {line_str} odds (float) vagy None
    """
    try:
        for item in (rows or []):
            for bk in (item.get("bookmakers") or []):
                for bet in (bk.get("bets") or []):
                    name = (bet.get("name") or "").lower()
                    if "over/under" in name:
                        for v in (bet.get("values") or []):
                            val = str(v.get("value","")).strip().lower()
                            if val == f"over {line_str}":
                                try:
                                    return float(v.get("odd"))
                                except Exception:
                                    pass
    except Exception:
        pass
    return None



# ====== API-Football státusz szűrők ======
FINISHED_STATUSES = {"FT","AET","PEN"}
CANCELLED_STATUSES = {"CANC","ABD","SUSP","PST"}
LIVE_STATUSES = {"1H","2H","ET","P","BT","LIVE"}

def _af_is_finished(st): return (st or "").upper() in FINISHED_STATUSES
def _af_is_dead(st): return (st or "").upper() in (FINISHED_STATUSES | CANCELLED_STATUSES)
def _af_is_live(st): return (st or "").upper() in LIVE_STATUSES

def _af_fetch_odds_rows_for_fixture(fid):
    """
    A meglévő _apif_get segéddel lekéri az adott fixture összes odds-sorát.
    """
    try:
        js = _apif_get("/odds", {"fixture": int(fid)})
        return (js or {}).get("response", [])
    except Exception:
        return []
# --- Min odds küszöbök (hozzáadva) ---
MIN_ODDS_ALL = 1.30
MIN_ODDS_PREDICT_O15 = 1.30

TIP_LOG_DIR  = _env_Path(_env_os.environ.get("TIP_LOG_DIR",  _ENV.get("TIP_LOG_DIR","/storage/emulated/0/zetbet/logs")))
TIP_CACHE_DIR= _env_Path(_env_os.environ.get("TIP_CACHE_DIR",_ENV.get("TIP_CACHE_DIR","/storage/emulated/0/zetbet/cache")))
globals()["LOG_DIR"] = LOG_DIR
globals()["CACHE_DIR"] = CACHE_DIR
try:
    TIP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    TIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Ha a fájl bárhol definiálna LOG_DIR/CACHE_DIR-t, felülírjuk a zetbet-es értékre:
try:
    LOG_DIR = TIP_LOG_DIR
except Exception:
    LOG_DIR = _env_Path("/storage/emulated/0/zetbet/logs")
try:
    CACHE_DIR = TIP_CACHE_DIR
except Exception:
    CACHE_DIR = _env_Path("/storage/emulated/0/zetbet/cache")
# ===== END .env blokk =====


# --- DAILY LOG – stabil írás a /Download/logs-ba -----------------------------
from pathlib import Path
from datetime import datetime
import re
import json, os

BASE_DIR = Path.home() / "zetbet"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


TIPLOG = LOG_DIR / "tiplog.jsonl"

def _daily_path() -> Path:
    from datetime import datetime as _dt
    return LOG_DIR / f"{_dt.now():%F}.jsonl"

def s_append_log(row: dict) -> None:
    try:
        p = _daily_path()
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        try:
            with (LOG_DIR / "global_errors.log").open("a", encoding="utf-8") as ge:
                ge.write(f"[{datetime.now().isoformat(timespec='seconds')}] APPEND_FAIL: {e}\n")
        except:
            pass
# -----------------------------------------------------------------------------



def _write_to_daily_log(entry: dict):
    from datetime import datetime
    daily_path = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.jsonl")
    with open(daily_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")



# naplózás tip_closed eseményhez
from datetime import datetime
import os
import json

def _emit_tip_closed(tip: dict, result: str):
    """
    result: 'won' vagy 'lost'
    """

    print(f"[DBG] _emit_tip_closed() tip_id={tip.get('tip_id')} event_id={tip.get('event_id')} market={tip.get('market')} result={result}")
    row = dict(tip)
    row["event"] = "tip_closed"
    row["closed_at"] = datetime.now().isoformat(timespec="seconds")
    row["result"] = result

    try:
        with TIPLOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        try:
            with (LOG_DIR / "global_errors.log").open("a", encoding="utf-8") as ge:
                ge.write(f"[{datetime.now().isoformat()}] tip_closed write error: {e}\n")
        except:
            pass

from pathlib import Path
import os

# ================== LIGA/CSAPAT SZŰRŐ (Women + U16..U22 + Liga 4) ==================
import re as _re_league

_WOMEN_TOKENS = "**********"
    "women","woman","ladies","female","girls","femen","femin","femenina","femminile",
    "damen","vrouw","kobiety","kvinner","kadin","mulheres","zeny","女子","女","női","noi","women's","womens"
]
_U_AGE_RE = _re_league.compile(r"\bU[- ]?(?:16|17|18|19|20|21|22)\b", _re_league.I)
_LIGA4_RE = _re_league.compile(r"\b(?:liga|league|division|div)\s*4\b", _re_league.I)

def _has_women_token(text: "**********":
    t = (text or "").lower()
    return any(tok in t for tok in _WOMEN_TOKENS)

def _has_u_token(text: "**********":
    return bool(_U_AGE_RE.search(text or ""))

def _is_liga4(text: str) -> bool:
    return bool(_LIGA4_RE.search(text or ""))

def s_league_guard(m) -> bool:
    """True, ha a meccs elemezhető (nem női, nem U16..U22, nem Liga 4)"""
    league = getattr(m, "league", "") or ""
    home = getattr(m, "home", "") or ""
    away = getattr(m, "away", "") or ""

    # Női jelölés a liga nevében vagy csapatnévben
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"_ "**********"h "**********"a "**********"s "**********"_ "**********"w "**********"o "**********"m "**********"e "**********"n "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"l "**********"e "**********"a "**********"g "**********"u "**********"e "**********") "**********"  "**********"o "**********"r "**********"  "**********"_ "**********"h "**********"a "**********"s "**********"_ "**********"w "**********"o "**********"m "**********"e "**********"n "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"h "**********"o "**********"m "**********"e "**********") "**********"  "**********"o "**********"r "**********"  "**********"_ "**********"h "**********"a "**********"s "**********"_ "**********"w "**********"o "**********"m "**********"e "**********"n "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"a "**********"w "**********"a "**********"y "**********") "**********": "**********"
        return False

    # U16..U22 jelölés liga- vagy csapatnévben
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"_ "**********"h "**********"a "**********"s "**********"_ "**********"u "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"l "**********"e "**********"a "**********"g "**********"u "**********"e "**********") "**********"  "**********"o "**********"r "**********"  "**********"_ "**********"h "**********"a "**********"s "**********"_ "**********"u "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"h "**********"o "**********"m "**********"e "**********") "**********"  "**********"o "**********"r "**********"  "**********"_ "**********"h "**********"a "**********"s "**********"_ "**********"u "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"a "**********"w "**********"a "**********"y "**********") "**********": "**********"
        return False

    # Liga 4/Division 4 jelölés
    if _is_liga4(league):
        return False

    return True
# ================== END LIGA/CSAPAT SZŰRŐ ==================

# ================== ROBUSZTUS HTTP-GET + CIRCUIT BREAKER ==================
import time as _t_net
import urllib.request as _u_req
import urllib.error as _u_err

_NET_CB = {"fails": 0, "open_until": 0}

def http_get(url, headers=None, timeout=7, retries=5):
    now = _t_net.time()
    if _NET_CB["open_until"] > now:
        return None

    headers = headers or {}
    req = _u_req.Request(url, headers=headers)
    delay = 0.8
    for _ in range(retries):
        try:
            with _u_req.urlopen(req, timeout=timeout) as r:
                data = r.read()
                _NET_CB["fails"] = 0
                _NET_CB["open_until"] = 0
                return data
        except _u_err.HTTPError as e:
            code_ = getattr(e, "code", 0)
            if code_ in (429, 500, 502, 503, 504):
                _t_net.sleep(delay)
                delay = min(delay * 1.7, 8.0)
                continue
            break
        except Exception:
            _t_net.sleep(delay)
            delay = min(delay * 1.7, 8.0)
            continue

    _NET_CB["fails"] += 1
    if _NET_CB["fails"] >= 3:
        _NET_CB["open_until"] = _t_net.time() + 180
    return None
# ================== END ROBUSZTUS HTTP-GET ==================

# ==== KAPCSOLÓK ====


# ==== PRE GOALSCORER (minimal, non-intrusive) ====
PRE_SCORER_ENABLE = True
PRE_SCORER_PROB_MIN = 0.28   # minimum valószínűség a pre gólszerzőhöz
PRE_SCORER_CAP = 3           # meccsenként max hány játékos javaslat

def _load_players_db(path="players_db.json"):
    try:
        return load_json(path, {})
    except Exception:
        return {}

def _normalize_team_key(name: str) -> str:
    try:
        return re.sub(r'[^a-z0-9]+', '', (name or '').lower())
    except Exception:
        return str(name or '').lower()

def _pre_scorer_candidates(home: str, away: str, league: str):
    """
    Visszaad [(player_name, prob, odd, team), ...] listát, már szűrve és rendezve.
    players_db.json struktúra ajánlott:
    {
      "teams": {
        "<norm_team_key>": {
           "players": [
               {"name":"...", "prob":0.31, "odd":2.1},
               {"name":"...", "xg":0.27}  # prob = 1-exp(-xg) fallback
           ]
        }
      }
    }
    """
    if not PRE_SCORER_ENABLE:
        return []
    db = _load_players_db()
    teams = (db.get("teams") or {}) if isinstance(db, dict) else {}
    out = []
    for team in (home, away):
        key = _normalize_team_key(team)
        entry = teams.get(key) or {}
        plist = entry.get("players") or []
        for p in plist:
            # prob derive
            prob = None
            if isinstance(p.get("prob"), (int, float)):
                prob = float(p["prob"])
            elif isinstance(p.get("xg"), (int, float)):
                try:
                    import math
                    prob = 1.0 - math.exp(-float(p["xg"]))
                except Exception:
                    prob = None
            if prob is None:
                continue
            if prob < PRE_SCORER_PROB_MIN:
                continue
            odd = p.get("odd")
            try:
                odd = float(odd) if odd is not None else None
            except Exception:
                odd = None
            out.append( (str(p.get("name") or "N/A"), float(prob), odd, team) )
    # rendezés prob szerint
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:PRE_SCORER_CAP]
# ==== END PRE GOALSCORER ====


# ====================== STATS (ikonos) + KÖZPONTI LOGOK ======================
import datetime as _dt
# ================= GOALSCORER SUPPORT (PRE + LIVE, HYBRID / NO DUMMY) =========
import json as _gs_json, re as _gs_re, unicodedata as _gs_uni, os as _gs_os

SCORER_MODE = "HYBRID"   # "ATTACH", "DISCOVER", "HYBRID"
SCORER_MIN_W = 0.80
SCORER_EXTRA_MAX = 3

def _resolve_players_db():
    env = os.environ.get("GS_PLAYERS_DB")
    if env and os.path.exists(env):
        return env
    candidates = [
        "/storage/emulated/0/Download/players_db.json",
        "/storage/downloads/players_db.json",
        os.path.join(os.path.dirname(__file__), "players_db.json"),
        "/storage/emulated/0/Tipster/players_db.json",
    ]
    for _p in candidates:
        try:
            if os.path.exists(_p):
                return _p
        except Exception:
            pass
    return None
GS_PLAYERS_DB = _resolve_players_db() or ""


def _gs_norm(s: str) -> str:
    if not s: return ""
    s = s.strip().lower()
    s = "".join(c for c in _gs_uni.normalize("NFKD", s) if _gs_uni.category(c) != "Mn")
    s = _gs_re.sub(r"[^a-z0-9]+", " ", s)
    s = _gs_re.sub(r"\s+", " ", s).strip()
    aliases = {
        "afc ajax": "ajax",
        "ajax amsterdam": "ajax",
        "psv eindhoven": "psv",
        "feyenoord rotterdam": "feyenoord",
        "ferencvarosi tc": "ferencvaros",
        "ferencvaros tc": "ferencvaros",
    }
    return aliases.get(s, s)

def _gs_load_db(path=GS_PLAYERS_DB):
    try:
        data = _gs_json.loads(open(path, "r", encoding="utf-8").read())
    except Exception as e:
        print(f"[GS] players_db betöltés hiba: {e} ({path})")
        return {}
    db = {}
    for team, arr in (data or {}).items():
        if isinstance(arr, list):
            db[_gs_norm(team)] = [{"name": str(x.get("name","")).strip(),
                                   "w": float(x.get("w", 0.0))} for x in arr if isinstance(x, dict) and x.get("name")]
    print(f"[GS] players_db betöltve: {path} | csapatok: {len(db)}")
    return db

_GS_DB = _gs_load_db()

def _gs_pick_names(tip: dict):
    h = tip.get("home") or tip.get("p1") or tip.get("team1") or tip.get("match_home") or ""
    a = tip.get("away") or tip.get("p2") or tip.get("team2") or tip.get("match_away") or ""
    if (not h or not a) and isinstance(tip.get("match"), str):
        parts = _gs_re.split(r"\s+[-–—]\s+", tip["match"])
        if len(parts) == 2:
            h = h or parts[0]; a = a or parts[1]
    return h, a

def _gs_top_players(team: str):
    key = _gs_norm(team)
    return sorted([(p["name"], float(p.get("w",0))) for p in _GS_DB.get(key,[]) if p.get("name")],
                  key=lambda x:(-x[1],x[0]))

def _gs_block(title: str, tips: list):
    print("\n" + "="*64)
    print(title)
    print("="*64)
    if not tips:
        print("(nincs mérkőzés a listában)"); return
    for i, tip in enumerate(tips,1):
        h,a = _gs_pick_names(tip)
        if not (h or a): continue
        htop, atop = _gs_top_players(h)[:3], _gs_top_players(a)[:3]
        if not htop and not atop:
            print(f"Meccs {i}: {h} – {a}: [(nincs adat)]")
        else:
            print(f"Meccs {i}: {h} – {a}")
            for nm,w in htop: print(f"  H: {nm} w={w:.2f}")
            for nm,w in atop: print(f"  V: {nm} w={w:.2f}")
        print("-"*64)

def _gs_discover(title: str, all_matches: list):
    if SCORER_MODE not in ("DISCOVER","HYBRID"): return
    picks = []
    for m in all_matches or []:
        h,a = _gs_pick_names(m)
        if not (h and a): continue
        top = max(_gs_top_players(h)[:1]+_gs_top_players(a)[:1], key=lambda x:x[1], default=None)
        if top and top[1]>=SCORER_MIN_W:
            picks.append(m)
    picks = picks[:SCORER_EXTRA_MAX]
    if picks:
        _gs_block(title+" – DISCOVER", picks)

# ===================== AUTO CLOSER (merged) =====================
import time as _t

_SCORE_RE = re.compile(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$")

def _parse_ft_score(row: dict) -> tuple|None:
    sc = row.get("final_score") or row.get("score") or ""
    m = _SCORE_RE.match(str(sc))
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None

def _infer_over_threshold(market: str) -> float|None:
    if not isinstance(market, str):
        return None
    m = re.search(r"over\s+(\d+(?:\.\d+)?)", market, flags=re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _already_closed(daily_rows:list, tip_id:str) -> bool:
    for r in daily_rows:
        if r.get("event") == "tip_closed" and r.get("tip_id") == tip_id:
            return True
    return False

def _load_daily() -> list:
    from datetime import date
    p = LOG_DIR / f"{date.today().isoformat()}.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _append_jsonl(path:str, row:dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _now_iso():
    from datetime import datetime as _dt_
    return _dt_.now().strftime("%Y-%m-%d %H:%M:%S")

def _close_event_from(base_tip:dict, result:bool|None) -> dict:
    ev = {
        "event": "tip_closed",
        "date": _now_iso(),
        "tip_id": base_tip.get("tip_id"),
        "sport": base_tip.get("sport", "soccer"),
        "league": base_tip.get("league"),
        "home": base_tip.get("home"),
        "away": base_tip.get("away"),
        "market": base_tip.get("market"),
        "status": "finished",
        "closed_ts": int(_t.time())
    }
    if isinstance(result, bool):
        ev["won"] = result
        ev["result"] = "won" if result else "lost"
    return ev

def _decide_outcome(base_tip:dict, state_row:dict) -> bool|None:
    # Only Over markets for now + BTTS basic
    market = base_tip.get("market") or state_row.get("market","")
    thr = _infer_over_threshold(market)
    sc = _parse_ft_score(state_row)
    if sc:
        total = sc[0] + sc[1]
        if thr is not None:
            return total > thr
        # BTTS
        if "btts" in str(market).lower():
            return (sc[0] >= 1 and sc[1] >= 1)
    return None

def close_pending_tips(max_to_close: int = 50, verbose: bool = True) -> int:
    """
    Close today's pending tips by checking the SofaScore event status + final score.

    - Reads today's daily log: logs/YYYY-MM-DD.jsonl (via _load_daily()).
    - Finds rows with event=="new_tip" and status=="pending" that are not yet closed.
    - For each candidate, fetches SofaScore event result (status + FT score).
    - If finished, appends a "tip_closed" row to the SAME daily log file.
    """
    def _parse_ft(score_str):
        # Accept "2-1", "2:1", "2 – 1" etc.
        if not score_str:
            return None
        m = re.search(r"(\d+)\s*[-:–]\s*(\d+)", str(score_str))
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def _market_threshold(market_str):
        s = str(market_str or "").lower().strip()
        # Over 1.5 / Over 2.5 / Over 5.5 / Over 2.0 etc.
        m = re.search(r"over\s*(\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1))
        return None

    def _decide_outcome(market, ft_score):
        # returns: (result: "won"/"lost"/"void"/None, won_bool_or_none)
        if not ft_score:
            return None, None
        a, b = ft_score
        total = a + b
        ms = str(market or "").lower()

        # BTTS
        if "btts" in ms or "gg" in ms:
            won = (a >= 1 and b >= 1)
            return ("won" if won else "lost"), won

        thr = _market_threshold(market)
        if thr is None:
            return None, None

        # Handle Over 2.0 push
        if abs(thr - 2.0) < 1e-9:
            if total > 2:
                return "won", True
            if total == 2:
                return "void", None
            return "lost", False

        won = (total > thr)
        return ("won" if won else "lost"), won

    try:
        daily = _load_daily()
    except Exception:
        daily = []

    # build "already closed" set by tip_id
    closed_tip_ids = set()
    for row in daily:
        if isinstance(row, dict) and row.get("event") == "tip_closed":
            tid = row.get("tip_id")
            if tid:
                closed_tip_ids.add(tid)

    pending = []
    for row in daily:
        if not isinstance(row, dict):
            continue
        if row.get("event") != "new_tip":
            continue
        if row.get("status") not in (None, "", "pending"):
            continue
        tid = row.get("tip_id")
        if tid and tid in closed_tip_ids:
            continue
        if not row.get("event_id"):
            continue
        pending.append(row)

    if verbose:
        print(f"[CLOSE] pending candidates: {len(pending)} (already closed: {len(closed_tip_ids)})")

    checked = 0
    newly_closed = 0

    for tip in pending[:max_to_close]:
        checked += 1
        eid = tip.get("event_id")
        tid = tip.get("tip_id")
        try:
            res = _sofa_fetch_event_result(eid)
        except Exception as e:
            if verbose:
                print("[CLOSE] fetch error:", eid, e)
            continue

        if not isinstance(res, dict):
            continue

        st = str(res.get("status") or "").lower()
        ft = _parse_ft(res.get("score"))
        if st != "finished" or not ft:
            # Not finished yet (or no FT score)
            continue

        result, won = _decide_outcome(tip.get("market"), ft)
        if result is None:
            continue

        ev = {
            "event": "tip_closed",
            "date": _now_iso(),
            "tip_id": tid,
            "sport": tip.get("sport", "soccer"),
            "event_id": eid,
            "start_ts": tip.get("start_ts"),
            "league": tip.get("league"),
            "home": tip.get("home"),
            "away": tip.get("away"),
            "market": tip.get("market"),
            "ft_score": f"{ft[0]}-{ft[1]}",
            "status": "finished",
            "result": result,
        }
        if isinstance(won, bool):
            ev["won"] = won

        try:
            _append_jsonl(str(_daily_path()), ev)
            newly_closed += 1
            if verbose:
                print(f"[CLOSE] CLOSED {tid} eid={eid} {ev['market']} -> {result} ({ev['ft_score']})")
        except Exception as e:
            if verbose:
                print("[CLOSE] write error:", e)

    if verbose:
        print(f"[CLOSE] checked={checked} newly_closed={newly_closed}")
    return newly_closed
def _stats_append_txt(lines):
    try:
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(STATS_LOG_TXT, "a", encoding="utf-8") as f:
            for ln in lines:
                f.write(f"[{ts}] {ln}\n")
    except Exception:
        pass

def _stats_append_json(block):
    try:
        block = dict(block)
        block.setdefault("timestamp", _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # betöltés (ha nem lista, reseteljük)
        arr = []
        try:
            with open(STATS_LOG_JSON, "r", encoding="utf-8") as f:
                arr = _json.load(f)
                if not isinstance(arr, list):
                    arr = []
        except Exception:
            arr = []
        arr.append(block)
        with open(STATS_LOG_JSON, "w", encoding="utf-8") as f:
            _json.dump(arr, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

ENABLE_TENNIS=False  # True = tenisz tippek engedélyezve, False = kikapcsolva
ENABLE_FOOTBALL=True  # True = foci tippek engedélyezve

# ===================== JSONL LOGGING (two-way sync ready) =====================
try:
    import uuid, time, json, os, sys
except Exception:
    pass
# === RESULT_EVAL_SHIM (Singles+Slips summary only) ===
from typing import List, Tuple, Dict, Any

# ============ CLOSER UTILS ============
def _norm_team(name: str) -> str:
    if not name: 
        return ""
    s = str(name).lower().strip()
    # remove extra spaces and common punctuation
    s = re.sub(r'[^a-z0-9áéíóöőúüű\s-]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def _key_from_tip(t: dict) -> tuple:
    return (_norm_team(t.get("home","")), _norm_team(t.get("away","")), t.get("market",""))

def _parse_score(score: str):
    if not score: 
        return None
    m = re.match(r'^\s*(\d+)\s*[-:]\s*(\d+)\s*$', str(score))
    if not m: 
        return None
    return int(m.group(1)), int(m.group(2))
import re

_LAST_TIPS: List[dict] = []

def _sg(d:dict, *ks, default=None):
    for k in ks:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def _parse_score(m:dict)->Tuple[int,int,int]:
    sc = _sg(m, "score","állás","allas","score_str", default="")
    if isinstance(sc, str):
        mm = re.search(r"(\d+)\s*[-–:]\s*(\d+)", sc)
        if mm:
            h, a = int(mm.group(1)), int(mm.group(2))
            return h+a, h, a
    h = int(_sg(m,"home",default=0) or 0)
    a = int(_sg(m,"away",default=0) or 0)
    return h+a, h, a

def _market_kind(market:str):
    if not isinstance(market, str): return None
    m = re.search(r"over\s*(\d+(?:\.\d+)?)", market, re.I)
    if m: return ("OVER", float(m.group(1)))
    if re.search(r"BTTS", market, re.I): return ("BTTS", None)
    return None

def _is_finished(m:dict)->bool:
    st = str(_sg(m,"status","minute","time_str","status_str", default="")).upper()
    if "FT" in st or "FULL" in st: return True
    mm = re.search(r"(\d+)", st)
    if mm and int(mm.group(1)) >= 90: return True
    if re.search(r"90\s*\+", st): return True
    return False

def _eval_tip(t:dict)->Tuple[str,str]:
    mk = str(_sg(t,"market","piac","tip","label", default=""))
    kind = _market_kind(mk)
    if not kind: return "SKIP", "Ismeretlen piac"
    total, h, a = _parse_score(t)
    if kind[0] == "OVER":
        thr = kind[1] or 0.0
        if _is_finished(t):
            return ("W" if total > thr-1e-9 else "L", f"FT {h}-{a} | need>{thr}")
        else:
            return "P", f"Live {h}-{a} | need>{thr}"
    if kind[0] == "BTTS":
        if _is_finished(t):
            return ("W" if (h>=1 and a>=1) else "L", f"FT {h}-{a} | need 1-1+")
        else:
            return "P", f"Live {h}-{a} | need 1-1+"
    return "SKIP","Piac nem támogatott"

_SLIP_KEYS = ["ticket_id","slip_id","szelveny_id","parlay_id","combo_id","group_id","coupon_id"]

def _slip_id(t:dict):
    for k in _SLIP_KEYS:
        if k in t: return t[k]
    return None

def _split_singles_slips(tips:List[dict]):
    singles, slips = [], {}
    for t in tips:
        sid = _slip_id(t)
        if sid is None:
            singles.append(t)
        else:
            slips.setdefault(sid, []).append(t)
    return singles, slips

def _box(title:str, lines:List[str])->str:
    width = max([len(title)+2] + [len(s) for s in lines] + [40])
    top = "┌" + "─"*width + "┐"
    bot = "└" + "─"*width + "┘"
    out = [top, "│ " + title.ljust(width-1) + "│"]
    for s in lines:
        out.append("│ " + s.ljust(width-1) + "│")
    out.append(bot)
    return "\n".join(out)

def stats_capture_tips(tips:List[dict]):
    global _LAST_TIPS
    try: _LAST_TIPS = list(tips or [])
    except Exception: _LAST_TIPS = []
    return tips

def print_tip_outcome_summary():
    tips = _LAST_TIPS or []
    singles, slips = _split_singles_slips(tips)

    W=L=P=0
    for t in singles:
        r,_ = _eval_tip(t)
        if r=='W': W+=1
        elif r=='L': L+=1
        elif r=='P': P+=1
    closed = W+L
    rate = (W/closed*100.0) if closed>0 else 0.0
    s_lines = [f"Összes: {len(singles)}   Lezárt: {closed}   Folyamatban: {P}",
               f"Jött: {W}   Nem jött: {L}   Találati arány: {rate:.1f}%"]

    SW=SL=SP=0
    for sid, legs in slips.items():
        leg_res = []
        for t in legs:
            r,_ = _eval_tip(t); leg_res.append(r)
        if all(r=='W' for r in leg_res): SW+=1
        elif any(r=='P' for r in leg_res): SP+=1
        elif any(r=='L' for r in leg_res): SL+=1
    sc_closed = SW+SL
    sc_rate = (SW/sc_closed*100.0) if sc_closed>0 else 0.0
    c_lines = [f"Szelvények: {len(slips)}   Lezárt: {sc_closed}   Folyamatban: {SP}",
               f"NYERT: {SW}   NEM NYERT: {SL}   Találati arány: {sc_rate:.1f}%"]

# [removed old panel: TIP-ÖSSZEGZŐ – SZINGLI (SOCCER)]
# [removed old panel: TIP-ÖSSZEGZŐ – SZELVÉNY (SOCCER)]
# === END RESULT_EVAL_SHIM ===

DEVICE_TAG = os.environ.get("TIP_DEVICE", "PHONE" if "android" in (sys.platform.lower()) else "PC")

def _log_path():
    fn = time.strftime("%Y-%m-%d") + ".jsonl"
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, fn)


# === W/L/ROI összesítő (SOCCER) ============================================
def _tip_odds_for_roi(t: dict) -> float:
    """ROI számoláshoz tipp-odds: placed_odds > odds > est_odds > 1.0"""
    try:
        for k in ("placed_odds", "odds", "est_odds"):
            v = t.get(k)
            if v is None:
                continue
            return float(v)
    except Exception:
        pass
    return 1.0

# --- Status normalizer for legacy/persisted tips -----------------------------
def _norm_status_for_stats(s):
    """
    Normalize various status encodings to one of: 'WIN', 'LOSE', 'PENDING'.
    Accepts symbols like check/times, strings (W,L,WIN,LOSE,OK,KO), booleans, and 1/0.
    """
    if s is True:
        return "WIN"
    if s is False:
        return "LOSE"
    if isinstance(s, (int, float)):
        if s == 1:
            return "WIN"
        if s == 0:
            return "LOSE"
    if isinstance(s, str):
        t = s.strip().upper()
        if t in ("WIN","W","OK","TRUE","YES","✔","DONE","GREEN"):
            return "WIN"
        if t in ("LOSE","L","KO","FALSE","NO","✖","RED"):
            return "LOSE"
    return "PENDING"
# -----------------------------------------------------------------------------

def compute_global_stats() -> dict:
    """Összesítő stat perzisztens SOCCER állapotból (stake=1/tipp)."""
    st = s_state()
    tips = list((st.get("tips") or {}).values())
    total = len(tips)
    closed = wins = losses = 0
    pnl = 0.0
    per_market = {}

    def upd_mkt(m, win, profit):
        m = (m or "?")
        d = per_market.setdefault(m, {"W": 0, "L": 0, "closed": 0, "pnl": 0.0})
        if win is True:
            d["W"] += 1
        elif win is False:
            d["L"] += 1
        d["closed"] += 1
        d["pnl"] += float(profit)

    for t in tips:
        raw_status = t.get("status")
        norm = _norm_status_for_stats(raw_status)
        if norm == "PENDING":
            continue
        closed += 1
        odds = max(1.0, _tip_odds_for_roi(t))
        if norm == "WIN":
            wins += 1
            profit = (odds - 1.0)
            pnl += profit
            upd_mkt(t.get("market", t.get("piac") or "?"), True, profit)
        else:
            losses += 1
            profit = -1.0
            pnl += profit
            upd_mkt(t.get("market", t.get("piac") or "?"), False, profit)

    hit = (wins / closed * 100.0) if closed else 0.0
    roi = (pnl / closed) if closed else 0.0
    return {
        "total_seen": total,
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "hit_rate_pct": round(hit, 1),
        "pnl_units": round(pnl, 2),
        "roi_per_tip": round(roi, 3),
        "per_market": per_market,
    }

def print_global_stats_box():
    s = compute_global_stats()
    lines = [
        f"Összes tipp (eddig): {s['total_seen']}",
        f"Lezárt: {s['closed']}   Jött: {s['wins']}   Nem jött: {s['losses']}   Találati arány: {s['hit_rate_pct']}%",
        f"PnL (egység): {s['pnl_units']}   ROI/tipp: {s['roi_per_tip']}",
    ]
    try:
        pm = s.get("per_market", {})
        best = sorted(pm.items(), key=lambda kv: kv[1]['pnl'], reverse=True)[:3]
        for m, d in best:
            lines.append(f"  • {m}: W{d['W']}/L{d['L']}  PnL={round(d['pnl'], 2)}")
    except Exception:
        pass
# [removed old panel: GLOBÁL STAT – W/L/ROI (SOCCER)]
# === END W/L/ROI ============================================================


def merge_unique_events(paths=None):
    # read all jsonl logs and return unique list by event_id (fallback composite key)
    if paths is None:
        paths = [LOG_DIR]
    seen = set(); out = []
    for pdir in paths:
        if not os.path.isdir(pdir): 
            continue
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith(".jsonl"):
                continue
            try:
                for line in open(os.path.join(pdir, fn), "r", encoding="utf-8", errors="ignore"):
                    line=line.strip()
                    if not line:
                        continue
                    try:
                        ev=json.loads(line)
                    except:
                        continue
                    key = ev.get("event_id") or (ev.get("ts"), ev.get("match_id"), ev.get("market"), ev.get("minute"), ev.get("score"))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(ev)
            except:
                continue
    return out
# ============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIFIED TIPS – Soccer + Tennis with BRUTAL AI Turbo
- Standard library only (Termux/Pydroid oké)
- Live sources (best-effort, public): SofaScore + ESPN
- PRE-szeló with NL time
- Online logistic models (per market) that learn from your results
- Ensemble: base heuristic p + ML p -> calibrated final p
- 5-run EMA stats

Run examples:
  python3 tenisz_foci_pre_brutal.py --sport both --once
  python3 tenisz_foci_pre_brutal.py --sport soccer --interval 60
"""

import sys, time, json, re, math, random, traceback, os, hashlib, datetime as dt
# ===== PRE-TUTI CONFIG =====
# Idő- és mennyiségi limitek + debug
PRE_TUTI_TIME_BUDGET = 8.0   # másodperc: ennyi ideje van a PRE-TUTI résznek összesen
PRE_TUTI_MAX_SCAN = 1200    # max. ennyi előmeccses jelöltet nézzen át részletesen
SCRAPER_TIMEOUT      = 4.5   # másodperc / kérés
PRE_TUTI_DEBUG       = True  # True-ra téve részletes log a kör végén
# --- lazább elfogadási küszöbök ---
PRE_TUTI_THRESH = {
    "O1_5": 0.70,   # Over 1.5
    "BTTS": 0.70,   # Mindkét csapat lő gólt
    "O2_5": 0.70    # Over 2.5 (ha nagyon erős jel)
}

PRE_TUTI_MIN_ODDS = {
    "O1_5": 1.20,
    "BTTS": 1.50,
    "O2_5": 1.30
}

# Ha nincs „tuti” a szigorú szűrővel, engedjünk egy „soft” fallback-et.
PRE_TUTI_ACCEPT_SOFT_FALLBACK = True
PRE_TUTI_SOFT_FALLBACK_PROB   = 0.62    # ha legalább ennyi és a piac O1.5 vagy BTTS
PRE_TUTI_PRINT_TOPK           = 5       # mindig írd ki a top 5 jelöltet
# ---- időjárás (wttr.in) ----
def _scrape_weather(home, away, kickoff_ts=None, loc_hint=None) -> float:
    """
    wttr.in i1 formátum:
      0 = jó / nincs gond, 1 = közepes, 2 = rossz idő
    Visszatér egy [0.0..1.0] közötti szorzóval (1.0 = nincs levonás).
    """
    if not (ENABLE_SCRAPERS and SCRAPER_WEATHER):
        return 0.0

    try:
        q = (loc_hint or home).split(" ")[0]
        url = f"https://wttr.in/{q}?format=i1"
        raw = _http_get(url, timeout=SCRAPER_TIMEOUT)

        s = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        s = s.strip()
        code = int(s) if s.isdigit() else 0

        # 0 jó, 1 közepes, 2 rossz – igény szerint szigorítható
        return 1.0 if code == 0 else (0.7 if code == 1 else 0.3)

    except Exception as e:
        if PRE_TUTI_DEBUG:
            print("[PRE_TUTI][WEATHER] hiba:", e)
        # hiba esetén se álljon meg a pontszámítás
        return 0.5

## === PRE_TUTI CONFIG START ===
# PRE-TUTI – előmeccses „tuti” jelölt kereső (csak foci)
PRE_TUTI_ENABLE       = True   # főkapcsoló
PRE_TUTI_TIME_BUDGET  = 8.0    # másodperc / kör
PRE_TUTI_MAX_SCAN     = 2400    # max. vizsgált előmeccses jelölt / kör
PRE_TUTI_DEBUG        = False  # részletes log

# Küszöbök (pontszám 0..10):
PRE_TUTI_MIN_SCORE    = 7.2    # legalább ennyi pont kell
PRE_TUTI_MIN_ODDS     = 1.28   # túl alacsony szorzót dobjuk
PRE_TUTI_MAX_ODDS     = 1.95           # túl magasat is

# Súlyok a pontozóhoz:
W_FORM        = 2.0    # forma, utolsó 5 meccs gólátlag, gólkülönbség
W_H2H         = 1.0    # egymás elleni közelmúlt
W_LEAGUE      = 0.7    # bajnokság gólátlaga
W_LINEUP      = 1.5    # hiányzók / kulcsjátékosok (heurisztika: csapatnév + hírszavak)
W_NEWS        = 1.0    # hírek (suspension/injury/coach out/etc.) – heurisztikus
W_WEATHER     = 0.6    # időjárás (eső/szél → óvatosabb over)
W_REFEREE     = 0.8    # bíró lap/gól átlag – ha elérhető a sorban (heurisztika)
W_MARKET      = 1.2    # odds-mozgás (ha látjuk a sorban)

## === PRE_TUTI CONFIG END ===
# ---- sérültek / hiányzók (gyors keresés) ----
def _scrape_injuries(home, away):
    if not (ENABLE_SCRAPERS and SCRAPER_INJURIES):
        return 0.0
    try:
        # nagyon light: csak megnézzük van-e „injuries” sztring a csapat nevére keresve
        base = "https://duckduckgo.com/html/?q="
        for team in (home, away):
            url = base + _u.quote_plus(f"{team} injuries")
            raw = _http_get(url, timeout=SCRAPER_TIMEOUT)
            txt = raw.decode("utf-8","ignore").lower()
            if "injur" in txt or "sidelined" in txt:
                return W_INJURY_KEY
    except Exception:
        return 0.0
    return 0.0

# ---- bírói statok ----
def _scrape_referee(ref_name):
    if not (ENABLE_SCRAPERS and SCRAPER_REFEREE):
        return 0.0
    if not ref_name:
        return 0.0
    try:
        slug = ref_name.lower().replace(" ", "-")
        url = f"https://www.worldfootball.net/referee_summary/{slug}/"
        raw = _http_get(url, timeout=SCRAPER_TIMEOUT)
        tx = raw.decode("utf-8","ignore")
        # durva regexek: 11-esek és sárgák/ vörösek (ha nincs, 0)
        pens = 0
        ypc = 0.0
        m = re.search(r"Penalty\s+(\d+)", tx, re.I)
        if m:
            pens = int(m.group(1))
        m = re.search(r"yellow cards.*?(\d+\.\d+)", tx, re.I|re.S)
        if m:
            ypc = float(m.group(1))
        bonus = 0.0
        if pens >= 5: bonus += W_REF_PENALTY
        if ypc >= 4.5: bonus += W_REF_YELLOW
        return bonus
    except Exception:
        return 0.0

def _pretuti_load_state(path):
    try:
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _pretuti_save_state(path, d):
    try:
        with open(path,"w",encoding="utf-8") as f:
            json.dump(d,f,ensure_ascii=False,indent=2)
    except Exception:
        pass

import urllib.request as _u
def _http_get(url, headers=None, timeout=None):
    if headers is None:
        headers = {}
    if "User-Agent" not in headers:
        headers["User-Agent"] = random.choice([
            "Mozilla/5.0 (Linux; Android 12) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114 Mobile",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/114 Safari/537.36",
        ])
    req = _u.Request(url, headers=headers)
    with _u.urlopen(req, timeout=(timeout or SCRAPER_TIMEOUT)) as r:
        data = r.read()
        # próbáljuk gunzip-et
        try:
            if len(data) >= 2 and data[:2] == b'\x1f\x8b':
                data = gzip.decompress(data)
        except Exception:
            pass
        return data

# Globális be/ki
ENABLE_PRE_TUTI   = True      # ← ha nem kell a napi „tuti”, állítsd False-ra
PRE_TUTI_TOP_N    = 2         # hányat írjon ki (1–3 javasolt)
PRE_TUTI_MIN_ODDS = 1.20      # alsó szorzó-határ
PRE_TUTI_MAX_ODDS = 2.10      # felső szorzó-határ

# Scraper összkapcsoló + alkapcsolók
ENABLE_SCRAPERS   = True
SCRAPER_WEATHER   = True
SCRAPER_INJURIES  = True
SCRAPER_REFEREE   = True

# Scraper súlyok (± pont a re-rangsoroláshoz)
W_WEATHER_BAD   = -0.12   # erős eső/szél -> kevesebb gól
W_WEATHER_GOOD  = +0.04
W_INJURY_KEY    = -0.08   # kulcsember hiánya
W_REF_YELLOW    = +0.03   # sok sárga/pöri -> több hiba, több gól esély
W_REF_PENALTY   = +0.05   # 11-es hajlam

# PRE-TUTI napló: ide teszi a legutóbbi ajánlásokat, hogy ne ismételjen
PRE_TUTI_STATE = "_pretuti_state.json"
# ===== END PRE-TUTI CONFIG =====
from typing import Optional, Dict, Any, List, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ---------------- Timezone: Amsterdam ----------------

# ==== ULTRA+ kapcsolók (bíró + liga-freq + Monte-Carlo) ====
ENABLE_REFEREE_TURBO   = True    # játékvezető over/under hajlam
ENABLE_LIVE_LEAGUE_FREQ= True    # élő/szűk-esemény liga over-frekvencia
ENABLE_MC_ENSEMBLE     = True    # Monte-Carlo tilt/enged
ULTRA_LOG_DEBUG        = False   # részletes ULTRA/ULTRA+ log

# súlyok (finom határok, ne borítsa fel az alaprendszert)
U_REFEREE      = 0.05   # ±5% a bírótól
U_LIVE_FREQ    = 0.05   # ±5% a liga aktuális „gólosságától”
U_MC_ALPHA     = 0.30   # MC p-keverési arány a végső valószínűségbe (30%)

# Monte-Carlo részletek
MC_SIMS        = 240    # szimulációk száma
MC_SEG_MIN     = 10     # perc/segment
MC_MAX_MIN     = 95     # max. percig modellezünk

# ==== ULTRA BRUTÁL focis turbó – kapcsolók ====
ENABLE_ULTRA_SOCCER = True       # főkapcsoló
ULTRA_SCRAPE_EXTRA  = True       # óvatos kiegészítő scrape (ha hiba, némán átugrik)
ULTRA_LOG_DEBUG     = True      # részletes log az ULTRA-ról

# súlyok – finoman, hogy ne borítsa fel az alaprendszert
U_LEAGUE_BASE   = 0.04   # liga-gólátlag hatás (±4%)
U_FORM_MOMENTUM = 0.05   # utóbbi 5 meccs góltempó (±5%)
U_SCORE_SHAPE   = 0.06   # kiütés / busz-parkolás helyzet (±6%)
U_MINUTE_MKT    = 0.05   # perc + piac (±5%)
U_RED_CARD      = 0.05   # piros lap hatás (±5%)

# korlátok
U_MUL_MIN = 0.85         # össz-szorzó alsó korlát
U_MUL_MAX = 1.15         # össz-szorzó felső korlát

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

TZ_NL = ZoneInfo("Europe/Amsterdam") if ZoneInfo else None
def now_nl() -> dt.datetime:
    return dt.datetime.now(TZ_NL) if TZ_NL else dt.datetime.now()

def now_eu() -> dt.datetime:
    return dt.datetime.now()

# ---------------- Config ----------------
RUN_EVERY_SEC = 60
DEBUG = True

USER_AGENTS = [
    "Mozilla/5.0 (Linux; Android 12; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17 Safari/605.1.15",
]
HTTP_TIMEOUT = 6
HTTP_RETRIES = 1

def dbg(msg: str) -> None:
    if DEBUG:
        print("[DEBUG]", msg)

def http_get(url: str, headers: Optional[Dict[str,str]] = None) -> Optional[bytes]:
    base = {"User-Agent": random.choice(USER_AGENTS), "Accept": "application/json,text/html;q=0.9,*/*;q=0.8", "Accept-Encoding": "gzip, deflate"}
    if headers: base.update(headers)
    for attempt in range(HTTP_RETRIES + 1):
        try:
            req = Request(url, headers=base)
            with urlopen(req, timeout=HTTP_TIMEOUT) as r:
                data = r.read()
                if len(data) >= 2 and data[:2] == b"\x1f\x8b":
                    try:
                        import gzip; data = gzip.decompress(data)
                    except Exception:
                        pass
                return data
        except (HTTPError, URLError) as e:
            if attempt >= HTTP_RETRIES:
                log_error(os.path.join(LOG_DIR, "global_errors.log"), f"HTTP GET {url} failed: {e}")
            time.sleep(0.8 * (attempt + 1))
        except Exception as e:
            if attempt >= HTTP_RETRIES:
                log_error(os.path.join(LOG_DIR, "global_errors.log"), f"HTTP GET {url} unexpected: {e}\n{traceback.format_exc()}")
            time.sleep(0.8 * (attempt + 1))
    return None

def log_error(path: str, msg: str) -> None:
    ts = now_eu().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

def save_json(path: str, data: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_error(os.path.join(LOG_DIR, "global_errors.log"), f"save_json({path}) failed: {e}")

def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path): return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def sigmoid(x: float) -> float:
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# ---------------- 5-run stats ----------------
STATS_PATH = os.path.join(LOG_DIR, "stats_state.json")
STATS_BATCH = 5
def stats_load() -> Dict[str, Any]:
    return load_json(STATS_PATH, {"runs": 0, "soccer": {"tips": 0, "avg_prob": 0.0, "avg_odds": 0.0},
                                  "tennis": {"tips": 0, "avg_prob": 0.0, "avg_odds": 0.0}})
def stats_update(section: str, tips: List[Dict[str, Any]]):
    st = stats_load()
    st["runs"] = int(st.get("runs", 0)) + 1
    agg = st.get(section, {"tips": 0, "avg_prob": 0.0, "avg_odds": 0.0})
    if tips:
        probs = sum(t.get("prob", 0.0) for t in tips) / max(1, len(tips))
        odds  = sum(float(t.get("est_odds") or 0.0) for t in tips) / max(1, len(tips))
        a = 0.5
        agg["avg_prob"] = a * probs + (1 - a) * agg.get("avg_prob", 0.0)
        agg["avg_odds"] = a * odds  + (1 - a) * agg.get("avg_odds", 0.0)
        agg["tips"] = int(agg.get("tips", 0)) + len(tips)
    st[section] = agg
    save_json(STATS_PATH, st)

def stats_maybe_print():
    st = stats_load()
    if int(st.get("runs", 0)) % STATS_BATCH != 0:
        return
    print("\n" + "="*72)
    # [removed old 5-run stats block]

# ---------------- Online Logistic Model (BRUTAL TURBO) ----------------
class OnlineLogit:
    """
    Egyszerű online logisztikus regresszió sztochasztikus gradienssel.
    features: dict[str,float] -> w: dict[str,float]
    p = sigmoid(sum_i w_i * x_i + b)
    """
    def __init__(self, lr: float = 0.08, l2: float = 0.0005):
        self.w: Dict[str, float] = {}
        self.b: float = 0.0
        self.lr = lr
        self.l2 = l2
        self.n = 0  # update count

    def predict(self, feats: Dict[str, float]) -> float:
        s = self.b
        for k, v in feats.items():
            s += self.w.get(k, 0.0) * v
        return sigmoid(s)

    def update(self, feats: Dict[str, float], y: int) -> None:
        # y in {0,1}
        p = self.predict(feats)
        err = p - float(y)  # dLoss/ds for logloss
        # L2 + gradient
        for k, v in feats.items():
            w = self.w.get(k, 0.0)
            w -= self.lr * (err * v + self.l2 * w)
            self.w[k] = w
        self.b -= self.lr * err
        self.n += 1

    def to_json(self) -> Dict[str, Any]:
        return {"w": self.w, "b": self.b, "lr": self.lr, "l2": self.l2, "n": self.n}

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "OnlineLogit":
        m = cls(lr=float(d.get("lr", 0.08)), l2=float(d.get("l2", 0.0005)))
        m.w = {str(k): float(v) for k, v in (d.get("w") or {}).items()}
        m.b = float(d.get("b", 0.0))
        m.n = int(d.get("n", 0))
        return m

def load_models(path: str) -> Dict[str, OnlineLogit]:
    raw = load_json(path, {})
    out: Dict[str, OnlineLogit] = {}
    for k, v in raw.items():
        try:
            out[k] = OnlineLogit.from_json(v)
        except Exception:
            out[k] = OnlineLogit()
    return out

def save_models(path: str, models: Dict[str, OnlineLogit]) -> None:
    data = {k: m.to_json() for k, m in models.items()}
    save_json(path, data)

def blend(base_p: float, ml_p: float, alpha: float = 0.6) -> float:
    # final = alpha*base + (1-alpha)*ml; clamp to [0.02,0.98]
    x = alpha * base_p + (1 - alpha) * ml_p
    return max(0.02, min(0.98, x))
try:
    _apply_tuning(globals())
except Exception:
    pass

# ===================================================================
# ========================== SOCCER MODULE ==========================
# ===================================================================
SOCCER_STATE = os.path.join(LOG_DIR, "soccer_state.json")
SOCCER_LOG   = os.path.join(LOG_DIR, "soccer_tips_log.jsonl")
SOCCER_ERR   = os.path.join(LOG_DIR, "soccer_errors.log")
SOCCER_MODELS_PATH = "soccer_ai_models.json"  # per-market OnlineLogit

SOCCER_USE_ODDS_WINDOW = True
SOCCER_ODDS_WINDOW = (1.35, 3.20)  # kicsit nyitottabb
SOCCER_PROB_MIN = 0.90             # relaxed from 0.82
SOCCER_MAX_TIPS = 5
SOCCER_MARKET_CAP = 1
SOCCER_LATE_MIN = 72

SOCCER_MIN_ODDS = {"O1_5": 1.20, "O2_5": 1.50, "BTTS": 1.60, "LATE": 1.30}
SOCCER_MAX_ODDS = {"O1_5": 2.40, "O2_5": 3.60, "BTTS": 2.50, "LATE": 2.40}
SOCCER_THRESH = {
    "O1_5": 0.74,
    "O2_5": 0.68,
    "BTTS": 0.70,
    "LATE": 0.70
}
# ===================== EXTERNAL TUNING (tuning.json) =====================
# Default path (Termux): /storage/emulated/0/zetbet/tuning.json
# Override path via env: TIP_TUNING_JSON
#def _load_tuning():
 #   try:
  #      import os, json
   #     p = os.environ.get("TIP_TUNING_JSON", "/storage/emulated/0/zetbet/tuning.json")
    #    if not os.path.exists(p):
     #       return {}
      #  with open(p, "r", encoding="utf-8") as f:
       #     d = json.load(f)
        #return d if isinstance(d, dict) else {}
   # except Exception:
    #    return {}

#def _apply_tuning(globals_dict):
 #   d = _load_tuning()
  #  if not d:
   #     return

    allow = {
        "SOCCER_PROB_MIN",
        "SOCCER_ODDS_WINDOW",
        "SOCCER_THRESH",
        "SOCCER_MIN_ODDS",
        "SOCCER_MAX_ODDS",
        "SOCCER_MAX_TIPS",
        "SOCCER_MARKET_CAP",
        "SOCCER_LATE_MIN",
        "AI_TURBO",
        "AI_MAX_BOOST",
        "AI_MIN_SAMPLES",
        "AI_EMA_ALPHA",
        "PRE_ENABLED",
        "PRE_MAX_MATCHES",
        "PRE_WINDOW_MIN_FROM",
        "PRE_WINDOW_MIN_TO",
    }

    for k, v in d.items():
        if k not in allow:
            continue
        try:
            globals_dict[k] = v
        except Exception:
            pass

    print("[TUNING] tuning.json applied")
# ========================================================================
# HT/FT korai ablak
ENABLE_HTFT = True
HTFT_START_MIN = 0
HTFT_END_MIN   = 20
HTFT_MIN_PROB = 0.06
HTFT_MAX_GLOBAL = 5
HTFT_MAX_PER_LEAGUE = 2
HTFT_MAX_GOALS = 6
HTFT_SKIP_SECOND_HALF = True
HTFT_SKIP_IF_BIG_LEAD_AFTER_MIN = 25
HTFT_BIG_LEAD_MARGIN = 2

# PRE
PRE_ENABLED = True
PRE_WINDOW_MIN_FROM = 15
PRE_WINDOW_MIN_TO   = 720
PRE_MAX_MATCHES     = 3

# PRE Turbo model (deterministic variety)
def _pre_hash_bias(*items) -> float:
    s = "|".join(str(x) for x in items)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    z = int(h[:8], 16) / float(16**8 - 1)
    return (z - 0.5) * 0.08  # ±0.04

def _pre_league_tier(league: str) -> int:
    name = (league or "").lower()
    tier1 = ["premier league","la liga","serie a","bundesliga","ligue 1","eredivisie","primeira liga","champions league","europa league"]
    tier2 = ["championship","segunda","2. bundesliga","ligue 2","eredivisie 2","serie b","mls","j1 league","süper lig","ekstraklasa"]
    if any(k in name for k in tier1): return 1
    if any(k in name for k in tier2): return 2
    return 3

def pre_prob_model(home: str, away: str, league: str, mins_to_ko: int):
    tier = _pre_league_tier(league)
    if tier == 1:
        base_o15, base_btts, base_o25 = 0.60, 0.56, 0.54; span = 0.08
    elif tier == 2:
        base_o15, base_btts, base_o25 = 0.59, 0.55, 0.53; span = 0.10
    else:
        base_o15, base_btts, base_o25 = 0.58, 0.54, 0.52; span = 0.12

    if mins_to_ko < 45:
        wind = -0.005 + (mins_to_ko/45.0)*0.01
    elif mins_to_ko <= 120:
        x = (mins_to_ko - 45) / 75.0
        wind = 0.02 * (1 - (2*x-1)**2)
    else:
        wind = -0.01 * ((mins_to_ko - 120) / 60.0)

    zb = _pre_hash_bias(home, away, league)
    z1 = zb; z2 = _pre_hash_bias(away, home, league) * 0.9; z3 = _pre_hash_bias(home, league, away) * 0.8

    def clamp01(x): return max(0.0, min(1.0, x))
    p15  = clamp01(base_o15 + wind + z1 * span)
    pbts = clamp01(base_btts + wind + z2 * span)
    p25  = clamp01(base_o25 + wind + z3 * span)

    arr = [p15, pbts, p25]
    if max(arr) - min(arr) < 0.03: p15 += 0.01; p25 -= 0.01

    lo, hi = 0.40, 0.74
    p15  = max(lo, min(hi, p15))
    pbts = max(lo, min(hi, pbts))
    p25  = max(lo, min(hi, p25))
    return p15, pbts, p25

# --- Data types ---
class SMatch:
    def __init__(self, source, league, home, away, minute, sh, sa, phase="1H", event_id=None, start_ts=None):
        self.source=source; self.league=league; self.home=home; self.away=away
        self.minute=minute; self.sh=sh; self.sa=sa; self.phase=phase
        self.event_id=event_id; self.start_ts=start_ts
    @property
    def tg(self): return self.sh + self.sa

# --- Fetchers ---
def extract_minute_phase_sofa(ev: dict) -> Tuple[int, str]:
    st = (ev.get("status") or {})
    desc = str(st.get("description") or "").lower()
    phase = "1H"
    if "second" in desc or "2nd" in desc or "2. félidő" in desc: phase = "2H"
    if "half" in desc and "time" in desc: phase = "HT"
    if "ended" in desc or "finished" in desc or "full time" in desc: phase = "FT"
    disp = str((ev.get("clock") or {}).get("displayValue") or "")
    m = re.search(r"(\d+)", disp)
    if m:
        minute = int(m.group(1))
        if phase == "2H" and minute <= 55: minute = min(90, 45 + minute)
        return (minute, phase)
    now_ts = int(time.time())
    start_ts = ev.get("startTimestamp") or 0
    cur_half_start = (ev.get("time") or {}).get("currentPeriodStartTimestamp") or 0
    minute = 0
    if cur_half_start:
        elapsed = max(0, now_ts - int(cur_half_start)); minute = elapsed // 60
        if "2" in desc or phase == "2H": minute = min(90, 45 + minute)
    elif start_ts:
        elapsed = max(0, now_ts - int(start_ts)); minute = min(95, elapsed // 60)
    if "half" in desc and "time" in desc: minute = 45; phase = "HT"
    return (int(minute), phase)

def fetch_soccer_sofa() -> List[SMatch]:
    out=[]
    raw=http_get("https://api.sofascore.com/api/v1/sport/football/events/live", headers={"Accept":"application/json"})
    if not raw: return out
    try:
        data=json.loads(raw.decode("utf-8","ignore"))
        for ev in data.get("events",[]):
            st=((ev.get("status") or {}).get("type") or "").lower()
            if st not in ("inprogress","live"): continue
            league=(ev.get("tournament") or {}).get("name") or "SofaScore"
            home=(ev.get("homeTeam") or {}).get("name") or "Home"
            away=(ev.get("awayTeam") or {}).get("name") or "Away"
            sh=int((ev.get("homeScore") or {}).get("current") or 0)
            sa=int((ev.get("awayScore") or {}).get("current") or 0)
            minute, phase = extract_minute_phase_sofa(ev)
            eid = ev.get("id")
            start_ts = ev.get("startTimestamp")
            out.append(SMatch("sofascore",league,home,away,minute,sh,sa,phase,event_id=eid,start_ts=start_ts))
        dbg(f"[SOCCER] Sofa live: {len(out)}")
    except Exception as e:
        log_error(SOCCER_ERR, f"sofa parse: {e}\n{traceback.format_exc()}")
    return out

def fetch_soccer_espn() -> List[SMatch]:
    out=[]
    raw=http_get("https://site.api.espn.com/apis/site/v2/sports/soccer/scoreboard", headers={"Accept":"application/json"})
    if not raw: return out
    try:
        data=json.loads(raw.decode("utf-8","ignore"))
        for ev in data.get("events", []):
            comp=(ev.get("competitions") or [{}])[0]
            status=(comp.get("status") or {}).get("type",{})
            st=(status.get("state") or "").upper()
            if st!="IN": continue
            comps=comp.get("competitors") or []
            home=away=""; sh=sa=0
            for c in comps:
                nm=(c.get("team") or {}).get("name","")
                sc=int(c.get("score") or 0)
                if c.get("homeAway")=="home":
                    home,sh=nm,sc
                else:
                    away,sa=nm,sc
            league=((ev.get("league") or {}).get("name")) or "ESPN"
            disp=str(status.get("displayClock") or ""); m=re.search(r"(\d+)",disp); minute=int(m.group(1)) if m else 0
            dp=str(status.get("displayPeriod") or "")
            phase = "1H"
            if "2" in dp or "2nd" in dp.lower():
                minute=min(90,45+minute); phase="2H"
            out.append(SMatch("espn",league,home,away,minute,sh,sa,phase))
        dbg(f"[SOCCER] ESPN IN: {len(out)}")
    except Exception as e:
        log_error(SOCCER_ERR, f"espn parse: {e}\n{traceback.format_exc()}")
    return out

def fetch_soccer_espn_all() -> List[dict]:
    raw=http_get("https://site.api.espn.com/apis/site/v2/sports/soccer/scoreboard", headers={"Accept":"application/json"})
    if not raw: return []
    try:
        return json.loads(raw.decode("utf-8","ignore")).get("events", [])
    except Exception:
        return []

# --- Odds + base heuristics ---
def s_estimate_odds(p: float, market: str) -> float:
    fair=max(1.01,1.0/max(1e-6,min(0.98,p)))
    lo=SOCCER_MIN_ODDS.get(market,1.2); hi=SOCCER_MAX_ODDS.get(market,3.5)
    return max(lo,min(hi,fair*1.08))

# ================= AI TURBO BOOSTER (port from PRE) =================
AI_TURBO = True
AI_MAX_BOOST = 0.20
AI_MIN_SAMPLES = 20
AI_EMA_ALPHA = 0.35

def _bin_minute(m:int)->str:
    if m<=15: return "00-15"
    if m<=30: return "16-30"
    if m<=45: return "31-45"
    if m<=60: return "46-60"
    if m<=75: return "61-75"
    if m<=90: return "76-90"
    return "90+"

def s_build_booster():
    # Boost probability based on recent tip outcomes by market/minute/score bins
    if not AI_TURBO:
        return lambda p, market, mobj: p
    st = s_state()
    tips = (st.get("tips") or {}).values()
    from collections import defaultdict
    wins=defaultdict(int); tot=defaultdict(int)
    for t in tips:
        if t.get("status") not in ("✅","❌"): 
            continue
        market=t.get("market","")
        minute=int(t.get("minute") or 0)
        sc=t.get("score","0-0")
        try:
            a,b=map(int, sc.split("-")); goals=a+b; diff=a-b
        except:
            goals=0; diff=0
        key=(market, _bin_minute(minute), str(goals), str(diff))
        tot[key]+=1
        if t["status"]=="✅": 
            wins[key]+=1

    def boost(p, market, mobj):
        try:
            minute=getattr(mobj,"minute",0)
            goals=getattr(mobj,"tg",0)
            diff=getattr(mobj,"sh",0)-getattr(mobj,"sa",0)
        except Exception:
            minute=0; goals=0; diff=0
        key=(market, _bin_minute(int(minute)), str(int(goals)), str(int(diff)))
        n=tot.get(key,0)
        if n<AI_MIN_SAMPLES: 
            return p
        wr=(wins.get(key,0)+1.0)/(n+2.0)  # Laplace
        mult=1.0 + max(-AI_MAX_BOOST, min(AI_MAX_BOOST, (wr-0.5)*0.8))
        p2 = max(0.0, min(0.99, p*mult))
        return p2
    return boost
# ====================================================================

def s_prob_o15(m)->float:  # base
    x=-0.2 + 1.1*(1 if m.tg>=1 else 0) + 0.9*min(m.minute,95)/95.0
    return sigmoid(x)
def s_prob_o25(m)->float:
    x=-0.6 + 0.9*m.tg + 1.0*min(m.minute,95)/95.0
    return sigmoid(x)
def s_prob_btts(m)->float:
    x=-1.1 + 1.0*(m.sh>0) + 1.0*(m.sa>0) + 0.4*min(m.minute,95)/95.0
    return sigmoid(x)
def s_prob_late(m)->float:
    x=-0.8 + 1.5*(1 if m.minute>=SOCCER_LATE_MIN else -0.5) + 0.6*(1 if m.tg in (0,1,2,3) else 0)
    return sigmoid(x)

# --- Brutal Turbo: features & ensemble ---
def s_league_tier(lg:str)->int:
    return _pre_league_tier(lg)

def s_minbin(m:int)->str:
    if m<=15: return "00-15"
    if m<=30: return "16-30"
    if m<=45: return "31-45"
    if m<=60: return "46-60"
    if m<=75: return "61-75"
    return "76-90"

def s_feats(m, market:str)->Dict[str,float]:
    feats={
        "bias":1.0,
        "minute": m.minute/95.0,
        "tg": float(m.tg),
        "diff": float(m.sh - m.sa),
        "tier1": 1.0 if s_league_tier(m.league)==1 else 0.0,
        "tier2": 1.0 if s_league_tier(m.league)==2 else 0.0,
        "tier3": 1.0 if s_league_tier(m.league)==3 else 0.0,
        "is2H": 1.0 if m.phase=="2H" else 0.0,
        "src_sofa": 1.0 if m.source=="sofascore" else 0.0,
        "src_espn": 1.0 if m.source=="espn" else 0.0,
    }
    feats[f"minbin_{s_minbin(m.minute)}"] = 1.0
    feats[f"market_{market}"] = 1.0
    return feats

def s_models() -> Dict[str, OnlineLogit]:
    return load_models(SOCCER_MODELS_PATH)

def s_models_save(models: Dict[str, OnlineLogit]) -> None:
    save_models(SOCCER_MODELS_PATH, models)

def s_ml_predict(m, market:str)->float:
    models = s_models()
    model = models.get(market) or OnlineLogit()
    p = model.predict(s_feats(m, market))
    return p

def s_ml_update(market:str, feats:Dict[str,float], y:int)->None:
    models = s_models()
    model = models.get(market) or OnlineLogit()
    model.update(feats, y)
    models[market] = model
    s_models_save(models)

# --- Tip pipeline ---
def s_state(): return load_json(SOCCER_STATE,{})
def s_save(st): save_json(SOCCER_STATE, st)

# ----------------- Market guards to avoid nonsensical tips -------------------
def _market_guard(market:str, tg:int, status:str=None):
    # tg = total goals (home+away). Avoid O1.5 if already >=2; O2.5 if >=3.
    if market in ("Over 1.5", "Over 1.5 (1st Half)") and tg >= 2:
        return False
    if market in ("Over 2.5", "Over 2.5 (1st Half)") and tg >= 3:
        return False
    # Late dynamic over uses line tg+0.5, inherently > tg, so OK.
    # Optional: block at full-time
    if status and status.upper() in ("FT","AET","FT_PEN","PEN","POSTP","ABN"):
        return False
    return True
# ----------------------------------------------------------------------------

def _is_match_finished(m) -> bool:
    """Conservative FT guard for inconsistent feeds."""
    try:
        ph = str(getattr(m, "phase", "") or "").upper()
        if ph.startswith("FT") or ph in ("AET", "FT_PEN"):
            return True
        minute = int(getattr(m, "minute", 0) or 0)
        if minute >= 90 and ph != "2H":
            return True
    except Exception:
        pass
    return False


def s_generate_tips(matches: List[SMatch]) -> List[Dict[str,Any]]:
    tips=[]
    lo,hi=SOCCER_ODDS_WINDOW
    def ok_odd(x): return x>0 and ((lo<=x<=hi) if SOCCER_USE_ODDS_WINDOW else True)
    for m in matches:
        # Finished and league guards
        if _is_match_finished(m):
            continue
        if not s_league_guard(m):
            continue
        if not s_league_guard(m):
            continue
        for market, base_fn, key in [
                    ("Over 1.5", s_prob_o15, "O1_5"),
                    ("Over 2.5", s_prob_o25, "O2_5"),
                    ("BTTS – Both Teams To Score", s_prob_btts, "BTTS"),
                    (f"Over {m.tg+0.5:.1f} (Live, {SOCCER_LATE_MIN}+)", s_prob_late, "LATE"),
                ]:

            if m.minute > 85:
                continue  # 85. perc után ne legyen tipp
            # hard guard: ha a jelenlegi gólszám már fedezi a piaci küszöböt, ugorjunk
            if not _market_guard(market, m.tg):
                continue
            if key=="LATE" and m.minute<SOCCER_LATE_MIN: continue
            if key=="BTTS" and (m.sh>0 and m.sa>0): continue  # ha már mindkettő lőtt, irreleváns
            base_p = base_fn(m)
            ml_p = s_ml_predict(m, market)
            p = blend(base_p, ml_p, alpha=(0.9 if ((s_models().get(market) or OnlineLogit()).n < 100) else 0.6))
            est = s_estimate_odds(p, key)
            if ((key=="O1_5" and p>=SOCCER_THRESH["O1_5"]) or
                (key=="O2_5" and p>=SOCCER_THRESH["O2_5"]) or
                (key=="BTTS" and p>=SOCCER_THRESH["BTTS"]) or
                (key=="LATE" and p>=SOCCER_THRESH["LATE"])) and ok_odd(est):
                ts=now_eu().strftime("%Y-%m-%d %H:%M:%S")
                tip={"tip_id": hashlib.sha1(f"{m.home}{m.away}{market}{ts}{getattr(m,'start_ts',None)}{getattr(m,'event_id',None)}".encode()).hexdigest()[:16],
                     "date":ts,"type":"single","sport":"soccer","event_id":getattr(m,"event_id",None),"start_ts":getattr(m,"start_ts",None),"source":m.source,"league":m.league,
                     "home":m.home,"away":m.away,"minute":m.minute,"score":f"{m.sh}-{m.sa}",
                     "market":market,"prob":round(p,3),"prob_base":round(base_p,3),"prob_ml":round(ml_p,3),
                     "est_odds":round(est,2),"status":"pending"}
                tips.append(tip); s_append_log({"event":"new_tip",**tip})
                st=s_state(); st.setdefault("tips",{})[tip["tip_id"]]=tip; s_save(st)
    return tips

def s_select_top(tips: List[Dict[str,Any]])->List[Dict[str,Any]]:
    """
    TIER-prioritásos szelekció: előbb Tier1 ligák tippjei, majd Tier2, aztán Tier3.
    Mindhárom szinten a valószínűség (prob) szerint csökkenő sorrendben választ.
    Per-piac limitet tiszteletben tartja (SOCCER_MARKET_CAP).
    """
    if not tips: return []
    # Számoljuk ki a liga tier-t
    for t in tips:
        t["tier"] = s_league_tier(t.get("league","") or t.get("competition","") or "")
    # Rendezés: jobb P előre, majd alacsonyabb tier érték előre
    tips.sort(key=lambda x: (-int(x.get("prob",0)*1000), x.get("tier",3)))
    out=[]; per_market={}; seen=set()
    # TIER prioritású kiválasztás
    for tier in (1,2,3):
        for t in tips:
            if t.get("tier",3)!=tier: continue
            # duplikált meccs kerülése (home-away kulcs)
            k=( _norm(t.get("home")), _norm(t.get("away")) )
            if k in seen: continue
            mkt=t.get("market")
            if per_market.get(mkt,0)>=SOCCER_MARKET_CAP: continue
            out.append(t); seen.add(k); per_market[mkt]=per_market.get(mkt,0)+1
            if len(out)>=SOCCER_MAX_TIPS: break
        if len(out)>=SOCCER_MAX_TIPS: break
    return out

def s_print_block(final_tips: List[Dict[str,Any]])->None:
    print("\n" + "="*72)
    print(f"FOCI LIVE TIPPEK ({len(final_tips)}) @ {now_eu().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*72)
    if not final_tips:
        print("Nincs tipp ebben a ciklusban.")
        return
    for i,t in enumerate(final_tips,1):
        print(f"TIP {i:02d}: {t['home']} – {t['away']}   [{t.get('league','')}]   ⏱ {t.get('minute','')}'   ⚽ {t.get('score','')}   {t.get('source','')}")
        print(f"Piac: {t.get('market','')}  P≈{int(t.get('prob',0)*100)}%  [tier={t.get('tier',3)}]  base {int(t.get('prob_base',0)*100)}% | ml {int(t.get('prob_ml',0)*100)}%)   ~{t.get('est_odds','')}")
        print("-"*72)

def s_build_combo(tips: List[Dict[str,Any]])->Optional[Dict[str,Any]]:
    if not tips: return None
    chosen=[]; seen=set()
    for t in tips:
        k=(_norm(t["home"]),_norm(t["away"]))
        if k in seen: continue
        chosen.append(t); seen.add(k)
        if len(chosen)>=3: break
    if len(chosen)<2: return None
    tot=1.0
    for t in chosen: tot*=float(t.get("est_odds") or 1.0)
    combo={"sport":"soccer","legs":chosen,"est_total_odds":round(tot,2),"status":"pending","combo_id":None}
    ts=now_eu().strftime("%Y-%m-%d %H:%M:%S")
    combo_id=hashlib.sha1(("|".join(l["tip_id"] for l in chosen)+ts).encode()).hexdigest()[:16]
    combo["combo_id"]=combo_id
    s_append_log({"event":"new_combo","combo_id":combo_id,"legs":[l["tip_id"] for l in chosen],"ts":ts})
    st=s_state(); st.setdefault("combos",{})[combo_id]=combo; s_save(st)
    print("\n" + "="*72)
    print("FOCI – LIVE SZELVÉNY")
    print("="*72)
    print(f"Össz-odds (becsült): ~{combo['est_total_odds']}")
    for i,leg in enumerate(combo["legs"],1):
        print(f"Meccs {i}: {leg['home']} – {leg['away']}   [{leg['league']}]  {leg['market']}  P≈{int(leg['prob']*100)}%  ~{leg['est_odds']}")
    return combo

def parse_soccer_board() -> Dict[Tuple[str,str], Dict[str,Any]]:
    out={}
    events=fetch_soccer_espn_all()
    for ev in events:
        comp=(ev.get("competitions") or [{}])[0]
        status=(comp.get("status") or {}).get("type",{})
        state=(status.get("state") or "").upper()
        comps=comp.get("competitors") or []
        home=away=""; sh=sa=0
        for c in comps:
            nm=(c.get("team") or {}).get("name","")
            sc=int(c.get("score") or 0)
            if c.get("homeAway")=="home":
                home,sh=nm,sc
            else:
                away,sa=nm,sc
        out[(_norm(home),_norm(away))] = {"state":state,"home":sh,"away":sa}
    return out

def soccer_check_results():
    st=s_state()
    if not st: return
    board=parse_soccer_board()
    changed=False
    for tid, tip in list(st.get("tips",{}).items()):
        if tip.get("status")!="pending": continue
        key=(_norm(tip["home"]), _norm(tip["away"]))
        ev=board.get(key)
        if not ev or ev["state"]!="POST": continue
        win=None
        market=tip.get("market","")
        tg=ev["home"]+ev["away"]
        if market.startswith("Over "):
            m=re.search(r"Over\s+([\d\.]+)", market)
            if m: win = tg > float(m.group(1))
        elif market.startswith("BTTS"):
            win = ev["home"]>0 and ev["away"]>0

        # update ML with result
        y = 1 if win else 0
        feats = s_feats(type("MM", (), tip)(), market)  # lightweight obj from tip
        try:
            s_ml_update(market, feats, y)
        except Exception as e:
            log_error(SOCCER_ERR, f"ml_update err: {e}")

        tip["status"]="⚠" if win is None else ("✅" if win else "❌")
        log_write(tip)
        st["tips"][tid]=tip
        s_append_log({"event":"tip_result","tip_id":tid,"status":tip["status"],"ft":[ev["home"],ev["away"]]})
        log_write(tip)
        changed=True
    for cid, combo in list(st.get("combos",{}).items()):
        if combo.get("status")!="pending": continue
        leg_ids=[l["tip_id"] for l in combo.get("legs",[])]
        leg_states=[st.get("tips",{}).get(tid,{}).get("status") for tid in leg_ids]
        if not leg_states or any(s=="pending" for s in leg_states): continue
        if any(s=="❌" for s in leg_states): combo["status"]="❌"
        elif all(s=="✅" for s in leg_states): combo["status"]="✅"
        else: combo["status"]="⚠"
        st["combos"][cid]=combo
        s_append_log({"event":"combo_result","combo_id":cid,"status":combo["status"],"legs":leg_ids})
        changed=True
    if changed:
        s_save(st)
        # removed empty try/except

# --------- PRE (scheduled) ---------
def fetch_soccer_sofa_scheduled() -> List[dict]:
    out: List[dict] = []
    try:
        today_utc = dt.datetime.now(dt.timezone.utc).date()
        for d in (today_utc, today_utc + dt.timedelta(days=1)):
            url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{d.isoformat()}"
            raw = http_get(url, headers={"Accept": "application/json"})
            if not raw:
                continue
            data = json.loads(raw.decode("utf-8", "ignore"))
            for ev in data.get("events", []):
                st = ((ev.get("status") or {}).get("type") or "").lower()
                if st not in ("notstarted", "scheduled", "pre"):
                    continue
                start_ts = ev.get("startTimestamp")
                if not start_ts:
                    continue
                ko_utc = dt.datetime.fromtimestamp(int(start_ts), tz=dt.timezone.utc)
                league = (ev.get("tournament") or {}).get("name") or "SofaScore"
                home = (ev.get("homeTeam") or {}).get("name") or "Home"
                away = (ev.get("awayTeam") or {}).get("name") or "Away"
                out.append({
                    "source": "sofascore",
                    "league": league,
                    "home": home,
                    "away": away,
                    "kickoff_utc": ko_utc,
                })
    except Exception as e:
        log_error(SOCCER_ERR, "fetch_soccer_sofa_scheduled error: %s\n%s" % (e, traceback.format_exc()))
    return out

def fetch_soccer_espn_scheduled() -> List[dict]:
    out: List[dict] = []
    for ev in fetch_soccer_espn_all():
        comp = (ev.get("competitions") or [{}])[0]
        status = (comp.get("status") or {}).get("type", {})
        state = (status.get("state") or "").upper()
        if state not in ("PRE", "SCHEDULED"):
            continue
        date_iso = comp.get("date") or ev.get("date")
        try:
            ko_utc = dt.datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
        except Exception:
            continue
        league = (ev.get("league") or {}).get("name") or "ESPN"
        home = away = ""
        for c in comp.get("competitors") or []:
            nm = (c.get("team") or {}).get("name") or ""
            if c.get("homeAway") == "home":
                home = nm
            else:
                away = nm
        out.append({
            "source": "espn",
            "league": league,
            "home": home,
            "away": away,
            "kickoff_utc": ko_utc,
        })
    return out

def soccer_preszelo():
    if not PRE_ENABLED: return
    try:
        es = fetch_soccer_espn_scheduled()
        ss = fetch_soccer_sofa_scheduled()
        candidates = es + ss
        dbg(f"[PRE] scheduled: {len(candidates)} jelölt")
        now = now_nl()
        from_min = PRE_WINDOW_MIN_FROM; to_min = PRE_WINDOW_MIN_TO
        in_window: List[dict] = []
        for r in candidates:
            ko_utc = r["kickoff_utc"]
            ko_nl = ko_utc.astimezone(TZ_NL) if TZ_NL else ko_utc
            delta_min = int((ko_nl - now).total_seconds() // 60)
            if from_min <= delta_min <= to_min:
                rr = dict(r); rr["delta_min"] = delta_min; rr["kickoff_nl"] = ko_nl; in_window.append(rr)
        in_window.sort(key=lambda x: x["kickoff_nl"])
        print("\n" + "="*72)
        print(f"PRE-SZELVÉNY – NL idő ({from_min}–{to_min} perc múlva kezdők) @ {now.strftime('%Y-%m-%d %H:%M')}")
        print("="*72)
        if not in_window:
            print("Nincs most megosztható PreSzeló jelölt."); return
        shown = 0
        for i, r in enumerate(in_window, 1):
            if shown >= PRE_MAX_MATCHES: break
            ko_str = r['kickoff_nl'].strftime('%Y-%m-%d %H:%M')
            print(f"{i}. {r['home']} – {r['away']}   [{r['league']}]   kickoff (NL): {ko_str}  (≈{r['delta_min']} perc)")
            p15, pbtts, p25 = pre_prob_model(r["home"], r["away"], r["league"], r["delta_min"])
            print(f"   • Over 1.5     P≈{int(p15*100)}%   ~{round(s_estimate_odds(p15,'O1_5'),2)}")
            print(f"   • BTTS         P≈{int(pbtts*100)}%   ~{round(s_estimate_odds(pbtts,'BTTS'),2)}")
            print(f"   • Over 2.5     P≈{int(p25*100)}%   ~{round(s_estimate_odds(p25,'O2_5'),2)}")
            # --- PRE Gólszerző jelöltek (ha van players_db) ---
            try:
                scorers = _pre_scorer_candidates(r["home"], r["away"], r["league"])
                if scorers:
                    for s_name, s_prob, s_odd, s_team in scorers:
                        odd_txt = f" ~{s_odd:.2f}" if isinstance(s_odd, (int,float)) else ""
                        print(f"   • Gólszerző  {s_team}: {s_name}    P≈{int(s_prob*100)}%{odd_txt}")
            except Exception as _e:
                # ne állítsa meg a pre-blokkot, ha nincs DB vagy hiba van
                pass
            print("-"*72); shown += 1
# --- export a legacy_adapter számára (PRE) ---
            global exported_preszelo
            exported_preszelo = in_window
    except Exception as e:
        log_error(SOCCER_ERR, "pre-szelvény error: %s\n%s" % (e, traceback.format_exc()))
        print("PRE-SZELVÉNY: hiba – részletek a logban.")

# ===================================================================
# ========================== TENNIS MODULE ==========================
# ===================================================================
TENNIS_STATE = "tennis_state.json"
TENNIS_LOG   = os.path.join(LOG_DIR, "tennis_tips_log.jsonl")
TENNIS_ERR   = os.path.join(LOG_DIR, "tennis_errors.log")
TENNIS_MODELS_PATH = "tennis_ai_models.json"  # per-market OnlineLogit

TENNIS_USE_ODDS_WINDOW = False
TENNIS_ODDS_WINDOW = (1.20, 1.80)
TENNIS_PROB_MIN = 0.50
TENNIS_MAX_TIPS = 8
TENNIS_MARKET_CAP = 5

T_MIN_ODDS = {"SET_WIN": 1.20, "SETS": 1.25, "GAMES": 1.25}
T_MAX_ODDS = {"SET_WIN": 2.80, "SETS": 3.60, "GAMES": 3.40}
T_THRESH   = {"SET_WIN": 0.50, "SETS": 0.50, "GAMES": 0.50}

class TMatch:
    def __init__(self, source, tour, p1, p2, scores, best_of=3, status="IN", gender=None):
        self.source=source; self.tour=tour or "Tennis"; self.p1=p1; self.p2=p2
        self.scores=scores[:]; self.best_of=best_of; self.status=status; self.gender=gender
    @property
    def is_live(self): return (self.status or "").upper() in ("IN","INPROGRESS","LIVE")
    @property
    def current_set(self): return max(1, len(self.scores))
    @property
    def games_in_current(self): return self.scores[-1] if self.scores else (0,0)

def infer_best_of(tour: str, gender: Optional[str]) -> int:
    n=(tour or "").lower()
    slam = any(k in n for k in ["wimbledon","roland garros","us open","australian open","grand slam"])
    if slam and (not gender or _norm(gender) not in ("women","wta","ladies","girls","female")):
        return 5
    return 3

def t_fetch_espn() -> List[TMatch]:
    out=[]
    raw=http_get("https://site.api.espn.com/apis/site/v2/sports/tennis/scoreboard", headers={"Accept":"application/json"})
    if not raw: return out
    try:
        data=json.loads(raw.decode("utf-8","ignore"))
        for ev in data.get("events", []):
            comp=(ev.get("competitions") or [{}])[0]
            st=((comp.get("status") or {}).get("type") or {}).get("state","").upper()
            if st!="IN": continue
            tour=((ev.get("league") or {}).get("name")) or ((comp.get("league") or {}).get("name")) or "ESPN"
            comps=comp.get("competitors") or []
            p1=p2=""; s1=[]; s2=[]
            for c in comps:
                nm=(c.get("athlete") or {}).get("displayName") or (c.get("team") or {}).get("displayName") or (c.get("name"))
                if c.get("homeAway")=="home":
                    p1=nm or "Player 1"; s1=[int(x.get("value",0) or 0) for x in (c.get("linescores") or [])]
                else:
                    p2=nm or "Player 2"; s2=[int(x.get("value",0) or 0) for x in (c.get("linescores") or [])]
            scores=[]; ml=max(len(s1),len(s2))
            for i in range(ml):
                a=s1[i] if i<len(s1) else 0; b=s2[i] if i<len(s2) else 0
                scores.append((a,b))
            gender=(ev.get("group") or {}).get("name") or None
            out.append(TMatch("espn",tour,p1,p2,scores,best_of=infer_best_of(tour,gender),status="IN",gender=gender))
        dbg(f"[TENNIS] ESPN live: {len(out)}")
    except Exception as e:
        log_error(TENNIS_ERR, f"espn parse: {e}\n{traceback.format_exc()}")
    return out

def t_fetch_sofa() -> List[TMatch]:
    out=[]
    raw=http_get("https://api.sofascore.com/api/v1/sport/tennis/events/live", headers={"Accept":"application/json"})
    if not raw: return out
    try:
        data=json.loads(raw.decode("utf-8","ignore"))
        for ev in data.get("events", []):
            st=((ev.get("status") or {}).get("type") or "").lower()
            if st not in ("inprogress","live"): continue
            tour=(ev.get("tournament") or {}).get("name") or "SofaScore"
            p1=(ev.get("homeTeam") or {}).get("name") or "Player 1"
            p2=(ev.get("awayTeam") or {}).get("name") or "Player 2"
            scores=[(0,0)]
            gender=(ev.get("category") or {}).get("name") or None
            out.append(TMatch("sofascore",tour,p1,p2,scores,best_of=infer_best_of(tour,gender),status="IN",gender=gender))
        dbg(f"[TENNIS] Sofa live: {len(out)}")
    except Exception as e:
        log_error(TENNIS_ERR, f"sofa parse: {e}\n{traceback.format_exc()}")
    return out

def t_estimate_odds(p: float, market: str) -> float:
    fair=max(1.01,1.0/max(1e-6,min(0.98,p)))
    lo=T_MIN_ODDS.get(market,1.2); hi=T_MAX_ODDS.get(market,3.6)
    return max(lo,min(hi,fair*1.08))

def _sets_won(scores: List[Tuple[int,int]])->Tuple[int,int]:
    s1=s2=0
    for g1,g2 in scores:
        if max(g1,g2)>=6 and (abs(g1-g2)>=2 or max(g1,g2)==7):
            if g1>g2: s1+=1
            elif g2>g1: s2+=1
    return s1,s2

def t_prob_set_winner(m:TMatch, who:int)->float:
    s1,s2=_sets_won(m.scores)
    g1,g2 = m.games_in_current
    lead = (g1-g2) if who==1 else (g2-g1)
    setd = (s1-s2) if who==1 else (s2-s1)
    x=-0.2 + 0.55*lead + 0.25*setd
    if (g1==6 and g2==6) or (g1==5 and g2==5): x += 0.05*lead
    if s1==1 and s2==0 and who==1 and g1>=4 and g1-g2>=1: x += 0.35
    if s2==1 and s1==0 and who==2 and g2>=4 and g2-g1>=1: x += 0.35
    if who==1 and g1 in (5,6) and g1-g2>=1: x += 0.25
    if who==2 and g2 in (5,6) and g2-g1>=1: x += 0.25
    return max(0.01,min(0.99,sigmoid(x)))

def t_prob_total_sets_over(m:TMatch)->Tuple[float,float]:
    bo5=(m.best_of==5)
    s1,s2=_sets_won(m.scores)
    g1,g2 = m.games_in_current
    close=-abs(g1-g2); prev_close_bonus=0.0
    for a,b in m.scores[:-1]:
        if abs(a-b)<=2 and max(a,b)>=6: prev_close_bonus += 0.15
    if not bo5:
        x=-0.95 + 0.9*(1 if s1==1 and s2==1 else 0) + 0.25*close + prev_close_bonus
        return (max(0.05,min(0.95,sigmoid(x))), 2.5)
    else:
        if s1==2 and s2==2:
            x=-1.2 + 0.95; return (max(0.05,min(0.95,sigmoid(x))), 4.5)
        x=-0.75 + 0.35*close + 0.4*(1 if s1==1 and s2==1 else 0) + 0.7*(1 if (s1, s2) in [(2,1),(1,2)] else 0) + 0.2*prev_close_bonus
        return (max(0.05,min(0.95,sigmoid(x))), 3.5)

def t_prob_total_games_over(m:TMatch)->Tuple[float,float]:
    s1=s2=0; close_bonus=0.0
    for i,(a,b) in enumerate(m.scores):
        if max(a,b)>=6 and (abs(a-b)>=2 or max(a,b)==7):
            if a>b: s1+=1
            else: s2+=1
        if i<len(m.scores)-1 and abs(a-b)<=2 and max(a,b)>=6: close_bonus+=0.2
    g1,g2=m.scores[-1] if m.scores else (0,0)
    cur_close=-abs(g1-g2)*0.06
    base=close_bonus+cur_close
    line=21.5
    if base>0.12: line=22.5
    if base>0.28: line=23.5
    x=-0.55 + 0.48*base + 0.25*(s1+s2) + 0.2*(1 if max(g1,g2)>=5 else 0)
    return (max(0.05,min(0.95,sigmoid(x))), float(line))

def run_tennis_once():
    if not ENABLE_TENNIS:
        print("[INFO][TENNIS] Disabled via ENABLE_TENNIS")
        return []

    print("\n" + "="*72)
    print("TENISZ – CIKLUS INDULT @", now_iso())
    print("="*72)

    matches = t_merged()
    print(f"[DEBUG] [TENNIS] Sofa live: {len(matches)}")
    print(f"[DEBUG] [TENNIS] merged: {len(matches)}")

    tennis_check_results()

    tips = t_generate_tips(matches)
    final = t_select_top(tips)
    t_print_block(final)

    combo = t_build_combo(final)
    if combo:
        exported_tennis_combo = combo

    stats_update("tennis", final)
    return final

# --- Tennis brutal turbo ---
def t_feats(m:TMatch, market:str)->Dict[str,float]:
    s1,s2=_sets_won(m.scores); g1,g2 = m.games_in_current
    feats={
        "bias":1.0,
        "set": float(m.current_set),
        "gdiff": float(g1-g2),
        "gmax": float(max(g1,g2)),
        "sets_won_p1": float(s1),
        "sets_won_p2": float(s2),
        "bo5": 1.0 if m.best_of==5 else 0.0,
        "src_sofa": 1.0 if m.source=="sofascore" else 0.0,
        "src_espn": 1.0 if m.source=="espn" else 0.0,
    }
    feats[f"market_{market}"]=1.0
    return feats

def t_models() -> Dict[str, OnlineLogit]:
    return load_models(TENNIS_MODELS_PATH)
def t_models_save(models: Dict[str, OnlineLogit]) -> None:
    save_models(TENNIS_MODELS_PATH, models)
def t_ml_predict(m:TMatch, market:str)->float:
    models = t_models()
    model = models.get(market) or OnlineLogit()
    return model.predict(t_feats(m, market))
def t_ml_update(market:str, feats:Dict[str,float], y:int)->None:
    models = t_models()
    model = models.get(market) or OnlineLogit()
    model.update(feats, y)
    models[market] = model
    t_models_save(models)

# --- Tip pipeline ---
def t_append_log(row):
    with open(TENNIS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
def t_state(): return load_json(TENNIS_STATE,{})
def t_save(st): save_json(TENNIS_STATE, st)

def t_est_ok(x: float)->bool:
    lo,hi=TENNIS_ODDS_WINDOW
    return x>0 and ((lo<=x<=hi) if TENNIS_USE_ODDS_WINDOW else True)

def t_generate_tips(matches: List[TMatch])->List[Dict[str,Any]]:
    tips=[]; cand_with_p=[]
    for m in matches:
        if not s_league_guard(m):
            continue
# SET WINNER
        p1=t_prob_set_winner(m,1); p2=t_prob_set_winner(m,2)
        p,best=(p1,1) if p1>=p2 else (p2,2)
        p_ml=t_ml_predict(m,"Set Winner"); p=blend(p, p_ml, alpha=(0.9 if ((s_models().get(market) or OnlineLogit()).n < 100) else 0.6))
        o=t_estimate_odds(p,"SET_WIN"); lab=f"Set Winner (Set {m.current_set}) – {m.p1 if best==1 else m.p2}"
        cand_with_p.append((p,o,m,lab,"SET_WIN"))
        if p>=T_THRESH["SET_WIN"] and t_est_ok(o):
            ts=now_eu().strftime("%Y-%m-%d %H:%M:%S")
            tip={"tip_id": hashlib.sha1(f"{m.home}{m.away}{market}{ts}{getattr(m,'start_ts',None)}{getattr(m,'event_id',None)}".encode()).hexdigest()[:16],
                 "date":ts,"type":"single","sport":"tennis","source":m.source,"tournament":m.tour,
                 "p1":m.p1,"p2":m.p2,"market":lab,"prob":round(p,3),"est_odds":round(o,2),
                 "score":m.scores,"best_of":m.best_of,"status":"pending"}
            tips.append(tip); t_append_log({"event":"new_tip",**tip})
            st=t_state(); st.setdefault("tips",{})[tip["tip_id"]]=tip; t_save(st)
        # TOTAL SETS
        ps,ls=t_prob_total_sets_over(m); p_ml=t_ml_predict(m,"Total Sets"); ps=blend(ps, p_ml, alpha=(0.9 if ((s_models().get(market) or OnlineLogit()).n < 100) else 0.6))
        os=t_estimate_odds(ps,"SETS"); lab=f"Total Sets – Over {ls}"
        cand_with_p.append((ps,os,m,lab,"SETS"))
        if ps>=T_THRESH["SETS"] and t_est_ok(os):
            ts=now_eu().strftime("%Y-%m-%d %H:%M:%S")
            tip={"tip_id": hashlib.sha1(f"{m.home}{m.away}{market}{ts}{getattr(m,'start_ts',None)}{getattr(m,'event_id',None)}".encode()).hexdigest()[:16],
                 "date":ts,"type":"single","sport":"tennis","source":m.source,"tournament":m.tour,
                 "p1":m.p1,"p2":m.p2,"market":lab,"prob":round(ps,3),"est_odds":round(os,2),
                 "score":m.scores,"best_of":m.best_of,"status":"pending"}
            tips.append(tip); t_append_log({"event":"new_tip",**tip})
            st=t_state(); st.setdefault("tips",{})[tip["tip_id"]]=tip; t_save(st)
        # TOTAL GAMES
        pg,lg=t_prob_total_games_over(m); p_ml=t_ml_predict(m,"Total Games"); pg=blend(pg, p_ml, alpha=(0.9 if ((s_models().get(market) or OnlineLogit()).n < 100) else 0.6))
        og=t_estimate_odds(pg,"GAMES"); lab=f"Total Games – Over {lg}"
        cand_with_p.append((pg,og,m,lab,"GAMES"))
        if pg>=T_THRESH["GAMES"] and t_est_ok(og):
            ts=now_eu().strftime("%Y-%m-%d %H:%M:%S")
            tip={"tip_id": hashlib.sha1(f"{m.home}{m.away}{market}{ts}{getattr(m,'start_ts',None)}{getattr(m,'event_id',None)}".encode()).hexdigest()[:16],
                 "date":ts,"type":"single","sport":"tennis","source":m.source,"tournament":m.tour,
                 "p1":m.p1,"p2":m.p2,"market":lab,"prob":round(pg,3),"est_odds":round(og,2),
                 "score":m.scores,"best_of":m.best_of,"status":"pending"}
            tips.append(tip); t_append_log({"event":"new_tip",**tip})
            st=t_state(); st.setdefault("tips",{})[tip["tip_id"]]=tip; t_save(st)
    if not tips and cand_with_p:
        cand_with_p.sort(key=lambda x:(x[0],x[1]), reverse=True)
        for p,o,m,lab,_ in cand_with_p[:4]:
            if p < 0.40: break
            ts=now_eu().strftime("%Y-%m-%d %H:%M:%S")
            tip={"tip_id": hashlib.sha1(f"{m.home}{m.away}{market}{ts}{getattr(m,'start_ts',None)}{getattr(m,'event_id',None)}".encode()).hexdigest()[:16],
                 "date":ts,"type":"single","sport":"tennis","source":m.source,"tournament":m.tour,
                 "p1":m.p1,"p2":m.p2,"market":lab,"prob":round(p,3),"est_odds":round(o,2),
                 "score":m.scores,"best_of":m.best_of,"status":"pending","note":"fallback"}
            tips.append(tip); t_append_log({"event":"new_tip",**tip})
            st=t_state(); st.setdefault("tips",{})[tip["tip_id"]]=tip; t_save(st)
    return tips

def t_select_top(tips: List[Dict[str,Any]])->List[Dict[str,Any]]:
    tips=[t for t in tips if t.get("prob",0)>=TENNIS_PROB_MIN]
    tips.sort(key=lambda x:(x.get("prob",0), x.get("est_odds",0)), reverse=True)
    out=[]; cap={}; seen=set()
    for t in tips:
        k=(_norm(t["p1"]),_norm(t["p2"]))
        if k in seen: continue
        if cap.get(t["market"],0) >= TENNIS_MARKET_CAP: continue
        out.append(t); seen.add(k); cap[t["market"]] = cap.get(t["market"],0)+1
        if len(out) >= TENNIS_MAX_TIPS: break
    return out

def t_print_block(final_tips: List[Dict[str,Any]])->None:
    print("\n" + "="*72)
    print(f"TENISZ TIPPEK ({len(final_tips)}) @ {now_eu().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*72)
    if not final_tips:
        print("Nincs tipp ebben a ciklusban.")
        return
    for i,t in enumerate(final_tips,1):
        print(f"TIP {i:02d}: {t['p1']} vs {t['p2']}   [{t['tournament']}]   {t['source']}")
        print(f"Piac: {t['market']}  P≈{int(t['prob']*100)}%  [tier={t.get('tier',3)}]  ~{t.get('est_odds','')}  best-of: {t.get('best_of','')}")
        print("-"*72)

def parse_finished_tennis() -> Dict[Tuple[str,str], Dict[str, Any]]:
    out={}
    raw=http_get("https://site.api.espn.com/apis/site/v2/sports/tennis/scoreboard", headers={"Accept":"application/json"})
    if not raw: return out
    try:
        data=json.loads(raw.decode("utf-8","ignore"))
        for ev in data.get("events", []):
            comp=(ev.get("competitions") or [{}])[0]
            st=((comp.get("status") or {}).get("type") or {}).get("state","").upper()
            comps=comp.get("competitors") or []
            p1=p2=""; s1=[]; s2=[]
            for c in comps:
                nm=(c.get("athlete") or {}).get("displayName") or (c.get("team") or {}).get("displayName") or (c.get("name"))
                if c.get("homeAway")=="home":
                    p1=nm or "Player 1"; s1=[int(x.get("value",0) or 0) for x in (c.get("linescores") or [])]
                else:
                    p2=nm or "Player 2"; s2=[int(x.get("value",0) or 0) for x in (c.get("linescores") or [])]
            scores=[]; ml=max(len(s1),len(s2))
            for i in range(ml):
                a=s1[i] if i<len(s1) else 0; b=s2[i] if i<len(s2) else 0
                scores.append((a,b))
            total_games=sum(a+b for a,b in scores)
            sw1=sw2=0
            for a,b in scores:
                if max(a,b)>=6 and (abs(a-b)>=2 or max(a,b)==7):
                    if a>b: sw1+=1
                    elif b>a: sw2+=1
            out[(_norm(p1),_norm(p2))] = {"state": st, "scores": scores, "sets_won": (sw1,sw2), "total_games": total_games}
    except Exception as e:
        log_error(TENNIS_ERR, f"parse_finished_tennis error: {e}\n{traceback.format_exc()}")
    return out

def tennis_check_results():
    st=t_state()
    if not st or not st.get("tips"): return
    board=parse_finished_tennis()
    changed=False
    for tid, tip in list(st["tips"].items()):
        if tip.get("status")!="pending": continue
        key=(_norm(tip["p1"]), _norm(tip["p2"]))
        ev=board.get(key)
        if not ev or ev["state"]!="POST": continue
        win=None
        market=tip.get("market","")
        scores=ev.get("scores") or []
        total_games=ev.get("total_games") or 0
        if market.startswith("Total Sets – Over"):
            m=re.search(r"Over\s+([\d\.]+)", market)
            if m:
                line=float(m.group(1))
                total_sets=sum(1 for s1,s2 in scores if max(s1,s2)>=6 and (abs(s1-s2)>=2 or max(s1,s2)==7))
                win = total_sets > line
        elif market.startswith("Total Games – Over"):
            m=re.search(r"Over\s+([\d\.]+)", market)
            if m:
                line=float(m.group(1))
                win = total_games > line

        # update ML with result
        y = 1 if win else 0
        feats = t_feats(type("TM", (), tip)(), market.split(" – ")[0])
        try:
            t_ml_update(market.split(" – ")[0], feats, y)
        except Exception as e:
            log_error(TENNIS_ERR, f"ml_update err: {e}")

        tip["status"]="⚠" if win is None else ("✅" if win else "❌")
        log_write(tip)
        st["tips"][tid]=tip
        t_append_log({"event":"tip_result","tip_id":tid,"status":tip["status"]})
        log_write(tip)
        changed=True
    if changed: t_save(st)

# ===================================================================
# ============================= MAIN LOOP ===========================
# ===================================================================
def s_merged() -> List[SMatch]:
    allm=[]
    try: allm.extend(fetch_soccer_sofa())
    except Exception as e: log_error(SOCCER_ERR, f"sofa err: {e}")
    try: allm.extend(fetch_soccer_espn())
    except Exception as e: log_error(SOCCER_ERR, f"espn err: {e}")
    uniq={}
    for m in allm:
        k=(_norm(m.home),_norm(m.away))
        if (k not in uniq) or (uniq[k].source!="sofascore"):
            uniq[k]=m
    res=list(uniq.values()); dbg(f"[SOCCER] merged: {len(res)}"); return res

def t_merged()->List[TMatch]:
    allm=[]
    try: allm.extend(t_fetch_espn())
    except Exception as e: log_error(TENNIS_ERR, f"espn err: {e}")
    try: allm.extend(t_fetch_sofa())
    except Exception as e: log_error(TENNIS_ERR, f"sofa err: {e}")
    uniq={}
    for m in allm:
        k=(_norm(m.p1),_norm(m.p2))
        if (k not in uniq) or (uniq[k].source!="espn"):
            uniq[k]=m
    res=list(uniq.values()); dbg(f"[TENNIS] merged: {len(res)}"); return res
# ============ Gólszerző modul ============


# =============================================================================
# FULL BRUTAL gólszerző-ajánló (heurisztika + players_db.json ha van)
# =============================================================================
def brutal_gs(final_tips):
    # --- v3.1: BRUTAL gólszerző – fuzzy team match + heurisztika fallback ---
    def brutal_gs(final_tips):
        import json, os
        from math import isfinite

        def _gs_norm(s):
            if not s: return ""
            s = s.lower()
            s = re.sub(r"[^a-z0-9]+"," ", s).strip()
            # rövid aliasok rendezése
            s = s.replace(" fc","").replace(" sc","").replace(" afc","")
            s = s.replace(" cf","").replace(" bk","").replace(" if","")
            return re.sub(r"\s+"," ", s)

        def _fuzzy_score(a,b):
            A=set(_gs_norm(a).split()); B=set(_gs_norm(b).split())
            if not A or not B: return 0.0
            inter=len(A&B); uni=len(A|B)
            return inter/uni

        # players_db.json beolvasása több tipikus helyről
        db_paths = [
            "players_db.json",
            #"/storage/emulated/0/Tipster/players_db.json",
            #"/storage/emulated/0/shared/tipster/players_db.json",
        #]
        players_db = {}
        for pth in db_paths:
            try:
                if os.path.exists(pth):
                    with open(pth,"r",encoding="utf-8") as f:
                        players_db = json.load(f)
                        break
            except Exception:
                pass

        # DB-t előkészítjük gyors kereséshez
        db_idx = []
        for team, lst in players_db.items():
            db_idx.append((_gs_norm(team), team, lst))

        def _pick_candidates(team_name, k=3, base_pct=70):
            # 1) próbáljuk DB-ből
            tnorm = _gs_norm(team_name)
            best = None
            best_sc = 0.0
            for nrm, orig, lst in db_idx:
                sc = 1.0 if nrm == tnorm else _fuzzy_score(nrm, tnorm)
                if sc > best_sc:
                    best_sc, best = sc, (orig, lst)
            if best and best_sc >= 0.5:  # elég jó egyezés
                orig, lst = best
                # súly szerinti top k
                lst2 = sorted(lst, key=lambda x: float(x.get("w",0)), reverse=True)[:k]
                out=[]
                for i,p in enumerate(lst2,1):
                    name = p.get("name","?")
                    w    = float(p.get("w",0))
                    pct  = int(min(95, max(55, base_pct + int(w*20))))
                    out.append((i, name, pct, orig))
                return out, f"DB egyezés: {orig} (score={best_sc:.2f})"

            # 2) nincs DB – heurisztika fallback a meglévő függvénnyel
            try:
                m = {"home": team_name, "away": ""}  # dummy
                from types import SimpleNamespace
                fake_match = {"home": team_name, "away": ""}
            except Exception:
                pass
            return [], "Nincs DB egyezés"

        # kiírás kezdete
        print("="*72)
        print("GÓLSZERZŐ-JELÖLTEK (FULL BRUTAL)")
        print("="*72)

        printed = 0
        for t in final_tips or []:
            h = t.get("home") or t.get("p1") or ""
            a = t.get("away") or t.get("p2") or ""
            if not (h or a): 
                continue

            # próbáljuk mindkét oldalra
            all_lines=[]
            meta=[]
            for team in (h,a):
                if not team: 
                    continue
                cand, info = _pick_candidates(team, k=3, base_pct=int(t.get("p",70)))
                meta.append(info)
                for i,name,pct,orig in cand:
                    all_lines.append(f"  {orig}: {i}. {name}  ~{pct}%")

            if not all_lines:
                # heurisztika teljes fallback, ha a modul elérhető
                try:
                    # ha van heurisztikus jelöltkereső: _gs_candidates_for_match
                    m = {"home": h, "away": a}
                    cands = _gs_candidates_for_match(m, players_db)  # nem baj, ha üres
                    if cands:
                        for i,c in enumerate(cands[:3],1):
                            pct = int(min(95, max(55, int(t.get('p',70)))))
                            all_lines.append(f"  {h or a}: {i}. {c.get('name','?')}  ~{pct}%")
                        meta.append("Heurisztika fallback")
                except Exception:
                    pass

            if all_lines:
                printed += 1
                print("-"*72)
                print(f"{h} - {a}")
                for ln in all_lines:
                    print(ln)
                if meta:
                    print("  [" + " | ".join(meta) + "]")

        if not printed:
            print("(nem találtam jelöltet)")
        print("="*72)
    # --- v3.1 END ---

def gs_recommend_goalscorers(final_tips):
    """
    Egyszerű gólszerző-jelölt ajánló.
    A későbbiekben bővíthető API/players_db.json alapján.
    """
    try:
        if not final_tips:
            return
        print("="*72)
        print("GÓLSZERZŐ-JELÖLTEK (heurisztika)")
        print("="*72)
        for tip in final_tips:
            home = tip.get("home", "?")
            away = tip.get("away", "?")
            print(f"{home} - {away}: [TOP gólveszélyes játékosok placeholder]")
        print("="*72)
    except Exception as e:
        print("[GS] belső hiba:", e)

# ================== AUTO-CLOSE FINISHED MATCHES (SofaScore) ==================
# NOTE: SofaScore is not an official public API. To avoid getting blocked we:
#  - throttle requests (min delay between calls)
#  - rotate User-Agent
#  - apply exponential backoff on 403/429
#  - cache event lookups for a short TTL

_SOFA_EVENT_CACHE = {}  # event_id -> (ts, payload)
_SOFA_LAST_CALL_TS = 0.0

_SOFA_UA_POOL = [
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
]

def _sofa_throttle(min_delay: float = 0.9):
    """Polite throttling between SofaScore calls."""
    import random
    global _SOFA_LAST_CALL_TS
    now = time.time()
    wait = (_SOFA_LAST_CALL_TS + float(min_delay)) - now
    if wait > 0:
        time.sleep(wait)
    # jitter so we don't look like a metronome
    time.sleep(0.15 + random.random() * 0.25)
    _SOFA_LAST_CALL_TS = time.time()

def _sofa_http_get_json(url: str, max_retries: int = 4):
    """Fetch JSON with basic anti-block hygiene (UA rotation + backoff)."""
    import random
    import json as _json
    import urllib.request as _ur
    import urllib.error as _ue

    last_err = None
    for att in range(int(max_retries)):
        try:
            _sofa_throttle()

            ua = random.choice(_SOFA_UA_POOL) if _SOFA_UA_POOL else "Mozilla/5.0"
            req = _ur.Request(url)
            req.add_header("Accept", "application/json, text/plain, */*")
            req.add_header("Accept-Language", "en-US,en;q=0.9,hu;q=0.8,nl;q=0.7")
            req.add_header("User-Agent", ua)
            req.add_header("Referer", "https://www.sofascore.com/")
            req.add_header("Origin", "https://www.sofascore.com")
            req.add_header("Connection", "keep-alive")

            with _ur.urlopen(req, timeout=12) as resp:
                raw = resp.read()
            return _json.loads(raw.decode("utf-8", "ignore"))

        except _ue.HTTPError as e:
            last_err = e
            code = getattr(e, "code", None)
            # 429/403: slow down + retry (most common block signals)
            if code in (403, 429):
                backoff = (2 ** att) * (2.0 + random.random() * 1.5)
                time.sleep(backoff)
                continue
            return None
        except Exception as e:
            last_err = e
            # transient errors: retry with small backoff
            time.sleep(0.7 + random.random() * 0.8)
            continue
    return None

def _sofa_fetch_event_result(event_id: int):
    """Return status + score for a SofaScore event id."""
    try:
        eid = int(event_id)
    except Exception:
        return None

    # cache
    now = time.time()
    cached = _SOFA_EVENT_CACHE.get(eid)
    if cached:
        ts, payload = cached
        # short TTL by default; longer once finished
        ttl = 60.0
        if isinstance(payload, dict) and _is_finished_state(str(payload.get("status","")) + " " + str(payload.get("desc",""))):
            ttl = 6 * 3600.0
        if now - float(ts) < ttl:
            return payload

    url = f"https://api.sofascore.com/api/v1/event/{eid}"
    data = _sofa_http_get_json(url)
    if not data:
        return None

    ev = (data.get("event") or {})
    st_obj = (ev.get("status") or {})
    st_type = (st_obj.get("type") or "").strip()
    st_desc = (st_obj.get("description") or "").strip()

    home = int((ev.get("homeScore") or {}).get("current") or 0)
    away = int((ev.get("awayScore") or {}).get("current") or 0)

    try:
        minute, phase = extract_minute_phase_sofa(ev)
    except Exception:
        minute, phase = 0, ""

    payload = {
        "status": st_type.lower(),
        "desc": st_desc.lower(),
        "score": f"{home}-{away}",
        "minute": int(minute) if minute is not None else 0,
        "phase": phase,
    }
    _SOFA_EVENT_CACHE[eid] = (now, payload)
    return payload

def _is_finished_state(st: str) -> bool:
    if not st: return False
    s = st.lower()
    return any(k in s for k in ("finished","after penalties","ended","ft","aet"))

def soccer_autoclose_pass():
    try:
        st = s_state()
        tips = list((st.get("tips") or {}).values())
    except Exception:
        tips = []
    if not tips:
        return

    now_ts = int(time.time())
    changed = False
    for t in tips:
        try:
            if t.get("status") in ("finished","won","lost","void"):
                continue
            start_ts = int(t.get("start_ts") or 0)
            if not start_ts and (now_ts - int(time.mktime(time.strptime(t.get("date","1970-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")))) > 4*3600:
                t["status"] = "finished"
                t["closed_ts"] = now_ts
                changed = True
                continue
            eid = t.get("event_id")
            res = _sofa_fetch_event_result(eid) if eid else None
            if res and (_is_finished_state((res.get("status","") or "") + " " + (res.get("desc","") or "")) or (start_ts and now_ts - start_ts > 3.5*3600)):
                t["status"] = "finished"
                t["final_score"] = res.get("score") if res else t.get("score")
                t["closed_ts"] = now_ts
                changed = True
        except Exception:
            continue
    if changed:
        newtips = { tip["tip_id"]: tip for tip in tips if tip.get("tip_id") }
        st = s_state()
        st["tips"] = newtips
        s_save(st)
# ================== END AUTO-CLOSE ==========================================
def run_soccer_once():
    soccer_preszelo()
    matches=s_merged()
    if not matches:
        print("\n" + "="*72)
        print("FOCI – NINCS ÉLŐ / ADAT HIBA")
        print("="*72)
        soccer_check_results()
        # removed empty try/except
        return
    tips = s_generate_tips(matches)
    final = s_select_top(tips)
    s_print_block(final)
    try:
        soccer_autoclose_pass()
    except Exception:
        pass

    try:
        _gs_block("GÓLSZERZŐ-JELÖLTEK (PRE)", final)
    except Exception as e:
        print("[GS][PRE] hiba:", e)
    _gs_discover("EXTRA GÓLSZERZŐ TIPPEK (PRE)", all_matches if "all_matches" in globals() else final)
    try:
        brutal_gs(final)
    except Exception as e:
        print("[GS] hiba:", e)
    combo = s_build_combo(final)
    # --- export a legacy_adapter számára (LIVE & SZELVÉNY) ---
    global exported_foci_live_tips, exported_foci_live_szelveny
    exported_foci_live_tips = final
    exported_foci_live_szelveny = [combo] if combo else []
    soccer_check_results()
    # removed empty try/except

    try:
        gs_recommend_goalscorers(final)
    except Exception as e:
        print("[GS] hibás futás:", e)

    stats_update("soccer", final)

def run_tennis_once():
    if not ENABLE_TENNIS:
        print('[INFO][TENNIS] Disabled via ENABLE_TENNIS=True')
        return []
    if not ENABLE_TENNIS:
        return
    print("\n" + "="*72)
    print("TENISZ – CIKLUS INDULT @", now_eu().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*72)
    matches=t_merged()
    if not matches:
        print("TENISZ – NINCS ÉLŐ / ADAT HIBA")
        tennis_check_results()
        return
    tips=t_generate_tips(matches)
    final=t_select_top(tips)
    t_print_block(final)
    tennis_check_results()
    stats_update("tennis", final)



# =================== STAT AGGREGÁTOR + CLI =======================
def compute_stats_by_period(tips: list):
    import datetime as dt
    daily = {}; weekly = {}; monthly = {}
    for t in tips:
        ts = t.get("closed_ts") or 0
        if not ts or t.get("status") != "finished":
            continue
        d = dt.datetime.fromtimestamp(ts)
        date = d.strftime("%Y-%m-%d")
        week = d.strftime("%Y-W%U")
        month = d.strftime("%Y-%m")
        pnl = max(1.0, float(t.get("placed_odds") or t.get("odds") or t.get("est_odds") or 1.0)) - 1.0 \
              if t.get("result","") in ("WIN","✔","OK","WINNER") else -1.0

        def upd(bucket, key):
            b = bucket.setdefault(key, {"count": 0, "win": 0, "loss": 0, "pnl": 0.0})
            b["count"] += 1
            if pnl > 0:
                b["win"] += 1
            else:
                b["loss"] += 1
            b["pnl"] += pnl

        upd(daily, date)
        upd(weekly, week)
        upd(monthly, month)
    return daily, weekly, monthly

def print_stats_by_period(daily, weekly, monthly):
    def print_block(name, block):
        print(f"\n=== {name.upper()} ===")
        for key in sorted(block)[-10:]:
            b = block[key]
            roi = b["pnl"] / max(1, b["count"])
            hr = b["win"] / max(1, (b["win"] + b["loss"])) * 100.0
            print(f"{key}: {b['count']} tip, ROI={roi:.2f}, Találati arány={hr:.1f}%, PnL={b['pnl']:.2f}")

    print_block("NAPI", daily)
    print_block("HETI", weekly)
    print_block("HAVI", monthly)

def run_stat_view():
    try:
        st = s_state()
        tips = list((st.get("tips") or {}).values())
    except Exception:
        tips = []
    if not tips:
        print("Nincs elérhető statisztikai adat.")
        return
    daily, weekly, monthly = compute_stats_by_period(tips)
    print_stats_by_period(daily, weekly, monthly)
# =================== END STAT AGGREGÁTOR ==========================
# === STATS MODULE (inserted before main) ===
from datetime import datetime, timedelta

def _iter_log_files(days: int = 120):
    base = Path(LOG_DIR)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    today = datetime.now().date()
    for i in range(days):
        d = today - timedelta(days=i)
        fp = base / f"{d.strftime('%Y-%m-%d')}.jsonl"
        if fp.exists():
            yield fp

def _parse_jsonl(fp):
    import json as _j
    text = ""
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield _j.loads(line)
        except Exception:
            pass

def _bucket_by_period(dt, mode):
    if mode == "daily":
        return dt.strftime("%Y-%m-%d")
    if mode == "weekly":
        y, w, _ = dt.isocalendar()
        return f"{y}-W{w:02d}"
    if mode == "monthly":
        return dt.strftime("%Y-%m")
    return dt.strftime("%Y-%m-%d")

def stats_collect(log_days=120, period="daily", sport=None, unit="tip"):
    agg = {}
    def add(pk, market, league, status, odds):
        if status not in ("won","lost","void"):
            return
        b = agg.setdefault(pk, {"count":0,"won":0,"lost":0,"void":0,"roi":0.0,"markets":{}})
        b["count"] += 1
        m = b["markets"].setdefault(market, {"count":0,"won":0,"lost":0,"void":0,"roi":0.0})
        m["count"] += 1
        if status == "won":
            b["won"] += 1; m["won"] += 1
        elif status == "lost":
            b["lost"] += 1; m["lost"] += 1
        elif status == "void":
            b["void"] += 1; m["void"] += 1
        try:
            est = float(odds or 0)
        except Exception:
            est = 0.0
        b["roi"] += (est-1.0) if status=="won" else (-1.0 if status=="lost" else 0.0)

    # JSONL logok
    for fp in _iter_log_files(days=log_days):
        for ev in _parse_jsonl(fp):
            if ev.get("event") != "new_tip":
                continue
            if sport and str(ev.get("sport","")).lower() != str(sport).lower():
                continue
            ts = ev.get("date") or ev.get("ts") or ""
            dt = None
            for fmt in ("%Y-%m-%d %H:%M:%S",):
                try:
                    dt = datetime.strptime(ts, fmt); break
                except Exception:
                    dt = None
            if dt is None:
                try:
                    dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
                except Exception:
                    continue
            pk = _bucket_by_period(dt, period)
            status = (ev.get("status") or "pending").lower()
            add(pk, ev.get("market") or "UNKNOWN", ev.get("league") or "N/A", status, ev.get("est_odds"))

    # State (aktuális/lezárt tippek)
    try:
        st = s_state()
        for tip in (st.get("tips") or {}).values():
            if sport and str(tip.get("sport","")).lower() != str(sport).lower():
                continue
            ts = tip.get("date") or ""
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            pk = _bucket_by_period(dt, period)
            status = (tip.get("status") or "pending").lower()
            add(pk, tip.get("market") or "UNKNOWN", tip.get("league") or "N/A", status, tip.get("est_odds"))
    except Exception:
        pass
    return agg

def stats_print(period="daily", sport=None, days=120, unit="tip"):
    agg = stats_collect(log_days=days, period=period, sport=sport, unit=unit)
    keys = sorted(agg.keys())
    print("\n" + "="*72)
    print(f"STAT – {period.upper()}  (sport={sport or 'all'}, unit={unit})")
    print("="*72)
    total = {"count":0,"won":0,"lost":0,"void":0,"roi":0.0}
    for k in keys:
        b = agg[k]; c,w,l,v,r = b["count"], b["won"], b["lost"], b["void"], b["roi"]
        total["count"] += c; total["won"] += w; total["lost"] += l; total["void"] += v; total["roi"] += r
        acc = (w / max(1, (w+l))) * 100.0 if (w+l)>0 else 0.0
        print(f"{k:>12} | tips={c:4d}  W={w:3d} L={l:3d} V={v:3d}  acc={acc:5.1f}%  ROI={r:+5.2f}")
    print("-"*72)
    acc = (total["won"] / max(1, (total["won"]+total["lost"]))) * 100.0 if (total["won"]+total["lost"])>0 else 0.0
    print(f"{'TOTAL':>12} | tips={total['count']:4d}  W={total['won']:3d} L={total['lost']:3d} V={total['void']:3d}  acc={acc:5.1f}%  ROI={total['roi']:+5.2f}")
# === END STATS MODULE ===



def _mark_finished_if_possible(tip):
    try:
        if tip.get("status") == "finished":
            return tip
        minute = int(tip.get("minute", 0))
        score = str(tip.get("score", "")).strip()
        phase = str(tip.get("phase", "")).lower()
        if "ft" in phase or "finished" in phase:
            tip["status"] = "finished"
        elif "-" in score:
            if "HT" in tip.get("market", "").upper() and minute >= 45:
                tip["status"] = "finished"
            elif minute >= 90:
                tip["status"] = "finished"
        return tip
    except Exception:
        return tip

def main():
    import argparse, os, traceback, time
    p = argparse.ArgumentParser(prog="tenisz_foci_v5_7.py")
    p.add_argument("--sport", choices=["soccer", "tennis", "both"], default="both",
                   help="Melyik sport fusson")
    p.add_argument("--once", action="store_true",
                   help="Csak egyszeri kör (nem loopol)")
    try:
        _default_interval = int(RUN_EVERY_SEC)  # a fájlban definiált alap
    except Exception:
        _default_interval = 60
    p.add_argument("--interval", type=int, default=_default_interval,
                   help="Loop intervallum másodpercben")
    p.add_argument("--stats-only", action="store_true",
                   help="Csak statisztika kiírása (nem generál új tippeket)")
    p.add_argument("--period", choices=["daily", "weekly", "monthly"], default="daily",
                   help="Statisztika bontás")
    p.add_argument("--days", type=int, default=120,
                   help="Hány nap logját olvassa vissza a stat modul")
    p.add_argument("--unit", choices=["tip", "slip"], default="tip",
                   help="Tippek (tip) vs szelvény (slip) alapú stat")
    p.add_argument("--close-only", action="store_true",
                   help="Csak nyitott tippek lezárása (nem generál újakat)")

    args = p.parse_args()

    # 1) Csak lezárás mód
    if args.close_only:
        try:
            if 'close_pending_tips' in globals() and callable(close_pending_tips):
                res = close_pending_tips()
                if isinstance(res, tuple) and len(res) == 2:
                    new_closed, considered = res
                    print(f"[closer] closed={new_closed} considered={considered}")
                else:
                    print("[closer] lefutott (részletes számok nélkül)")
            else:
                print("[closer] close_pending_tips() nem elérhető ebben a buildben")
        except Exception as e:
            try:
                log_error(os.path.join(LOG_DIR, "global_errors.log"),
                          f"Closer error: {e}\\n{traceback.format_exc()}")
            except Exception:
                pass
        return

    # 2) Csak statisztika mód
    if args.stats_only:
        try:
            if 'stats_print' in globals() and callable(stats_print):
                sport_arg = (args.sport if args.sport != 'both' else None)
                stats_print(period=args.period, sport=sport_arg, days=args.days, unit=args.unit)
            else:
                print("[stats] stats_print() nem elérhető ebben a buildben")
        except Exception as e:
            try:
                log_error(os.path.join(LOG_DIR, "global_errors.log"),
                          f"Stats error: {e}\\n{traceback.format_exc()}")
            except Exception:
                pass
        return

    # 3) Egyszeri kör
    if args.once:
        try:
            if args.sport in ("soccer", "both"):
                if 'run_soccer_once' in globals() and callable(run_soccer_once):
                    run_soccer_once()
            if 'ENABLE_TENNIS' in globals() and ENABLE_TENNIS and args.sport in ("tennis", "both"):
                if 'run_tennis_once' in globals() and callable(run_tennis_once):
                    run_tennis_once()
            if 'stats_maybe_print' in globals() and callable(stats_maybe_print):
                stats_maybe_print()
        except Exception as e:
            try:
                log_error(os.path.join(LOG_DIR, "global_errors.log"),
                          f"Once error: {e}\\n{traceback.format_exc()}")
            except Exception:
                pass
        return

    # 4) Folyamatos loop
    print(f"Unified tipper running | sport={args.sport} | Ctrl+C a kilépéshez")
    while True:
        try:
            if args.sport in ("soccer", "both"):
                if 'run_soccer_once' in globals() and callable(run_soccer_once):
                    run_soccer_once()
            if 'ENABLE_TENNIS' in globals() and ENABLE_TENNIS and args.sport in ("tennis", "both"):
                if 'run_tennis_once' in globals() and callable(run_tennis_once):
                    run_tennis_once()
            if 'stats_maybe_print' in globals() and callable(stats_maybe_print):
                stats_maybe_print()
        except KeyboardInterrupt:
            print("\\nLeállítva.")
            break
        except Exception as e:
            try:
                log_error(os.path.join(LOG_DIR, "global_errors.log"),
                          f"Main loop error: {e}\\n{traceback.format_exc()}")
            except Exception:
                pass
        try:
            time.sleep(max(10, int(args.interval)))
        except Exception:
            time.sleep(60)


if __name__ == "__main__":
    main()


# ===== API-FOOTBALL minimal wrapper (cache + TTL) =====
import time as _apif_t, json as _apif_json, subprocess as _apif_sp, shlex as _apif_shlex
APIF_BASE = "https://v3.football.api-sports.io"
APIF_KEY  = TIP_API_FOOTBALL_KEY

_ODDS_CACHE = (LOG_DIR / "odds_cache.json")
_FIXT_CACHE = (CACHE_DIR / "fixture_cache.json")
for _p in (_ODDS_CACHE, _FIXT_CACHE):
    try:
        if not _p.exists(): _p.write_text("{}", encoding="utf-8")
    except Exception:
        pass

def _apif_cache_load(p):
    try: return _apif_json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}
def _apif_cache_save(p, d):
    tmp = str(p)+".tmp"
    try:
        _env_Path(tmp).write_text(_apif_json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        pass

def _apif_get(path, params:dict):
    if not APIF_KEY: return None, "NO_KEY"
    q="&".join(f"{k}={_apif_shlex.quote(str(v))}" for k,v in (params or {}).items() if v is not None)
    url=f"{APIF_BASE}{path}?{q}" if q else f"{APIF_BASE}{path}"
    cmd=["curl","-sS","-m","10","--retry","2","-H",f"x-apisports-key: {APIF_KEY}","-H","Accept: application/json",url]
    try:
        out=_apif_sp.check_output(cmd).decode("utf-8","ignore")
        return _apif_json.loads(out), None
    except Exception as e:
        return None, f"APIF_ERR {e}"

def apif_odds_best(fixture_id:int, is_live:bool, ttl_live=120, ttl_pre=1800):
    now=int(_apif_t.time())
    cache=_apif_cache_load(_ODDS_CACHE)
    ent=cache.get(str(fixture_id))
    ttl = ttl_live if is_live else ttl_pre
    if ent and (now-int(ent.get('ts',0))<ttl):
        return ent
    data, err = _apif_get("/odds/live" if is_live else "/odds", {"fixture": fixture_id})
    if err or not data:
        return ent
    best={}
    try:
        for resp in data.get("response", []):
            for bk in (resp.get("bookmakers") or []):
                for bet in (bk.get("bets") or []):
                    nm=(bet.get("name") or "").lower()
                    for v in (bet.get("values") or []):
                        val=(v.get("value") or "").lower().replace(" ","")
                        try: odd=float(v.get("odd"))
                        except: continue
                        if "over/under" in nm:
                            if val in ("over1.5","over15"): best["over 1.5"]=max(best.get("over 1.5",0),odd)
                            if val in ("over2.5","over25"): best["over 2.5"]=max(best.get("over 2.5",0),odd)
                            if val in ("over5.5","over55"): best["over 5.5"]=max(best.get("over 5.5",0),odd)
                        if "both teams to score" in nm or "btts" in nm:
                            if val in ("yes","igen","ja","oui"): best["btts"]=max(best.get("btts",0),odd)
    except Exception:
        pass
    ent={"ts":now,"best":best}
    cache[str(fixture_id)]=ent
    _apif_cache_save(_ODDS_CACHE, cache)
    return ent

def apif_fixture_status(fixture_id:int, ttl=60):
    now=int(_apif_t.time())
    cache=_apif_cache_load(_FIXT_CACHE)
    ent=cache.get(str(fixture_id))
    if ent and (now-int(ent.get('ts',0))<ttl):
        return ent
    data, err = _apif_get("/fixtures", {"id": fixture_id})
    if err or not data or not data.get("response"):
        return ent
    try:
        r=data["response"][0]
        st=r.get("fixture",{}).get("status",{}).get("short")
        hs=r.get("goals",{}).get("home",0); as_=r.get("goals",{}).get("away",0)
        ent={"ts":now,"status":st,"home":hs,"away":as_}
        cache[str(fixture_id)]=ent
        _apif_cache_save(_FIXT_CACHE, cache)
        return ent
    except Exception:
        return ent
# ===== END API-FOOTBALL wrapper =====



# ===== OK3 QUALITY PACK (8 tuning) + CLOSER – v3 =====
# A blokk önálló: hívható bármely jelölt listára.
# Függ: LOG_DIR, apif_odds_best(), apif_fixture_status(), TIP_API_FOOTBALL_KEY

import time as _ok3_time

OK3_CFG = {
    "TIER_WEIGHTS": {"T1":1.0, "T2":0.7, "T3":0.4},
    "MARKET_RULES": {
        "over 1.5": {"min_prob": {"T1":0.72,"T2":0.75,"T3":0.78}, "odds": (1.20,1.60)},
        "over 2.5": {"min_prob": {"T1":0.68,"T2":0.70,"T3":0.73}, "odds": (1.45,2.10)},
        "btts":     {"min_prob": {"T1":0.62,"T2":0.65,"T3":0.68}, "odds": (1.60,2.10)},
        "over 5.5": {"min_prob": {"T1":0.40,"T2":0.45,"T3":0.50}, "odds": (3.20,6.50)},
    },
    "LIVE_MINUTE_CAP": 88,
}

def ok3_min_prob_for(market, tier):
    r = OK3_CFG["MARKET_RULES"].get((market or "").lower().strip())
    return r["min_prob"].get(tier, max(r["min_prob"].values())) if r else 1.0

def ok3_odds_window_for(market):
    r = OK3_CFG["MARKET_RULES"].get((market or "").lower().strip())
    return r["odds"] if r else (1.01, 99.0)

def ok3_passes_ht_o05_rule(c):
    m=(c.get("market") or "").lower()
    if "ht" in m and "over" in m and "0.5" in m:
        minute = int(c.get("minute") or 0)
        try:
            hs,as_=(c.get("score") or "0-0").split("-")
            hs,as_=int(hs),int(as_)
        except: hs,as_=0,0
        return minute >= 30 and hs==0 and as_==0
    return True

_OK3_RECENT_ODDS = {}  # (fixture_id, market) -> (odd, ts)
def ok3_stable_odds_api(c, now_ts):
    if not TIP_API_FOOTBALL_KEY: 
        return True
    fid = int(c.get("event_id") or c.get("fixture_id") or 0)
    if not fid: return True
    market = (c.get("market") or "").lower()
    is_live = c.get("minute") is not None
    ent = apif_odds_best(fid, is_live)
    if not ent: return True
    best = ent.get("best", {})
    if market not in best: return True
    try: o=float(best[market])
    except: return True
    prev=_OK3_RECENT_ODDS.get((fid, market))
    _OK3_RECENT_ODDS[(fid, market)]=(o, now_ts)
    # frissítjük a jelölt becsült oddsát

    # Over 1.5 floor + Asian Over 2.0 (push@2) fallback
    if market == "over 1.5" and (o < MIN_ODDS_PREDICT_O15):
        rows = _af_fetch_odds_rows_for_fixture(fid)
        alt = _af_pick_ou_over_from_rows(rows, "2")  # many feeds use "2" for 2.0
        if alt is not None and alt >= MIN_ODDS_ALL:
            o = float(alt)
            c["market"] = "Asian Over 2.0 (push@2)"
            market = "over 2.0"

    # Global minimum odds enforcement
    if o < MIN_ODDS_ALL:
        return True

    # frissítjük a jelölt becsült oddsát
    c["est_odds"] = o
    c["est_odds_src"] = "APIF"

    if not prev: return True
    po, pts = prev
    drift = abs(o-po)/max(po,1e-9)
    return not (drift > 0.12 and (now_ts-pts) < 120)

def ok3_tier_weight(t): return OK3_CFG["TIER_WEIGHTS"].get(t,0.3)

def ok3_odds_quality(odds, win):
    lo,hi=win; mid=(lo+hi)/2.0; span=(hi-lo)/2.0
    try: o=float(odds)
    except: return 0.0
    return 0.0 if span<=0 else max(0.0, 1.0-abs(o-mid)/span)

def ok3_market_bonus(m, pb):
    m=(m or "").lower(); pb=float(pb or 0.0)
    if "over 1.5" in m: return 3.0
    if "over 2.5" in m: return 2.0
    if "btts" in m:     return 1.0
    if "over 5.5" in m: return 5.0 if pb>=0.80 else -6.0
    return 0.0

def ok3_kickoff_bonus(c):
    if c.get("minute") is not None: return 0.0
    try: mins=int(c.get("mins_to_kickoff") or 999)
    except: mins=999
    return 1.5 if 60 <= mins <= 75 else 0.0

def ok3_tip_score(c):
    pb=float(c.get("prob_base") or c.get("prob") or 0.0)
    tw=ok3_tier_weight(c.get("tier") or "T3")
    oq=ok3_odds_quality(c.get("est_odds"), ok3_odds_window_for(c.get("market")))
    pen= 5.0 if (c.get("minute") and int(c["minute"])>OK3_CFG["LIVE_MINUTE_CAP"]) else 0.0
    return 100*pb + 20*tw + 10*oq + ok3_market_bonus(c.get("market"), pb) + ok3_kickoff_bonus(c) - pen

def ok3_select_diversified(pool, cap=3):
    ranked=sorted(pool, key=ok3_tip_score, reverse=True)
    out=[]; by_match=set(); by_team={}; by_league={}
    for c in ranked:
        keym=(c.get("home"),c.get("away"),c.get("start_ts"))
        if keym in by_match: continue
        h,a=c.get("home"),c.get("away"); lg=c.get("league") or "UNK"
        if by_team.get(h,0)>=2 or by_team.get(a,0)>=2: continue
        if by_league.get(lg,0)>=3: continue
        out.append(c); by_match.add(keym)
        by_team[h]=by_team.get(h,0)+1; by_team[a]=by_team.get(a,0)+1
        by_league[lg]=by_league.get(lg,0)+1
        if len(out)>=cap: break
    return out

def ok3_redcard_ok(c):
    if not c.get("red_card"): return True
    try:
        hs,as_=(c.get("score") or "0-0").split("-")
        return (int(hs)+int(as_))>=1
    except: 
        return True

def ok3_pass_filters(c, now_ts):
    # női + U22↓ kiszűrése maradjon a meglévő logikádban (nem duplikáljuk itt)
    if not ok3_passes_ht_o05_rule(c): return False, "ht_o05_rule"

    # odds ablak (API-ból csak itt kérünk, ha kell)
    lo,hi = ok3_odds_window_for(c.get("market"))
    o=None
    try: o=float(c.get("est_odds") or 0)
    except: o=0.0
    if not (o and lo<=o<=hi):
        fid = int(c.get("event_id") or c.get("fixture_id") or 0)
        ent = apif_odds_best(fid, c.get("minute") is not None) if fid else None
        if ent:
            b = ent.get("best", {}).get((c.get("market") or "").lower())
            if b:
                c["est_odds"]=b
                o=float(b)
    if not (o and lo<=o<=hi): return False, "odds_window"

    # min. prob a TIER szerint
    pb=float(c.get("prob_base") or c.get("prob") or 0.0)
    tier=c.get("tier") or "T3"
    if pb < ok3_min_prob_for(c.get("market"), tier): return False, "low_prob"

    # odds drift (ha van kulcs)
    if not ok3_stable_odds_api(c, now_ts): return False, "odds_drift"

    # live perccap
    if c.get("minute") is not None and int(c["minute"])>OK3_CFG["LIVE_MINUTE_CAP"]:
        return False, "late_minute"

    if not ok3_redcard_ok(c): return False, "redcard_rule"
    return True, None

def ok3_filter_and_select(pool, cap=3, verbose=False):
    """Visszaad: tips, reasons  (tips: kiválasztott listája)"""
    now_ts = int(_ok3_time.time())
    filtered=[]; reasons={}
    for c in (pool or []):
        ok, why = ok3_pass_filters(c, now_ts)
        if ok: filtered.append(c)
        else: reasons[why]=reasons.get(why,0)+1
    tips = ok3_select_diversified(filtered, cap=cap)
    if verbose:
        print(f"[OK3] pool={len(pool)} filtered={len(filtered)} picked={len(tips)} reasons={reasons}")
    return tips, reasons

# ---- CLOSER: zárás API-Football fixture alapján (TTL:60s) ----
def ok3_try_close_open_tips(open_tips, log_callback=None):
    """ open_tips: iterable[dict]  (event_id/fixture_id, market)
        log_callback: callable(dict) -> None  (opcionális: napi jsonl-be íráshoz)
    """
    by_fix={}
    for t in open_tips or []:
        fid=int(t.get("event_id") or t.get("fixture_id") or 0)
        if fid: by_fix[fid]=None

    for fid in list(by_fix.keys()):
        by_fix[fid]=apif_fixture_status(fid)

    for t in open_tips or []:
        fid=int(t.get("event_id") or t.get("fixture_id") or 0)
        ent=by_fix.get(fid) or {}
        st=ent.get("status")
        if st in ("FT","AET","PEN"):
            hs,as_=int(ent.get("home",0)), int(ent.get("away",0))
            m=(t.get("market") or "").lower()
            result="lost"
            if "over 1.5" in m: result = "won" if (hs+as_)>1 else "lost"
            elif "over 2.5" in m: result = "won" if (hs+as_)>2 else "lost"
            elif "over 5.5" in m: result = "won" if (hs+as_)>5 else "lost"
            elif "btts" in m:     result = "won" if (hs>0 and as_>0) else "lost"
            # HT piacokhoz félidei score kell – ha nincs, ne zárjuk automatikusan.
            ev={"event":"tip_closed","result":result,"final_score":f"{hs}-{as_}","status":st}
            try:
                ev.update({k:t.get(k) for k in ("event_id","market","home","away","league","kickoff_ts")})
            except Exception:
                pass
            if callable(log_callback):
                try: log_callback(ev)
                except Exception: pass
# ===== END OK3 QUALITY PACK v3 =====


# ===== API-Football kliens + logger =====
import time as _apif_time
from pathlib import Path as _APIF_Path

APIF_MIN_ODDS = 1.30  # minden odds legalább ennyi legyen (globális küszöb)

def _apif_log(endpoint, status, remain=None, limit=None):
    try:
        logdir = _APIF_Path(str(ZETBET_ROOT / "logs"))
        logdir.mkdir(parents=True, exist_ok=True)
        with (logdir/"apif_calls.log").open("a", encoding="utf-8") as f:
            f.write(f"{_apif_time.strftime('%Y-%m-%d %H:%M:%S')} | {endpoint} | http={status} | remain={remain}/{limit}\n")
    except Exception:
        pass

def _apif_headers():
    """API kulcs olvasása env-ből vagy .env-ből, és a helyes fejlécek összeállítása."""
    import os
    key = os.environ.get("TIP_API_FOOTBALL_KEY","")
    if not key:
       # try:
        #    for ln in open("/storage/emulated/0/zetbet/.env","r",encoding="utf-8"):
         #       if ln.startswith("TIP_API_FOOTBALL_KEY="):
          #          key = ln.split("=",1)[1].strip(); break
       # except Exception:
        #    pass
    return key, {"x-apisports-key": key, "Accept":"application/json"}

def _apif_get(endpoint, params=None, timeout=15):
    """Egységes GET – requests + x-apisports-key; rate-limit loggal."""
    if params is None: params = {}
    key, headers = _apif_headers()
    if not key:
        print("[APIF] nincs TIP_API_FOOTBALL_KEY (.env vagy export)"); return None
    base = "https://v3.football.api-sports.io"
    url = base + endpoint
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After","10")); _apif_time.sleep(min(wait,30))
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
        # rate-limit log
        remain = r.headers.get("x-ratelimit-requests-remaining")
        limit  = r.headers.get("x-ratelimit-requests-limit")
        _apif_log(endpoint, r.status_code, remain, limit)
        if not r.ok:
            print("[APIF] ERROR", r.status_code, r.text[:180]); return None
        return r.json()
    except Exception as e:
        print("[APIF] EXC", e); return None

def _pick_best_odds_from_response(resp, min_odds=APIF_MIN_ODDS):
    """bookmakers/bets/values fából legnagyobb odds >= min_odds"""
    try:
        best = None
        for item in resp:
            for bm in item.get("bookmakers", []):
                for bet in bm.get("bets", []):
                    for val in bet.get("values", []):
                        v = val.get("odd") or val.get("value")
                        try: vv = float(str(v).replace(",","."))
                        except Exception: continue
                        if vv >= min_odds and (best is None or vv > best):
                            best = vv
        return best
    except Exception:
        return None

# ===== Közvetlen helper függvények =====

def apif_pre_odds_for_fixture(fid, min_odds=APIF_MIN_ODDS):
    js = _apif_get("/odds", {"fixture": int(fid)})
    resp = (js or {}).get("response", [])
    return _pick_best_odds_from_response(resp, min_odds=min_odds)

def apif_live_odds_for_fixture(fid, min_odds=APIF_MIN_ODDS):
    js = _apif_get("/odds/live", {"fixture": int(fid)})
    resp = (js or {}).get("response", [])
    return _pick_best_odds_from_response(resp, min_odds=min_odds)

def apif_fixture_status(fid):
    """Rövid státusz: NS, 1H, 2H, FT, PST, ..."""
    try:
        js = _apif_get("/fixtures", {"id": int(fid)})
        fx = (js or {}).get("response", [])
        if not fx: return None
        return ((fx[0].get("fixture",{}) or {}).get("status",{}) or {}).get("short")
    except Exception:
        return None

def apif_fixture_result_by_id(fid):
    """{'status','home','away','hs','as'} – FT esetén gólokkal."""
    js = _apif_get("/fixtures", {"id": int(fid)})
    resp = (js or {}).get("response", [])
    if not resp: return None
    fx = resp[0]
    st = (fx.get("fixture",{}).get("status",{}) or {}).get("short")
    teams = fx.get("teams",{})
    goals = fx.get("goals",{})
    return {
        "status": st,
        "home": (teams.get("home",{}) or {}).get("name"),
        "away": (teams.get("away",{}) or {}).get("name"),
        "hs": goals.get("home"),
        "as": goals.get("away"),
    }

# ===== Kompatibilitási réteg az ok3_af_odds_patch-hez =====

class _AFCompat:
    @staticmethod
    def af_get_pre_odds(fid, min_odds=APIF_MIN_ODDS):
        return apif_pre_odds_for_fixture(fid, min_odds=min_odds)

    @staticmethod
    def af_get_live_odds(fid, min_odds=APIF_MIN_ODDS):
        return apif_live_odds_for_fixture(fid, min_odds=min_odds)

    @staticmethod
    def af_fixture_status(fid):
        return apif_fixture_status(fid)

    @staticmethod
    def af_fixture_result(fid):
        return apif_fixture_result_by_id(fid)

AF_COMPAT = _AFCompat()



# === API-FOOTBALL – mini kliens + gyors teszt ===
import requests as _apif_req
from datetime import date as _apif_date

_APIF_BASE = "https://v3.football.api-sports.io"

def _apif_load_key():
    # 1) env változó
    k = os.environ.get("TIP_API_FOOTBALL_KEY","").strip()
    if k: return k
    return ""

def apif_get(endpoint:str, params:dict=None, timeout:int=15):
    """
    Egyszerű GET a v3 API-hoz. Vissza: (status_code, json | None, response_obj)
    Napló: LOG_DIR/apif_calls.log
    """
    params = params or {}
    key = _apif_load_key()
    if not key:
        print("[APIF] Nincs kulcs (TIP_API_FOOTBALL_KEY) – kihagyva.")
        return 0, None, None

    url = (_APIF_BASE + (endpoint if endpoint.startswith("/") else ("/"+endpoint)))
    headers = {"x-apisports-key": key, "Accept": "application/json"}
    try:
        r = _apif_req.get(url, headers=headers, params=params, timeout=timeout)
        try:
            js = r.json()
        except Exception:
            js = None

        # napló
        logd = str(ZETBET_ROOT / "logs")
        try: os.makedirs(logd, exist_ok=True)
        except: pass
        logp = os.path.join(logd, "apif_calls.log")
        try:
            with open(logp,"a",encoding="utf-8") as f:
                lim = r.headers.get("X-RateLimit-Limit","?")
                rem = r.headers.get("X-RateLimit-Remaining","?")
                res = (js or {}).get("results", "?") if isinstance(js, dict) else "?"
                err = (js or {}).get("errors", {}) if isinstance(js, dict) else {}
                f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} "
                        f"{r.request.method} {endpoint} -> {r.status_code} "
                        f"| limit:{lim} remain:{rem} | results:{res} | errors:{err}\n")
        except Exception:
            pass

        if r.ok:
            return r.status_code, js, r
        return r.status_code, js, r
    except Exception as e:
        print("[APIF] Hiba:", e)
        return 0, None, None

def apif_touch_today(limit:int=3):
    """
    Kényszerített gyors teszt/hívás: /status (kvóta nem fogy), /countries, 
    /fixtures (mai nap), /odds (mai nap – ha van adat). A 'limit' a kvótát fogyasztó
    hívások maximuma.
    """
    import os
    if not os.getenv("TIP_API_FOOTBALL_KEY"):
        print("[APIF] disabled: no key")
        return 0
    calls = 0
    # /status – csak ellenőrzés (kvótát nem fogyaszt)
    sc, js, _ = apif_get("/status")
    if sc:
        plan = (((js or {}).get("response") or {}).get("subscription") or {}).get("plan","?")
        print("HTTP", sc, "| plan:", plan)


# === END API-FOOTBALL blokk ===

# ## APIF_EXPORT_OK
apif_touch_today_alias = apif_touch_today


# ======= APIF TOUCH + KOMPAT IMPORT (auto-insert) =======
# Ha van af_client, onnan vesszük a wrappert; különben fallbacket használunk.
try:
    from af_client import AF_COMPAT as _AF
    apif_fixture_status = _AF.af_fixture_status
    apif_odds_best      = _AF.af_odds_best
except Exception:
    # Mini fallback: _apif_get-re támaszkodunk (ez a fájlban már megvan).
    def apif_fixture_status(params: dict):
        try:
            js = _apif_get("/fixtures", params or {})
            return js or {}
        except Exception:
            return {}
    def apif_odds_best(fid: int):
        """Nagyon light 'best' OU 1.5 kivonás a /odds válaszból."""
        try:
            js = _apif_get("/odds", {"fixture": int(fid)}) or {}
            rows = (js or {}).get("response", [])
            o15  = _af_pick_ou_over_from_rows(rows, "1.5")
            return {"O1_5": o15}
        except Exception:
            return {}

def _apif_log(line: str):
    """Egysoros napló az API-érintésekhez."""
    import os, time
    p = LOG_DIR / "apif_calls.log"
    os.makedirs(logd exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {line}\n")
    except Exception:
        pass

def apif_touch_today(limit: int = 3):
    """Gyors 'életjel' hívás: mai fixturek + egy OU1.5 odds minta."""
    from datetime import date
    today = date.today().isoformat()
    r = apif_fixture_status({"date": today, "timezone": "UTC"})
    arr = (r or {}).get("response", [])[: max(0, int(limit))]
    print(f"GET /fixtures -> {len(arr)}")
    _apif_log(f"/fixtures {len(arr)}")
    for fx in arr:
        fid = (fx.get("fixture") or {}).get("id") or fx.get("id")
        league = (fx.get("league") or {}).get("name", "")
        home = ((fx.get("teams") or {}).get("home") or {}).get("name", "")
        away = ((fx.get("teams") or {}).get("away") or {}).get("name", "")
        if fid:
            odds = apif_odds_best(fid) or {}
        else:
            odds = {}
        print(f"{league}: {home} - {away}  ➜ odds: {odds}")
        _apif_log(f"/odds fid={fid} -> {odds}")
# Alias (ha a kód máshol erre hivatkozna)
apif_touch_today_alias = apif_touch_today
# ======= /APIF TOUCH =======


# ======= APIF BLOKK (önellátó + af_client kompat) =======
# 1) _apif_get fallback (csak ha nincs)
try:
    _apif_get
except NameError:
    def _apif_get(endpoint: str, params=None, timeout=20):
        """Közvetlen API-FOOTBALL hívás .env kulccsal (requests)."""
        import os, requests, time
        BASE = "https://v3.football.api-sports.io"
        key = os.environ.get("TIP_API_FOOTBALL_KEY","")
        if not key:
            # próbáljuk .env-ből
            envp = str(ZETBET_ROOT / .env")
            try:
                for ln in open(envp,"r",encoding="utf-8"):
                    if ln.startswith("TIP_API_FOOTBALL_KEY="):
                        key = ln.split("=",1)[1].strip(); break
            except Exception:
                pass
        if not key:
            print("[APIF] Kulcs hiányzik."); return None
        H={"x-apisports-key":key,"Accept":"application/json"}
        url = BASE + endpoint
        try:
            r = requests.get(url, headers=H, params=params or {}, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After","2"))
                time.sleep(max(2, wait))
                r = requests.get(url, headers=H, params=params or {}, timeout=timeout)
            if not r.ok:
                print("[APIF] Hiba:", r.status_code, r.text[:200])
                return None
            return r.json()
        except Exception as e:
            print("[APIF] Exception:", e)
            return None

# 2) af_client kompat import + fallback
apif_fixture_status = None
apif_odds_best      = None
try:
    import af_client as _AFM
    # AF_COMPAT osztály?
    _AF = getattr(_AFM, "AF_COMPAT", None)
    if _AF:
        apif_fixture_status = getattr(_AF, "af_fixture_status", None)
        apif_odds_best      = getattr(_AF, "af_odds_best", None)
    # közvetlen függvények?
    if apif_fixture_status is None:
        apif_fixture_status = (getattr(_AFM, "apif_fixture_status", None) or
                               getattr(_AFM, "af_fixture_status", None))
    if apif_odds_best is None:
        apif_odds_best = (getattr(_AFM, "apif_odds_best", None) or
                          getattr(_AFM, "af_odds_best", None))
except Exception:
    pass

# ha továbbra sincs, adjunk mini fallbacket
if apif_fixture_status is None:
    def apif_fixture_status(params: dict):
        js = _apif_get("/fixtures", params or {})
        return js or {}
if apif_odds_best is None:
    def _af_pick_ou_over_from_rows(rows, line_str="1.5"):
        """Megpróbál OU piacból 'Over line' szorzót kiolvasni."""
        try:
            for row in rows or []:
                for bm in row.get("bookmakers", []) or []:
                    for bet in bm.get("bets", []) or []:
                        name = (bet.get("name") or "").lower()
                        if "over/under" in name:
                            for v in bet.get("values", []) or []:
                                if (v.get("value") or "").strip() == line_str:
                                    od = v.get("odd")
                                    try: return float(od)
                                    except: return od
        except Exception:
            pass
        return None
    def apif_odds_best(fid: int):
        js = _apif_get("/odds", {"fixture": int(fid)}) or {}
        rows = (js or {}).get("response", [])
        return {"O1_5": _af_pick_ou_over_from_rows(rows, "1.5")}

# 3) egyszerű napló
def _apif_log(line: str):
    import os, time
    os.makedirs("/storage/emulated/0/zetbet/logs", exist_ok=True)
    p = "/storage/emulated/0/zetbet/logs/apif_calls.log"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(p,"a",encoding="utf-8") as f: f.write(f"[{ts}] {line}\n")
    except Exception:
        pass

# 4) napi gyors teszt eszköz
def apif_touch_today(limit: int = 3):
    from datetime import date
    today = date.today().isoformat()
    r = apif_fixture_status({"date": today, "timezone": "UTC"})
    arr = (r or {}).get("response", [])[: max(0, int(limit))]
    print(f"GET /fixtures -> {len(arr)}")
    _apif_log(f"/fixtures {len(arr)}")
    for fx in arr:
        fid = (fx.get("fixture") or {}).get("id") or fx.get("id")
        league = (fx.get("league") or {}).get("name","")
        home = ((fx.get("teams") or {}).get("home") or {}).get("name","")
        away = ((fx.get("teams") or {}).get("away") or {}).get("name","")
        odds = apif_odds_best(fid) if fid else {}
        print(f"{league}: {home} - {away}  ➜ odds: {odds}")
        _apif_log(f"/odds fid={fid} -> {odds}")
apif_touch_today_alias = apif_touch_today
# ======= /APIF BLOKK =======
# ===== API-Football DISABLED (key inactive) =====

def apif_touch_today(*args, **kwargs):
    print("[APIF] disabled (no active key)")
    return 0

def apif_fixture_status(*args, **kwargs):
    return {}

def apif_odds_best(*args, **kwargs):
    return {}

