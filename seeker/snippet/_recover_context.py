#date: 2026-01-29T17:15:31Z
#url: https://api.github.com/gists/8b8025d74508a72c276b652c66900282
#owner: https://api.github.com/users/darconada

#!/usr/bin/env python3
"""
Context Recovery Script - Detecta amnesia y ayuda a recuperar contexto.

Uso:
    python3 memory/_recover_context.py              # Chequear si hay amnesia
    python3 memory/_recover_context.py --recover    # Chequear + mostrar resumen para recuperar
    python3 memory/_recover_context.py --update     # Actualizar estado (post-recuperación)
    python3 memory/_recover_context.py --status     # Solo mostrar estado actual

Flujo recomendado:
1. Al inicio de sesión o tras mensaje sospechoso: python3 memory/_recover_context.py
2. Si detecta amnesia: python3 memory/_recover_context.py --recover
3. Tras recuperar contexto: python3 memory/_recover_context.py --update
"""

import json
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime

STATE_FILE = Path(__file__).parent / "context-state.json"

def get_session_status(session_key=None):
    """Obtiene estado actual de la sesión.
    
    Si no se especifica session_key, intenta detectar la sesión más reciente.
    """
    sessions_file = Path.home() / ".clawdbot/agents/main/sessions/sessions.json"
    if not sessions_file.exists():
        return None
    
    with open(sessions_file) as f:
        sessions = json.load(f)
    
    # Si se especifica session_key, usarlo
    if session_key and session_key in sessions:
        target_session = sessions[session_key]
    else:
        # Detectar la sesión más recientemente actualizada
        most_recent = None
        most_recent_time = 0
        for key, data in sessions.items():
            updated = data.get("updatedAt", 0)
            if updated > most_recent_time:
                most_recent_time = updated
                most_recent = (key, data)
        
        if most_recent:
            session_key, target_session = most_recent
        else:
            target_session = sessions.get("agent:main:main", {})
            session_key = "agent:main:main"
    
    return {
        "sessionKey": session_key,
        "compactionCount": target_session.get("compactionCount", 0),
        "model": target_session.get("model", "unknown"),
        "contextTokens": "**********"
        "totalTokens": "**********"
        "sessionFile": target_session.get("sessionFile", ""),
        "updatedAt": target_session.get("updatedAt", 0)
    }

def load_state():
    """Carga el estado guardado."""
    if not STATE_FILE.exists():
        return None
    with open(STATE_FILE) as f:
        return json.load(f)

def save_state(state):
    """Guarda el estado."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def detect_amnesia(saved_state, current_status):
    """Detecta si hubo amnesia comparando estados."""
    if not saved_state or not current_status:
        return False, "No hay estado previo para comparar"
    
    reasons = []
    
    # Compactación
    saved_compactions = saved_state.get("lastCompactionCount", 0)
    current_compactions = current_status.get("compactionCount", 0)
    if current_compactions > saved_compactions:
        reasons.append(f"⚠️ Compactación detectada: {saved_compactions} → {current_compactions}")
    
    # Cambio de modelo
    saved_model = saved_state.get("lastModel", "")
    current_model = current_status.get("model", "")
    if saved_model and current_model and saved_model != current_model:
        reasons.append(f"⚠️ Modelo cambió: {saved_model} → {current_model}")
    
    # Context tokens muy bajo vs anterior (posible reset)
    saved_context = saved_state.get("lastContextPercent", 0)
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"a "**********"v "**********"e "**********"d "**********"_ "**********"c "**********"o "**********"n "**********"t "**********"e "**********"x "**********"t "**********"  "**********"> "**********"  "**********"5 "**********"0 "**********"  "**********"a "**********"n "**********"d "**********"  "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********"_ "**********"s "**********"t "**********"a "**********"t "**********"u "**********"s "**********". "**********"g "**********"e "**********"t "**********"( "**********"" "**********"c "**********"o "**********"n "**********"t "**********"e "**********"x "**********"t "**********"T "**********"o "**********"k "**********"e "**********"n "**********"s "**********"" "**********", "**********"  "**********"0 "**********") "**********"  "**********"< "**********"  "**********"1 "**********"0 "**********"0 "**********"0 "**********"0 "**********": "**********"
        reasons.append(f"⚠️ Context tokens muy bajo ({current_status.get('contextTokens', 0)}) vs anterior ({saved_context}%)")
    
    if reasons:
        return True, "\n".join(reasons)
    return False, "✅ No se detecta amnesia"

def get_recent_history(session_file, limit=20):
    """Lee los últimos N mensajes del historial."""
    if not session_file or not Path(session_file).exists():
        return []
    
    messages = []
    with open(session_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "message":
                    msg = entry.get("message", {})
                    role = msg.get("role", "unknown")
                    content_arr = msg.get("content", [])
                    
                    # Extraer texto de content array
                    content = ""
                    if isinstance(content_arr, list):
                        for c in content_arr:
                            if isinstance(c, dict) and c.get("type") == "text":
                                text = c.get("text", "")
                                # Ignorar textos que parecen ser JSON dumps (resultados de tools)
                                if text and not text.startswith("{") and not text.startswith("["):
                                    content += text + " "
                            elif isinstance(c, str):
                                if not c.startswith("{") and not c.startswith("["):
                                    content += c + " "
                    elif isinstance(content_arr, str):
                        if not content_arr.startswith("{") and not content_arr.startswith("["):
                            content = content_arr
                    
                    content = content.strip()
                    # Filtrar mensajes vacíos y mensajes del sistema
                    if content and role in ("user", "assistant") and len(content) > 10:
                        # Limpiar message_id si está presente
                        if "[message_id:" in content:
                            content = content.split("[message_id:")[0].strip()
                        messages.append({
                            "role": role,
                            "content": content[:500] + "..." if len(content) > 500 else content,
                            "timestamp": entry.get("timestamp", 0)
                        })
            except json.JSONDecodeError:
                continue
    
    return messages[-limit:]

def format_recovery_summary(messages):
    """Formatea un resumen para recuperación."""
    if not messages:
        return "No hay mensajes recientes para mostrar."
    
    lines = ["## Últimos mensajes (para recuperar contexto)\n"]
    for msg in messages:
        role = "👤 Usuario" if msg["role"] == "user" else "🤖 Ava"
        ts = ""
        if msg.get("timestamp"):
            try:
                ts = datetime.fromtimestamp(msg["timestamp"]/1000).strftime(" (%H:%M)")
            except:
                pass
        lines.append(f"**{role}**{ts}: {msg['content']}\n")
    
    return "\n".join(lines)

def get_today_messages(session_file):
    """Cuenta mensajes de hoy."""
    if not session_file or not Path(session_file).exists():
        return 0
    
    today = datetime.now().strftime("%Y-%m-%d")
    count = 0
    with open(session_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "message":
                    ts = entry.get("timestamp", 0)
                    if ts:
                        # Timestamp puede ser string o int
                        if isinstance(ts, str):
                            ts = int(ts) if ts.isdigit() else 0
                        if ts > 0:
                            entry_date = datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d")
                            if entry_date == today:
                                count += 1
            except (json.JSONDecodeError, ValueError, OSError):
                continue
    return count

def count_all_messages(session_file):
    """Cuenta todos los mensajes."""
    if not session_file or not Path(session_file).exists():
        return 0
    
    count = 0
    with open(session_file) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("type") == "message":
                    count += 1
            except json.JSONDecodeError:
                continue
    return count

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detector de amnesia y recuperación de contexto")
    parser.add_argument("--recover", action="store_true", help="Mostrar historial para recuperar")
    parser.add_argument("--update", action="store_true", help="Actualizar estado guardado")
    parser.add_argument("--status", action="store_true", help="Solo mostrar estado")
    parser.add_argument("--reset", action="store_true", help="Reset estado (usar antes de /new para empezar limpio)")
    parser.add_argument("--load", type=str, default=None, help="Cargar historial: número, 'today', o 'all'")
    parser.add_argument("--history", type=int, default=200, help="Número de mensajes a mostrar")
    parser.add_argument("--session", type=str, default=None, help="Session key (auto-detecta si no se especifica)")
    args = parser.parse_args()
    
    # Reset: borrar estado para empezar limpio
    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            print("🧹 Estado borrado. Próxima sesión empezará limpia.")
        else:
            print("ℹ️ No había estado guardado.")
        return
    
    current = get_session_status(args.session)
    saved = load_state()
    
    # Mostrar qué sesión se está usando
    if current:
        print(f"📍 Sesión detectada: {current.get('sessionKey', 'unknown')}")
    
    # Load: cargar historial forzado (sin detectar amnesia)
    if args.load:
        session_file = current.get("sessionFile", "") if current else ""
        if not session_file:
            print("❌ No se encontró archivo de sesión")
            return
        
        if args.load.lower() == "today":
            # Contar mensajes de hoy
            today_count = get_today_messages(session_file)
            limit = max(today_count, 50)  # Mínimo 50 por si hay pocos
            print(f"📅 Cargando mensajes de hoy (~{today_count} mensajes)")
        elif args.load.lower() == "all":
            limit = count_all_messages(session_file)
            print(f"📚 Cargando historial completo ({limit} mensajes)")
        else:
            try:
                limit = int(args.load)
                print(f"📜 Cargando últimos {limit} mensajes")
            except ValueError:
                print(f"❌ Valor inválido: {args.load}. Usa número, 'today', o 'all'")
                return
        
        print("\n" + "="*50)
        messages = get_recent_history(session_file, limit)
        print(format_recovery_summary(messages))
        print("="*50)
        print(f"\n📊 Mostrados: {len(messages)} mensajes")
        return
    
    if args.status:
        print("=== Estado Guardado ===")
        print(json.dumps(saved, indent=2) if saved else "No hay estado guardado")
        print("\n=== Estado Actual ===")
        print(json.dumps(current, indent=2) if current else "No se pudo obtener")
        return
    
    if args.update:
        if current:
            new_state = {
                "lastMessageId": "manual-update",
                "lastCompactionCount": current.get("compactionCount", 0),
                "lastModel": current.get("model", "unknown"),
                "lastContextPercent": 0,  # No tenemos % aquí
                "lastTimestamp": int(datetime.now().timestamp() * 1000),
                "sessionKey": current.get("sessionKey", "agent:main:main"),
                "note": "Actualizado manualmente post-recuperación"
            }
            save_state(new_state)
            print("✅ Estado actualizado")
            print(json.dumps(new_state, indent=2))
        else:
            print("❌ No se pudo obtener estado actual")
        return
    
    # Detección de amnesia
    amnesia, reason = detect_amnesia(saved, current)
    print(reason)
    
    if amnesia or args.recover:
        print("\n" + "="*50)
        session_file = current.get("sessionFile", "") if current else ""
        messages = get_recent_history(session_file, args.history)
        print(format_recovery_summary(messages))
        print("="*50)
        print("\n💡 Tras recuperar contexto, ejecuta: python3 memory/_recover_context.py --update")

if __name__ == "__main__":
    main()