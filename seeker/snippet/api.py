#date: 2026-02-06T17:21:42Z
#url: https://api.github.com/gists/c6c9e1cb15ad168e130e395ef067c3c7
#owner: https://api.github.com/users/debostic

#!/usr/bin/env python3
"""
Dashboard v4 API — Real data from OpenClaw filesystem
Port: 5556
"""
import os
import json
import subprocess
import glob
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from collections import defaultdict
import re

app = Flask(__name__)
CORS(app)

# Serve index.html at root
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Paths
OPENCLAW_HOME = Path.home() / ".openclaw"
SESSIONS_DIR = OPENCLAW_HOME / "agents" / "main" / "sessions"
CRON_JOBS = OPENCLAW_HOME / "cron" / "jobs.json"
CRON_RUNS_DIR = OPENCLAW_HOME / "cron" / "runs"
WATCHDOG_STATE = OPENCLAW_HOME / "logs" / "watchdog-state.json"
MEMORY_DIR = Path.home() / ".openclaw" / "workspace" / "memory"
MEMORY_FILE = Path.home() / ".openclaw" / "workspace" / "MEMORY.md"
WORKSPACE_DIR = Path.home() / ".openclaw" / "workspace"

# Cache for JSONL parsing
jsonl_cache = {}

def get_file_size(path):
    """Get file size, return 0 if doesn't exist"""
    try:
        return os.path.getsize(path)
    except:
        return 0

def parse_jsonl_incremental(path, cache_key=None):
    """Parse JSONL file incrementally (only new lines since last read)"""
    if cache_key is None:
        cache_key = str(path)
    
    current_size = get_file_size(path)
    cached = jsonl_cache.get(cache_key, {"size": 0, "lines": []})
    
    if current_size == cached["size"]:
        return cached["lines"]
    
    # File changed, re-parse from scratch (simple approach)
    lines = []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    
    jsonl_cache[cache_key] = {"size": current_size, "lines": lines}
    return lines

def get_gateway_uptime():
    """Get gateway uptime from ps output"""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,lstart,command"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if 'openclaw-gateway' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 6:
                    # Extract start time (format: Thu Feb  5 13:49:57 2026)
                    try:
                        start_str = ' '.join(parts[1:6])
                        start_time = datetime.strptime(start_str, '%a %b %d %H:%M:%S %Y')
                        uptime_seconds = (datetime.now() - start_time).total_seconds()
                        return int(uptime_seconds)
                    except:
                        pass
                break
    except:
        pass
    return None

def format_uptime(seconds):
    """Format uptime in human readable form"""
    if seconds is None:
        return "unknown"
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    mins = (seconds % 3600) // 60
    if days > 0:
        return f"{int(days)}d {int(hours)}h"
    elif hours > 0:
        return f"{int(hours)}h {int(mins)}m"
    else:
        return f"{int(mins)}m"

@app.route('/api/status')
def api_status():
    """Gateway health, uptime, RAM, disk, active sessions, current model"""
    # Load watchdog state
    watchdog = {}
    try:
        with open(WATCHDOG_STATE, 'r') as f:
            watchdog = json.load(f)
    except:
        pass
    
    # Get uptime
    uptime_seconds = get_gateway_uptime()
    
    # Count active sessions (session files modified in last hour)
    active_sessions = 0
    try:
        now = datetime.now().timestamp()
        for session_file in SESSIONS_DIR.glob("*.jsonl"):
            if now - session_file.stat().st_mtime < 3600:
                active_sessions += 1
    except:
        pass
    
    # Get last activity from most recent JSONL entry
    last_activity = None
    try:
        all_sessions = sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
        for session_file in all_sessions[:5]:  # Check 5 most recent
            lines = parse_jsonl_incremental(session_file)
            if lines:
                last_msg = lines[-1]
                if 'timestamp' in last_msg:
                    last_activity = last_msg['timestamp']
                    break
    except:
        pass
    
    return jsonify({
        "status": watchdog.get("status", "unknown"),
        "uptime": format_uptime(uptime_seconds),
        "uptimeSeconds": uptime_seconds,
        "memory": f"{watchdog.get('gatewayMemoryMB', 0)}MB",
        "disk": watchdog.get("details", "").split("disk=")[1].split()[0] if "disk=" in watchdog.get("details", "") else "unknown",
        "activeSessions": active_sessions,
        "currentModel": "anthropic/claude-opus-4-6",  # From runtime context
        "lastActivity": last_activity,
        "gatewayPid": watchdog.get("gatewayPid"),
        "lastCheck": watchdog.get("lastCheck")
    })

@app.route('/api/activity')
def api_activity():
    """Recent activity from JSONL transcripts"""
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    filter_type = request.args.get('filter', 'all')  # all, tool, user, assistant
    
    activities = []
    
    # Get all session files sorted by modification time
    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    for session_file in session_files:
        lines = parse_jsonl_incremental(session_file)
        session_id = session_file.stem
        
        for entry in reversed(lines):  # Newest first
            if entry.get('type') != 'message':
                continue
            
            msg = entry.get('message', {})
            role = msg.get('role')
            timestamp = entry.get('timestamp')
            model = msg.get('model', 'unknown')
            content = msg.get('content', [])
            
            # Extract text and tool calls
            text_parts = []
            tool_calls = []
            
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'toolCall':
                        tool_calls.append({
                            'name': item.get('name'),
                            'id': item.get('id')
                        })
            
            text = ' '.join(text_parts)[:200]  # Truncate
            
            # Filter
            if filter_type == 'tool' and not tool_calls:
                continue
            elif filter_type == 'user' and role != 'user':
                continue
            elif filter_type == 'assistant' and role != 'assistant':
                continue
            
            # Build activity entry
            activity = {
                'timestamp': timestamp,
                'session': session_id,
                'role': role,
                'model': model,
                'text': text,
                'tools': [t['name'] for t in tool_calls],
                'cost': msg.get('usage', {}).get('cost', {}).get('total', 0)
            }
            
            activities.append(activity)
            
            if len(activities) >= offset + limit:
                break
        
        if len(activities) >= offset + limit:
            break
    
    return jsonify({
        'activities': activities[offset:offset+limit],
        'total': len(activities)
    })

@app.route('/api/crons')
def api_crons():
    """All cron jobs with schedule, status, next run, last run"""
    try:
        with open(CRON_JOBS, 'r') as f:
            data = json.load(f)
            jobs = data.get('jobs', [])
            
            # Enhance with run history
            for job in jobs:
                job_id = job['id']
                run_file = CRON_RUNS_DIR / f"{job_id}.jsonl"
                
                if run_file.exists():
                    runs = parse_jsonl_incremental(run_file)
                    # Get last 5 runs
                    finished_runs = [r for r in runs if r.get('action') == 'finished']
                    job['recentRuns'] = finished_runs[-5:] if finished_runs else []
                else:
                    job['recentRuns'] = []
            
            return jsonify({'jobs': jobs})
    except Exception as e:
        return jsonify({'error': str(e), 'jobs': []}), 500

@app.route('/api/search')
def api_search():
    """Full-text search across memory, transcripts, workspace files"""
    query = request.args.get('q', '').lower()
    search_type = request.args.get('type', 'all')  # all, memory, transcripts, workspace
    
    if not query:
        return jsonify({'results': []})
    
    results = []
    
    # Search memory files
    if search_type in ['all', 'memory']:
        for mem_file in MEMORY_DIR.glob("*.md"):
            try:
                with open(mem_file, 'r') as f:
                    content = f.read()
                    if query in content.lower():
                        # Find context around match
                        lines = content.split('\n')
                        matches = []
                        for i, line in enumerate(lines):
                            if query in line.lower():
                                start = max(0, i-2)
                                end = min(len(lines), i+3)
                                context = '\n'.join(lines[start:end])
                                matches.append({
                                    'line': i+1,
                                    'context': context[:300]
                                })
                        
                        results.append({
                            'type': 'memory',
                            'file': mem_file.name,
                            'path': str(mem_file),
                            'matches': matches[:3]  # Top 3 matches
                        })
            except:
                pass
    
    # Search MEMORY.md
    if search_type in ['all', 'memory']:
        try:
            with open(MEMORY_FILE, 'r') as f:
                content = f.read()
                if query in content.lower():
                    lines = content.split('\n')
                    matches = []
                    for i, line in enumerate(lines):
                        if query in line.lower():
                            start = max(0, i-2)
                            end = min(len(lines), i+3)
                            context = '\n'.join(lines[start:end])
                            matches.append({
                                'line': i+1,
                                'context': context[:300]
                            })
                    
                    results.append({
                        'type': 'memory',
                        'file': 'MEMORY.md',
                        'path': str(MEMORY_FILE),
                        'matches': matches[:3]
                    })
        except:
            pass
    
    # Search transcripts
    if search_type in ['all', 'transcripts']:
        for session_file in list(SESSIONS_DIR.glob("*.jsonl"))[:20]:  # Limit to 20 most recent
            lines = parse_jsonl_incremental(session_file)
            matches = []
            
            for entry in lines:
                if entry.get('type') != 'message':
                    continue
                
                msg = entry.get('message', {})
                content = msg.get('content', [])
                
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text = item.get('text', '')
                        if query in text.lower():
                            matches.append({
                                'timestamp': entry.get('timestamp'),
                                'role': msg.get('role'),
                                'text': text[:300]
                            })
            
            if matches:
                results.append({
                    'type': 'transcript',
                    'file': session_file.name,
                    'path': str(session_file),
                    'matches': matches[:3]
                })
    
    # Search workspace files
    if search_type in ['all', 'workspace']:
        for pattern in ['*.md', '*.py', '*.sh', '*.json']:
            for ws_file in WORKSPACE_DIR.rglob(pattern):
                if 'node_modules' in str(ws_file) or '.git' in str(ws_file):
                    continue
                
                try:
                    with open(ws_file, 'r') as f:
                        content = f.read()
                        if query in content.lower():
                            lines = content.split('\n')
                            matches = []
                            for i, line in enumerate(lines):
                                if query in line.lower():
                                    start = max(0, i-2)
                                    end = min(len(lines), i+3)
                                    context = '\n'.join(lines[start:end])
                                    matches.append({
                                        'line': i+1,
                                        'context': context[:300]
                                    })
                            
                            results.append({
                                'type': 'workspace',
                                'file': ws_file.name,
                                'path': str(ws_file.relative_to(WORKSPACE_DIR)),
                                'matches': matches[:3]
                            })
                except:
                    pass
    
    return jsonify({
        'results': results[:50],  # Limit results
        'total': len(results)
    })

@app.route('/api/costs')
def api_costs():
    """Cost breakdown by model, session, day"""
    period = request.args.get('period', 'week')  # today, week, month
    
    # Calculate date range
    now = datetime.now()
    if period == 'today':
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'week':
        start_date = now - timedelta(days=7)
    elif period == 'month':
        start_date = now - timedelta(days=30)
    else:
        start_date = now - timedelta(days=7)
    
    # Aggregate costs
    by_model = defaultdict(float)
    by_session = defaultdict(float)
    by_day = defaultdict(float)
    total = 0.0
    
    for session_file in SESSIONS_DIR.glob("*.jsonl"):
        lines = parse_jsonl_incremental(session_file)
        session_id = session_file.stem
        
        for entry in lines:
            if entry.get('type') != 'message':
                continue
            
            timestamp = entry.get('timestamp')
            if timestamp:
                try:
                    msg_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if msg_time < start_date:
                        continue
                except:
                    continue
            
            msg = entry.get('message', {})
            model = msg.get('model', 'unknown')
            cost = msg.get('usage', {}).get('cost', {}).get('total', 0)
            
            if cost > 0:
                by_model[model] += cost
                by_session[session_id] += cost
                day_key = msg_time.strftime('%Y-%m-%d')
                by_day[day_key] += cost
                total += cost
    
    # Sort by value
    by_model = dict(sorted(by_model.items(), key=lambda x: x[1], reverse=True))
    by_session = dict(sorted(by_session.items(), key=lambda x: x[1], reverse=True)[:10])  # Top 10
    by_day = dict(sorted(by_day.items()))
    
    return jsonify({
        'total': round(total, 4),
        'byModel': {k: round(v, 4) for k, v in by_model.items()},
        'bySession': {k: round(v, 4) for k, v in by_session.items()},
        'byDay': {k: round(v, 4) for k, v in by_day.items()},
        'period': period
    })

@app.route('/api/memory')
def api_memory():
    """List memory files with previews"""
    files = []
    
    for mem_file in sorted(MEMORY_DIR.glob("*.md"), reverse=True):
        try:
            with open(mem_file, 'r') as f:
                content = f.read()
                preview = content[:500]  # First 500 chars
                
                files.append({
                    'name': mem_file.name,
                    'date': mem_file.stem,
                    'size': len(content),
                    'preview': preview,
                    'modified': datetime.fromtimestamp(mem_file.stat().st_mtime).isoformat()
                })
        except:
            pass
    
    return jsonify({'files': files})

@app.route('/api/memory/<date>')
def api_memory_content(date):
    """Full content of a memory file"""
    mem_file = MEMORY_DIR / f"{date}.md"
    
    if not mem_file.exists():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        with open(mem_file, 'r') as f:
            content = f.read()
            return jsonify({
                'date': date,
                'content': content,
                'size': len(content)
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions')
def api_sessions():
    """Active and recent sessions with metadata"""
    sessions = []
    
    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    for session_file in session_files[:20]:  # Last 20 sessions
        lines = parse_jsonl_incremental(session_file)
        
        # Find session metadata
        session_meta = None
        for entry in lines:
            if entry.get('type') == 'session':
                session_meta = entry
                break
        
        # Count messages
        message_count = sum(1 for e in lines if e.get('type') == 'message')
        
        # Get last activity
        last_activity = None
        last_model = None
        total_tokens = "**********"
        total_cost = 0.0
        
        for entry in reversed(lines):
            if entry.get('type') == 'message':
                if not last_activity:
                    last_activity = entry.get('timestamp')
                    last_model = entry.get('message', {}).get('model')
                
                usage = entry.get('message', {}).get('usage', {})
                total_tokens += "**********"
                total_cost += usage.get('cost', {}).get('total', 0)
        
        # Is it active? (modified in last hour)
        is_active = (datetime.now().timestamp() - session_file.stat().st_mtime) < 3600
        
        sessions.append({
            'id': session_file.stem,
            'active': is_active,
            'messageCount': message_count,
            'lastActivity': last_activity,
            'model': last_model,
            'totalTokens': "**********"
            'totalCost': round(total_cost, 4),
            'created': session_meta.get('timestamp') if session_meta else None
        })
    
    return jsonify({'sessions': sessions})

@app.route('/api/health')
def api_health():
    """Watchdog data + historical if available"""
    try:
        with open(WATCHDOG_STATE, 'r') as f:
            current = json.load(f)
    except:
        current = {}
    
    # Parse watchdog log for history
    history = []
    log_file = OPENCLAW_HOME / "logs" / "watchdog.log"
    
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-50:]:  # Last 50 log entries
                    # Parse log format: [timestamp] status: details
                    match = re.match(r'\[(.*?)\]\s+(\w+):\s+(.*)', line)
                    if match:
                        timestamp, status, details = match.groups()
                        history.append({
                            'timestamp': timestamp,
                            'status': status,
                            'details': details
                        })
        except:
            pass
    
    return jsonify({
        'current': current,
        'history': history
    })

if __name__ == '__main__':
    print("🚀 Dashboard v4 API starting on port 5556...")
    print(f"📁 Sessions: {SESSIONS_DIR}")
    print(f"📁 Memory: {MEMORY_DIR}")
    print(f"📁 Cron: {CRON_JOBS}")
    app.run(host='0.0.0.0', port=5556, debug=True)
rue)
