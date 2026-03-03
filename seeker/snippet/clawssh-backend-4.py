#date: 2026-03-03T17:29:29Z
#url: https://api.github.com/gists/336d1d44a8f09df817f80775fcc379e9
#owner: https://api.github.com/users/edelah

#!/usr/bin/env python3

import argparse
import base64
import importlib
import importlib.util
import json
import os
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


EMBEDDED_BACKEND_MODULES = {
    'auth.py': "import argparse\nimport hashlib\nimport http.server\nimport json\nimport os\nimport re\nimport stat\nimport subprocess\nimport ssl\nimport tempfile\nimport time\nfrom datetime import datetime, timezone\nfrom pathlib import Path\n\n\nSTATE_DIR_NAME = '.clawssh'\nDEVICES_FILE_NAME = 'devices.json'\nCA_CERT_FILE_NAME = 'ca.crt'\nCA_KEY_FILE_NAME = 'ca.key'\nSERVER_CERT_FILE_NAME = 'server.crt'\nSERVER_KEY_FILE_NAME = 'server.key'\n\n\ndef default_state_dir() -> Path:\n    return Path(os.getenv('CLAWSSH_STATE_DIR', Path.home() / STATE_DIR_NAME)).expanduser()\n\n\ndef utc_now_iso() -> str:\n    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')\n\n\ndef ensure_state_dir(state_dir: Path | None = None) -> Path:\n    path = Path(state_dir or default_state_dir()).expanduser()\n    path.mkdir(parents=True, exist_ok=True)\n    os.chmod(path, 0o700)\n    return path\n\n\ndef devices_file_path(state_dir: Path | None = None) -> Path:\n    return ensure_state_dir(state_dir) / DEVICES_FILE_NAME\n\n\ndef ca_cert_file_path(state_dir: Path | None = None) -> Path:\n    return ensure_state_dir(state_dir) / CA_CERT_FILE_NAME\n\n\ndef ca_key_file_path(state_dir: Path | None = None) -> Path:\n    return ensure_state_dir(state_dir) / CA_KEY_FILE_NAME\n\n\ndef server_cert_file_path(state_dir: Path | None = None) -> Path:\n    return ensure_state_dir(state_dir) / SERVER_CERT_FILE_NAME\n\n\ndef server_key_file_path(state_dir: Path | None = None) -> Path:\n    return ensure_state_dir(state_dir) / SERVER_KEY_FILE_NAME\n\n\ndef device_dir_path(device_id: str, state_dir: Path | None = None) -> Path:\n    normalized = (device_id or '').strip()\n    if not normalized:\n        raise RuntimeError('Device ID is required.')\n    path = ensure_state_dir(state_dir) / 'devices' / normalized\n    path.mkdir(parents=True, exist_ok=True)\n    os.chmod(path, 0o700)\n    return path\n\n\ndef ensure_devices_file(state_dir: Path | None = None) -> Path:\n    path = devices_file_path(state_dir)\n    if not path.exists():\n        payload = {\n            'version': 1,\n            'devices': [],\n        }\n        _write_json_atomic(path, payload)\n    os.chmod(path, 0o600)\n    return path\n\n\ndef load_devices(state_dir: Path | None = None) -> dict:\n    path = ensure_devices_file(state_dir)\n    with path.open('r', encoding='utf-8') as handle:\n        payload = json.load(handle)\n    if not isinstance(payload, dict):\n        raise RuntimeError(f'Invalid device registry: {path}')\n    devices = payload.get('devices')\n    if not isinstance(devices, list):\n        raise RuntimeError(f'Invalid device registry device list: {path}')\n    payload.setdefault('version', 1)\n    return payload\n\n\ndef save_devices(payload: dict, state_dir: Path | None = None) -> Path:\n    path = ensure_devices_file(state_dir)\n    _write_json_atomic(path, payload)\n    os.chmod(path, 0o600)\n    return path\n\n\ndef certificate_fingerprint_from_der(certificate_der: bytes) -> str:\n    digest = hashlib.sha256(certificate_der).hexdigest().upper()\n    return ':'.join(digest[i:i + 2] for i in range(0, len(digest), 2))\n\n\ndef certificate_fingerprint_from_pem_file(cert_path: str | os.PathLike[str]) -> str:\n    import ssl\n\n    pem_text = Path(cert_path).read_text(encoding='utf-8')\n    certificate_der = ssl.PEM_cert_to_DER_cert(pem_text)\n    return certificate_fingerprint_from_der(certificate_der)\n\n\ndef find_device(payload: dict, device_id: str) -> dict | None:\n    normalized = (device_id or '').strip()\n    if not normalized:\n        return None\n    for device in payload.get('devices', []):\n        if device.get('client_id') == normalized:\n            return device\n    return None\n\n\ndef register_device(\n    device_id: str,\n    certificate_path: str | os.PathLike[str],\n    state_dir: Path | None = None,\n    invitation_code: str = '',\n) -> dict:\n    payload = load_devices(state_dir)\n    normalized = (device_id or '').strip()\n    if not normalized:\n        raise RuntimeError('Device ID is required.')\n\n    existing = find_device(payload, normalized)\n    if existing and existing.get('status') != 'revoked':\n        raise RuntimeError(f'Device already enrolled: {normalized}')\n\n    fingerprint = certificate_fingerprint_from_pem_file(certificate_path)\n    now = utc_now_iso()\n    record = {\n        'client_id': normalized,\n        'fingerprint': fingerprint,\n        'enrolled_at': now,\n        'last_connection_at': existing.get('last_connection_at', '') if existing else '',\n        'last_remote_ip': existing.get('last_remote_ip', '') if existing else '',\n        'status': 'active',\n    }\n    devices = payload.setdefault('devices', [])\n    replaced = False\n    for index, device in enumerate(devices):\n        if device.get('client_id') == normalized:\n            devices[index] = record\n            replaced = True\n            break\n    if not replaced:\n        devices.append(record)\n    devices.sort(key=lambda item: item.get('client_id', ''))\n    save_devices(payload, state_dir)\n    return record\n\n\ndef register_device_pem(\n    device_id: str,\n    certificate_pem: str,\n    state_dir: Path | None = None,\n    invitation_code: str = '',\n) -> dict:\n    cert_file = device_dir_path(device_id, state_dir) / 'client.crt'\n    cert_file.write_text(certificate_pem, encoding='utf-8')\n    os.chmod(cert_file, 0o600)\n    return register_device(device_id, cert_file, state_dir, invitation_code)\n\n\ndef revoke_device(device_id: str, state_dir: Path | None = None) -> dict:\n    payload = load_devices(state_dir)\n    record = find_device(payload, device_id)\n    if record is None:\n        raise RuntimeError(f'Unknown device: {device_id}')\n    record['status'] = 'revoked'\n    record['revoked_at'] = utc_now_iso()\n    save_devices(payload, state_dir)\n    return record\n\n\ndef resolve_device_for_certificate(certificate_der: bytes, state_dir: Path | None = None) -> tuple[dict | None, str]:\n    payload = load_devices(state_dir)\n    fingerprint = certificate_fingerprint_from_der(certificate_der)\n    for record in payload.get('devices', []):\n        if record.get('fingerprint') != fingerprint:\n            continue\n        status = record.get('status', '')\n        if status == 'revoked':\n            return None, 'Revoked Device'\n        if status != 'active':\n            return None, 'Inactive Device'\n        return record, 'Authorized'\n    return None, 'Unknown Certificate'\n\n\ndef mark_device_connection(device_id: str, remote_ip: str, state_dir: Path | None = None) -> dict:\n    payload = load_devices(state_dir)\n    record = find_device(payload, device_id)\n    if record is None:\n        raise RuntimeError(f'Unknown device: {device_id}')\n    record['last_connection_at'] = utc_now_iso()\n    record['last_remote_ip'] = remote_ip\n    save_devices(payload, state_dir)\n    return record\n\n\ndef authorize_websocket_peer(websocket, state_dir: Path | None = None) -> tuple[dict | None, str]:\n    transport = getattr(websocket, 'transport', None)\n    if transport is None:\n        return None, 'Missing TLS transport'\n    ssl_object = transport.get_extra_info('ssl_object')\n    if ssl_object is None:\n        return None, 'Missing Client Certificate'\n    certificate_der = ssl_object.getpeercert(binary_form=True)\n    if not certificate_der:\n        return None, 'Missing Client Certificate'\n    return resolve_device_for_certificate(certificate_der, state_dir)\n\n\ndef verify_permissions(state_dir: Path | None = None) -> list[str]:\n    errors: list[str] = []\n    state_path = ensure_state_dir(state_dir)\n    mode = stat.S_IMODE(state_path.stat().st_mode)\n    if mode != 0o700:\n        errors.append(f'{state_path} permissions are {oct(mode)}; expected 0o700')\n    registry_path = ensure_devices_file(state_path)\n    registry_mode = stat.S_IMODE(registry_path.stat().st_mode)\n    if registry_mode != 0o600:\n        errors.append(f'{registry_path} permissions are {oct(registry_mode)}; expected 0o600')\n    return errors\n\n\ndef format_devices_table(payload: dict) -> str:\n    headers = ['Device ID', 'Status', 'Enrolled', 'Last Connection', 'Last Remote IP']\n    rows = []\n    for device in payload.get('devices', []):\n        rows.append(\n            [\n                str(device.get('client_id', '')),\n                str(device.get('status', '')),\n                str(device.get('enrolled_at', '')),\n                str(device.get('last_connection_at', '')),\n                str(device.get('last_remote_ip', '')),\n            ]\n        )\n    widths = [len(header) for header in headers]\n    for row in rows:\n        for index, value in enumerate(row):\n            widths[index] = max(widths[index], len(value))\n    lines = ['  '.join(header.ljust(widths[index]) for index, header in enumerate(headers))]\n    lines.append('  '.join('-' * width for width in widths))\n    for row in rows:\n        lines.append('  '.join(value.ljust(widths[index]) for index, value in enumerate(row)))\n    return '\\n'.join(lines)\n\n\ndef issue_client_certificate_from_csr(\n    device_id: str,\n    csr_pem: str,\n    state_dir: Path | None = None,\n    invitation_code: str = '',\n) -> tuple[dict, str]:\n    state_path = ensure_state_dir(state_dir)\n    ca_cert = ca_cert_file_path(state_path)\n    ca_key = ca_key_file_path(state_path)\n    if not ca_cert.is_file() or not ca_key.is_file():\n        raise RuntimeError('CA certificate/key are missing. Run backend/clawssh.py setup first.')\n\n    work_dir = Path(tempfile.mkdtemp(prefix='clawssh-csr-', dir=state_path))\n    csr_file = work_dir / 'client.csr'\n    cert_file = work_dir / 'client.crt'\n    ext_file = work_dir / 'client.ext'\n    csr_file.write_text(csr_pem, encoding='utf-8')\n    ext_file.write_text(\n        '\\n'.join(\n            [\n                'basicConstraints=CA:FALSE',\n                'keyUsage=digitalSignature,keyEncipherment',\n                'extendedKeyUsage=clientAuth',\n                'subjectKeyIdentifier=hash',\n                'authorityKeyIdentifier=keyid,issuer',\n            ]\n        )\n        + '\\n',\n        encoding='utf-8',\n    )\n    try:\n        subprocess.run(\n            ['openssl', 'req', '-in', str(csr_file), '-verify', '-noout'],\n            check=True,\n            capture_output=True,\n            text=True,\n        )\n        subject_result = subprocess.run(\n            ['openssl', 'req', '-in', str(csr_file), '-noout', '-subject', '-nameopt', 'RFC2253'],\n            check=True,\n            capture_output=True,\n            text=True,\n        )\n        subject_text = subject_result.stdout.strip()\n        subject_match = re.search(r'CN=([^,]+)', subject_text)\n        if subject_match and subject_match.group(1).strip() != device_id:\n            raise RuntimeError('CSR subject CN does not match the requested device ID.')\n\n        subprocess.run(\n            [\n                'openssl',\n                'x509',\n                '-req',\n                '-in',\n                str(csr_file),\n                '-CA',\n                str(ca_cert),\n                '-CAkey',\n                str(ca_key),\n                '-CAcreateserial',\n                '-out',\n                str(cert_file),\n                '-days',\n                '825',\n                '-sha256',\n                '-extfile',\n                str(ext_file),\n            ],\n            check=True,\n            capture_output=True,\n            text=True,\n        )\n        certificate_pem = cert_file.read_text(encoding='utf-8')\n        record = register_device_pem(device_id, certificate_pem, state_path, invitation_code)\n        return record, certificate_pem\n    finally:\n        for child in work_dir.iterdir():\n            child.unlink(missing_ok=True)\n        work_dir.rmdir()\n\n\ndef serve_single_use_enrollment(\n    *,\n    state_dir: Path,\n    bind_host: str,\n    port: int,\n    expected_invitation_code: str,\n    ws_url: str,\n    suggested_device_id: str = '',\n    timeout_seconds: int = 300,\n) -> dict:\n    state_path = ensure_state_dir(state_dir)\n    server_cert = server_cert_file_path(state_path)\n    server_key = server_key_file_path(state_path)\n    ca_cert = ca_cert_file_path(state_path)\n    if not server_cert.is_file() or not server_key.is_file():\n        raise RuntimeError('Server certificate/key are missing. Run backend/clawssh.py setup first.')\n    if not ca_cert.is_file():\n        raise RuntimeError('CA certificate is missing. Run backend/clawssh.py setup first.')\n\n    shared_state: dict = {\n        'done': False,\n        'result': None,\n    }\n\n    class EnrollmentHandler(http.server.BaseHTTPRequestHandler):\n        def do_POST(self):\n            if self.path != '/enroll':\n                self.send_error(404, 'Not found')\n                return\n\n            length = int(self.headers.get('Content-Length', '0') or '0')\n            raw = self.rfile.read(length)\n            try:\n                payload = json.loads(raw.decode('utf-8'))\n            except Exception:\n                self._send_json(400, {'error': 'Invalid JSON payload'})\n                return\n\n            invitation_code = str(payload.get('invitationCode', '')).strip()\n            device_id = str(payload.get('deviceId', '')).strip() or suggested_device_id.strip()\n            csr_pem = str(payload.get('csrPem', '')).strip()\n            if invitation_code != expected_invitation_code:\n                self._send_json(403, {'error': 'Invalid invitation code'})\n                return\n            if not device_id:\n                self._send_json(400, {'error': 'Missing deviceId'})\n                return\n            if 'BEGIN CERTIFICATE REQUEST' not in csr_pem:\n                self._send_json(400, {'error': 'Missing CSR PEM'})\n                return\n\n            try:\n                record, certificate_pem = issue_client_certificate_from_csr(\n                    device_id=device_id,\n                    csr_pem=csr_pem,\n                    state_dir=state_path,\n                    invitation_code=expected_invitation_code,\n                )\n            except RuntimeError as exc:\n                self._send_json(400, {'error': str(exc)})\n                return\n\n            shared_state['done'] = True\n            shared_state['result'] = record\n            self._send_json(\n                200,\n                {\n                    'deviceId': record['client_id'],\n                    'clientCertificatePem': certificate_pem,\n                    'caCertificatePem': ca_cert.read_text(encoding='utf-8'),\n                    'wsUrl': ws_url,\n                },\n            )\n\n        def log_message(self, format, *args):  # noqa: A003\n            return\n\n        def _send_json(self, status_code: int, payload: dict) -> None:\n            body = json.dumps(payload).encode('utf-8')\n            self.send_response(status_code)\n            self.send_header('Content-Type', 'application/json')\n            self.send_header('Cache-Control', 'no-store')\n            self.send_header('Content-Length', str(len(body)))\n            self.end_headers()\n            self.wfile.write(body)\n\n    server = http.server.ThreadingHTTPServer((bind_host, port), EnrollmentHandler)\n    tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)\n    tls_context.load_cert_chain(certfile=server_cert, keyfile=server_key)\n    server.socket = tls_context.wrap_socket(server.socket, server_side=True)\n    server.timeout = 0.5\n\n    deadline = time.time() + timeout_seconds\n    try:\n        while time.time() < deadline and not shared_state['done']:\n            server.handle_request()\n    finally:\n        server.server_close()\n\n    if not shared_state['done'] or not isinstance(shared_state['result'], dict):\n        raise RuntimeError('Enrollment timed out before a CSR was received.')\n    return shared_state['result']\n\n\ndef _write_json_atomic(path: Path, payload: dict) -> None:\n    ensure_state_dir(path.parent)\n    temp_handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False)\n    try:\n        with temp_handle as handle:\n            json.dump(payload, handle, indent=2, sort_keys=True)\n            handle.write('\\n')\n        os.chmod(temp_handle.name, 0o600)\n        os.replace(temp_handle.name, path)\n    finally:\n        try:\n            if os.path.exists(temp_handle.name):\n                os.unlink(temp_handle.name)\n        except FileNotFoundError:\n            pass\n\n\ndef _main() -> int:\n    parser = argparse.ArgumentParser(description='ClawSSH auth state helpers')\n    parser.add_argument('--state-dir', default=str(default_state_dir()))\n    subparsers = parser.add_subparsers(dest='command', required=True)\n\n    subparsers.add_parser('ensure-state')\n\n    register_parser = subparsers.add_parser('register-device')\n    register_parser.add_argument('--device-id', required=True)\n    register_parser.add_argument('--cert-file', required=True)\n    register_parser.add_argument('--invitation-code', default='')\n\n    list_parser = subparsers.add_parser('list-devices')\n    list_parser.add_argument('--json', action='store_true')\n\n    revoke_parser = subparsers.add_parser('revoke-device')\n    revoke_parser.add_argument('--device-id', required=True)\n\n    enroll_server_parser = subparsers.add_parser('serve-enrollment')\n    enroll_server_parser.add_argument('--bind-host', required=True)\n    enroll_server_parser.add_argument('--port', type=int, required=True)\n    enroll_server_parser.add_argument('--invitation-code', required=True)\n    enroll_server_parser.add_argument('--ws-url', required=True)\n    enroll_server_parser.add_argument('--suggested-device-id', default='')\n    enroll_server_parser.add_argument('--timeout-seconds', type=int, default=300)\n\n    args = parser.parse_args()\n    state_dir = Path(args.state_dir).expanduser()\n\n    if args.command == 'ensure-state':\n        ensure_state_dir(state_dir)\n        ensure_devices_file(state_dir)\n        for error in verify_permissions(state_dir):\n            raise RuntimeError(error)\n        return 0\n\n    if args.command == 'register-device':\n        record = register_device(\n            device_id=args.device_id,\n            certificate_path=args.cert_file,\n            state_dir=state_dir,\n            invitation_code=args.invitation_code,\n        )\n        print(json.dumps(record, indent=2, sort_keys=True))\n        return 0\n\n    if args.command == 'list-devices':\n        payload = load_devices(state_dir)\n        if args.json:\n            print(json.dumps(payload, indent=2, sort_keys=True))\n        else:\n            print(format_devices_table(payload))\n        return 0\n\n    if args.command == 'revoke-device':\n        record = revoke_device(args.device_id, state_dir)\n        print(json.dumps(record, indent=2, sort_keys=True))\n        return 0\n\n    if args.command == 'serve-enrollment':\n        record = serve_single_use_enrollment(\n            state_dir=state_dir,\n            bind_host=args.bind_host,\n            port=args.port,\n            expected_invitation_code=args.invitation_code,\n            ws_url=args.ws_url,\n            suggested_device_id=args.suggested_device_id,\n            timeout_seconds=args.timeout_seconds,\n        )\n        print(json.dumps(record, indent=2, sort_keys=True))\n        return 0\n\n    raise RuntimeError(f'Unsupported command: {args.command}')\n\n\nif __name__ == '__main__':\n    try:\n        raise SystemExit(_main())\n    except RuntimeError as exc:\n        print(f'ERROR: {exc}')\n        raise SystemExit(1)\n",
    'config.py': "import os\nimport re\nimport subprocess\n\n\ndef _first_nonempty_line(text):\n    for line in text.splitlines():\n        line = line.strip()\n        if line:\n            return line\n    return ''\n\n\ndef _env(name: str, legacy_name: str | None = None, default: str = '') -> str:\n    value = os.getenv(name, '').strip()\n    if value:\n        return value\n    if legacy_name:\n        legacy_value = os.getenv(legacy_name, '').strip()\n        if legacy_value:\n            return legacy_value\n    return default\n\n\ndef _detect_tailscale_ip():\n    try:\n        output = subprocess.check_output(\n            ['tailscale', 'ip', '-4'],\n            text=True,\n            stderr=subprocess.DEVNULL,\n        )\n        candidate = _first_nonempty_line(output)\n        if candidate:\n            return candidate, 'tailscale-cli'\n    except Exception:\n        pass\n\n    try:\n        output = subprocess.check_output(\n            ['ip', '-o', '-4', 'addr', 'show', 'dev', 'tailscale0'],\n            text=True,\n            stderr=subprocess.DEVNULL,\n        )\n        match = re.search(r'\\binet\\s+(\\d+\\.\\d+\\.\\d+\\.\\d+)/', output)\n        if match:\n            return match.group(1), 'tailscale0'\n    except Exception:\n        pass\n\n    return None, None\n\n\ndef _default_host():\n    env_host = _env('CLAWSSH_HOST', 'TELETERM_HOST')\n    if env_host:\n        return env_host, 'CLAWSSH_HOST'\n\n    env_tailscale_ip = _env('CLAWSSH_TAILSCALE_IP', 'TELETERM_TAILSCALE_IP')\n    if env_tailscale_ip:\n        return env_tailscale_ip, 'CLAWSSH_TAILSCALE_IP'\n\n    detected_ip, source = _detect_tailscale_ip()\n    if detected_ip:\n        return detected_ip, source\n\n    raise RuntimeError(\n        'Unable to resolve a Tailscale IPv4 address. Set CLAWSSH_TAILSCALE_IP '\n        'or CLAWSSH_HOST explicitly (for example your 100.x.y.z address).'\n    )\n\n\nHOST, HOST_SOURCE = _default_host()\nPORT = int(_env('CLAWSSH_PORT', 'TELETERM_PORT', '8765'))\n_tls_enabled_env = _env('CLAWSSH_TLS_ENABLED', 'TELETERM_TLS_ENABLED', 'true').lower()\nALLOW_UNSAFE_NO_AUTH = _env('CLAWSSH_ALLOW_UNSAFE_NO_AUTH', default='false').lower() == 'true'\nSTATE_DIR = os.path.expanduser(_env('CLAWSSH_STATE_DIR', default='~/.clawssh'))\n\nif _tls_enabled_env in ('0', 'false', 'no'):\n    if not ALLOW_UNSAFE_NO_AUTH:\n        raise RuntimeError(\n            'Plain ws:// mode is no longer supported. ClawSSH backend requires TLS '\n            '(wss:// only). Remove CLAWSSH_TLS_ENABLED=false or set '\n            'CLAWSSH_ALLOW_UNSAFE_NO_AUTH=true for local testing.'\n        )\n    TLS_ENABLED = False\nelse:\n    TLS_ENABLED = True\nTLS_CA_CERT_FILE = _env('CLAWSSH_TLS_CA_CERT_FILE', default=os.path.join(STATE_DIR, 'ca.crt'))\nTLS_CERT_FILE = _env('CLAWSSH_TLS_CERT_FILE', 'TELETERM_TLS_CERT_FILE', os.path.join(STATE_DIR, 'server.crt'))\nTLS_KEY_FILE = _env('CLAWSSH_TLS_KEY_FILE', 'TELETERM_TLS_KEY_FILE', os.path.join(STATE_DIR, 'server.key'))\nTMUX_BACKEND_MODE = 'control'\nBACKEND_VERSION = '2026-02-21-resize-ack-1'\n",
    'server.py': "**********": configuration, authentication, tmux bridging,\nsession management, and setup CLI in a single, transparent file.\n"""\n\nimport argparse\nimport asyncio\nimport base64\nimport datetime\nimport errno\nimport fcntl\nimport hashlib\nimport http.server\nimport json\nimport logging\nimport os\nimport pty\nimport re\nimport secrets\nimport select\nimport shutil\nimport signal\nimport socket\nimport ssl\nimport stat\nimport struct\nimport subprocess\nimport sys\nimport tempfile\nimport threading\nimport time\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Dict, List, Set, Optional, Tuple, Any\nfrom urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit\n\ntry:\n    import websockets\nexcept ImportError:\n    websockets = None\n\n# --- CONSTANTS ---\nBACKEND_VERSION = \'2026-03-02-monolithic\'\nDEVICES_FILE_NAME = \'devices.json\'\n\n# --- UTILS ---\ndef utc_now_iso() -> str:\n    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(\'+00:00\', \'Z\')\n\ndef get_env(name: str, default: str = \'\') -> str:\n    return os.getenv(name, default).strip()\n\ndef state_dir() -> Path:\n    return Path(get_env(\'CLAWSSH_STATE_DIR\', \'~/.clawssh\')).expanduser()\n\ndef ensure_state_dir() -> Path:\n    path = state_dir()\n    path.mkdir(parents=True, exist_ok=True)\n    os.chmod(path, 0o700)\n    (path / \'devices\').mkdir(parents=True, exist_ok=True)\n    os.chmod(path / \'devices\', 0o700)\n    return path\n\ndef ca_cert_file() -> Path: return state_dir() / \'ca.crt\'\ndef ca_key_file() -> Path: return state_dir() / \'ca.key\'\ndef server_cert_file() -> Path: return state_dir() / \'server.crt\'\ndef server_key_file() -> Path: return state_dir() / \'server.key\'\ndef port_file() -> Path: return state_dir() / \'backend_port\'\n\ndef current_port() -> int:\n    port_path = port_file()\n    if port_path.is_file():\n        try:\n            return int(port_path.read_text(encoding=\'utf-8\').strip())\n        except ValueError: pass\n    return int(get_env(\'CLAWSSH_PORT\', \'8765\'))\n\n# --- CONFIGURATION ---\ndef detect_tailscale_ip() -> Optional[str]:\n    for cmd in [[\'tailscale\', \'ip\', \'-4\'], [\'ip\', \'-o\', \'-4\', \'addr\', \'show\', \'dev\', \'tailscale0\']]:\n        try:\n            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)\n            if \'tailscale ip\' in \' \'.join(cmd):\n                res = output.strip().splitlines()\n                if res: return res[0].strip()\n            else:\n                match = re.search(r\'\\binet\\s+(\\d+\\.\\d+\\.\\d+\\.\\d+)/\', output)\n                if match: return match.group(1)\n        except Exception: pass\n    return None\n\ndef detect_bind_host() -> str:\n    host = get_env(\'CLAWSSH_HOST\') or get_env(\'CLAWSSH_TAILSCALE_IP\')\n    if host: return host\n    return detect_tailscale_ip() or \'127.0.0.1\'\n\nBIND_HOST = detect_bind_host()\nTLS_ENABLED = get_env(\'CLAWSSH_TLS_ENABLED\', \'true\').lower() in (\'true\', \'1\', \'yes\')\nALLOW_UNSAFE = get_env(\'CLAWSSH_ALLOW_UNSAFE_NO_AUTH\', \'false\').lower() in (\'true\', \'1\', \'yes\')\n\n# --- CRYPTO & PKI ---\ndef ensure_ca():\n    cert_path, key_path = ca_cert_file(), ca_key_file()\n    if cert_path.is_file() and key_path.is_file(): return\n    ensure_state_dir()\n    print("Generating Local CA...")\n    subprocess.run([\'openssl\', \'ecparam\', \'-name\', \'prime256v1\', \'-genkey\', \'-noout\', \'-out\', str(key_path)], check=True)\n    subprocess.run([\'openssl\', \'req\', \'-x509\', \'-new\', \'-nodes\', \'-key\', str(key_path), \'-sha256\', \'-days\', \'3650\', \'-out\', str(cert_path), \'-subj\', \'/CN=ClawSSH Local CA\'], check=True)\n    os.chmod(key_path, 0o600)\n    os.chmod(cert_path, 0o600)\n\ndef generate_server_cert(force=False):\n    cert_path, key_path = server_cert_file(), server_key_file()\n    if not force and cert_path.is_file() and key_path.is_file(): return\n    ensure_ca()\n    print("Generating Server Certificate...")\n    with tempfile.TemporaryDirectory() as tmp:\n        tmp_path = Path(tmp)\n        csr = tmp_path / \'server.csr\'\n        ext = tmp_path / \'server.ext\'\n        subprocess.run([\'openssl\', \'genrsa\', \'-out\', str(key_path), \'2048\'], check=True)\n        subprocess.run([\'openssl\', \'req\', \'-new\', \'-key\', str(key_path), \'-out\', str(csr), \'-subj\', f\'/CN={BIND_HOST}\'], check=True)\n        ext.write_text(f"subjectAltName=IP:{BIND_HOST},DNS:{BIND_HOST}\\nextendedKeyUsage=serverAuth\\n", encoding=\'utf-8\')\n        subprocess.run([\'openssl\', \'x509\', \'-req\', \'-in\', str(csr), \'-CA\', str(ca_cert_file()), \'-CAkey\', str(ca_key_file()), \'-CAcreateserial\', \'-out\', str(cert_path), \'-days\', \'825\', \'-sha256\', \'-extfile\', str(ext)], check=True)\n    os.chmod(key_path, 0o600)\n    os.chmod(cert_path, 0o600)\n\ndef get_cert_fingerprint(cert_path: Path) -> str:\n    pem = cert_path.read_text(encoding=\'utf-8\')\n    der = ssl.PEM_cert_to_DER_cert(pem)\n    digest = hashlib.sha256(der).hexdigest().upper()\n    return \':\'.join(digest[i:i+2] for i in range(0, len(digest), 2))\n\n# --- DEVICE REGISTRY ---\ndef _write_json_atomic(path: Path, data: dict):\n    tmp = path.with_suffix(\'.tmp\')\n    with tmp.open(\'w\', encoding=\'utf-8\') as f:\n        json.dump(data, f, indent=2, sort_keys=True)\n    os.replace(tmp, path)\n\ndef load_devices() -> dict:\n    path = state_dir() / DEVICES_FILE_NAME\n    if not path.is_file(): return {\'version\': 1, \'devices\': []}\n    with path.open(\'r\', encoding=\'utf-8\') as f: return json.load(f)\n\ndef save_devices(data: dict):\n    _write_json_atomic(state_dir() / DEVICES_FILE_NAME, data)\n\ndef register_device(device_id: str, cert_pem: str):\n    data = load_devices()\n    der = ssl.PEM_cert_to_DER_cert(cert_pem)\n    fp = hashlib.sha256(der).hexdigest().upper()\n    fp = \':\'.join(fp[i:i+2] for i in range(0, len(fp), 2))\n    record = {\'client_id\': device_id, \'fingerprint\': fp, \'enrolled_at\': utc_now_iso(), \'status\': \'active\'}\n    data[\'devices\'] = [d for d in data.get(\'devices\', []) if d[\'client_id\'] != device_id] + [record]\n    save_devices(data)\n    (state_dir() / \'devices\' / device_id).mkdir(exist_ok=True)\n    (state_dir() / \'devices\' / device_id / \'client.crt\').write_text(cert_pem, encoding=\'utf-8\')\n    return record\n\n# --- TMUX BRIDGE ---\nclass TmuxBridge:\n    def __init__(self, session_name=\'main\'):\n        self.session_name = session_name\n        self.master_fd = None\n        self.process = None\n\n    def open(self):\n        self.master_fd, slave_fd = pty.openpty()\n        if subprocess.run([\'tmux\', \'has-session\', \'-t\', self.session_name], capture_output=True).returncode != 0:\n            subprocess.run([\'tmux\', \'new-session\', \'-d\', \'-s\', self.session_name])\n        self.process = subprocess.Popen([\'tmux\', \'attach-session\', \'-t\', self.session_name], stdin=slave_fd, stdout=slave_fd, stderr=slave_fd, preexec_fn=os.setsid)\n\n    def write(self, data):\n        if self.master_fd: os.write(self.master_fd, data if isinstance(data, bytes) else data.encode())\n\n    async def read(self):\n        loop = asyncio.get_event_loop()\n        while self.process and self.process.poll() is None:\n            try:\n                r, _, _ = await loop.run_in_executor(None, select.select, [self.master_fd], [], [], 0.1)\n                if self.master_fd in r: return await loop.run_in_executor(None, os.read, self.master_fd, 4096)\n            except Exception: break\n        return b\'\'\n\n    def resize(self, c, r):\n        if self.master_fd: fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, struct.pack(\'HHHH\', r, c, 0, 0))\n\n# --- SERVER & ENROLLMENT ---\nactive_bridges: Dict[str, TmuxBridge] = {}\n\ndef authorize_peer(websocket) -> Tuple[Optional[dict], str]:\n    if not TLS_ENABLED: return {\'client_id\': \'unsafe-user\'}, \'Authorized (Unsafe)\'\n    ssl_obj = websocket.transport.get_extra_info(\'ssl_object\')\n    if not ssl_obj: return None, \'No SSL\'\n    cert_der = ssl_obj.getpeercert(binary_form=True)\n    if not cert_der: return None, \'No Cert\'\n    fp = hashlib.sha256(cert_der).hexdigest().upper()\n    fp = \':\'.join(fp[i:i+2] for i in range(0, len(fp), 2))\n    for dev in load_devices().get(\'devices\', []):\n        if dev[\'fingerprint\'] == fp and dev[\'status\'] == \'active\': return dev, \'Authorized\'\n    return None, \'Unauthorized\'\n\nasync def route_client(ws):\n    dev, msg = authorize_peer(ws)\n    if not dev: await ws.close(1008, msg); return\n    cid = dev[\'client_id\']\n    print(f"Client {cid} connected.")\n    bridge = active_bridges.setdefault(cid, TmuxBridge())\n    if not bridge.process: bridge.open()\n    \n    async def forward():\n        while True:\n            data = await bridge.read()\n            if not data: break\n            await ws.send(json.dumps({\'type\': \'output\', \'data\': base64.b64encode(data).decode()}))\n    \n    task = asyncio.create_task(forward())\n    try:\n        async for m in ws:\n            msg = json.loads(m)\n            if msg[\'type\'] == \'input\': bridge.write(base64.b64decode(msg[\'data\']))\n            elif msg[\'type\'] == \'resize\': bridge.resize(msg.get(\'cols\', 80), msg.get(\'rows\', 24))\n    finally:\n        task.cancel()\n        print(f"Client {cid} disconnected.")\n\nasync def start_server():\n    if not websockets: print("Error: \'websockets\' package not found. Run: pip install websockets"); return\n    generate_server_cert()\n    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)\n    ctx.load_cert_chain(server_cert_file(), server_key_file())\n    ctx.load_verify_locations(ca_cert_file())\n    ctx.verify_mode = ssl.CERT_REQUIRED if TLS_ENABLED else ssl.CERT_NONE\n    print(f"Starting server on wss://{BIND_HOST}:{current_port()}...")\n    async with websockets.serve(route_client, BIND_HOST, current_port(), ssl=ctx):\n        await asyncio.Future()\n\n# --- ENROLLMENT SERVER ---\ndef serve_enrollment(code, device_id, port, ws_url):\n    class Handler(http.server.BaseHTTPRequestHandler):\n        def do_POST(self):\n            if self.path != \'/enroll\': self.send_error(404); return\n            payload = json.loads(self.rfile.read(int(self.headers[\'Content-Length\'])).decode())\n            if payload.get(\'invitationCode\') != code: self.send_error(403); return\n            record = register_device(payload.get(\'deviceId\', device_id), payload[\'csrPem\'])\n            # Sign the CSR (simplified mock signing for now)\n            with tempfile.TemporaryDirectory() as tmp:\n                csr_p = Path(tmp) / \'c.csr\'; crt_p = Path(tmp) / \'c.crt\'\n                csr_p.write_text(payload[\'csrPem\'], encoding=\'utf-8\')\n                subprocess.run([\'openssl\', \'x509\', \'-req\', \'-in\', str(csr_p), \'-CA\', str(ca_cert_file()), \'-CAkey\', str(ca_key_file()), \'-CAcreateserial\', \'-out\', str(crt_p), \'-days\', \'825\'], check=True, capture_output=True)\n                cert_pem = crt_p.read_text(encoding=\'utf-8\')\n            \n            res = {\'deviceId\': record[\'client_id\'], \'clientCertificatePem\': cert_pem, \'caCertificatePem\': ca_cert_file().read_text(), \'wsUrl\': ws_url}\n            self.send_response(200); self.send_header(\'Content-Type\', \'application/json\'); self.end_headers()\n            self.wfile.write(json.dumps(res).encode())\n            nonlocal done; done = True\n        def log_message(self, *args): pass\n\n    done = False\n    server = http.server.ThreadingHTTPServer((BIND_HOST, port), Handler)\n    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)\n    ctx.load_cert_chain(server_cert_file(), server_key_file())\n    server.socket = ctx.wrap_socket(server.socket, server_side=True)\n    print(f"Waiting for device on https://{BIND_HOST}:{port}/enroll ...")\n    while not done: server.handle_request()\n\n# --- CLI FLOWS ---\ndef enroll_flow():\n    ensure_ca(); generate_server_cert()\n    device_id = input("Enter Device ID: ").strip() or f"dev-{secrets.token_hex(2)}"\n    code = secrets.token_urlsafe(16)\n    port = 8766\n    ws_url = f"wss://{BIND_HOST}:{current_port()}"\n    enroll_url = f"https://{BIND_HOST}:{port}/enroll"\n    fp = get_cert_fingerprint(ca_cert_file())\n    \n    payload = {\'t\': \'c-e\', \'w\': ws_url, \'e\': enroll_url, \'c\': code, \'f\': fp, \'l\': device_id}\n    invite_json = json.dumps(payload, separators=(\',\', \':\'))\n    invite_link = \'clawssh://enroll?p=\' + base64.urlsafe_b64encode(invite_json.encode()).decode().rstrip(\'=\')\n    \n    print(f"\\n=== Enrollment Invite ===\\nFingerprint: {fp}\\n")\n    if shutil.which(\'qrencode\'):\n        subprocess.run([\'qrencode\', \'-t\', \'UTF8\', invite_link])\n    else:\n        print(f"Invite Link: {invite_link}")\n    \n    serve_enrollment(code, device_id, port, ws_url)\n\ndef setup_menu():\n    print(f"\\n=== ClawSSH {BACKEND_VERSION} ===")\n    print("1) Start Server")\n    print("2) Enroll Device")\n    print("3) List Devices")\n    print("4) Rotate Server Cert")\n    print("5) Exit")\n    c = input("Option: ").strip()\n    if c == \'1\': asyncio.run(start_server())\n    elif c == \'2\': enroll_flow()\n    elif c == \'3\':\n        for d in load_devices().get(\'devices\', []): print(f"{d[\'client_id\']} [{d[\'status\']}] Fingerprint: {d[\'fingerprint\']}")\n    elif c == \'4\': generate_server_cert(force=True)\n    elif c == \'5\': sys.exit(0)\n\nif __name__ == \'__main__\':\n    import termios\n    if \'--server\' in sys.argv: asyncio.run(start_server())\n    else: setup_menu()\n',
    'session.py': "**********":\\x07|\\x1b\\\\\\\\)\')\n_CONTROL_REPAINT_MIN_INTERVAL_SECONDS = 0.25\n\n\ndef _enable_tcp_nodelay_for_websocket(websocket) -> bool:\n    """Best-effort TCP_NODELAY on the accepted websocket TCP socket."""\n    try:\n        transport = getattr(websocket, \'transport\', None)\n        if transport is None:\n            return False\n        sock = transport.get_extra_info(\'socket\')\n        if sock is None:\n            return False\n        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)\n        return True\n    except Exception as exc:\n        print(f\'[backend] failed to enable TCP_NODELAY: {exc}\')\n        return False\n\n\ndef _maybe_force_control_repaint(bridge: TmuxBridge) -> None:\n    """\n    Control-mode pane-pipe streams shell output, but tmux layout operations\n    (focus/unzoom/resize takeover) may not emit any pane bytes. A throttled\n    Ctrl+L keeps the terminal visible after these actions.\n    """\n    if getattr(bridge, \'backend_mode\', \'\') != \'tmux_control\':\n        return\n    now = time.monotonic()\n    last_at = float(getattr(bridge, \'_last_control_repaint_at\', 0.0) or 0.0)\n    if now - last_at < _CONTROL_REPAINT_MIN_INTERVAL_SECONDS:\n        return\n    setattr(bridge, \'_last_control_repaint_at\', now)\n    try:\n        bridge.write_input(\'\\x0c\')\n    except Exception:\n        pass\n\n\nclass FocusPaneReadyCoordinator:\n    def __init__(self):\n        self._token = 0\n        self._pending_target = \'\'\n        self._debounce_task = None\n        self._fallback_task = None\n\n    def begin(self, websocket, target: str):\n        self.cancel()\n        self._token += 1\n        self._pending_target = (target or \'\').strip()\n        if not self._pending_target:\n            return\n        token = self._token\n        target_copy = self._pending_target\n        # Fallback if tmux redraw is silent or output is delayed longer than expected.\n        self._fallback_task = asyncio.create_task(\n            self._emit_ready_after_delay(websocket, token, target_copy, delay_seconds=1.5, reason=\'timeout\')\n        )\n\n    def on_terminal_output(self, websocket):\n        if not self._pending_target:\n            return\n        if self._debounce_task is not None:\n            self._debounce_task.cancel()\n        token = self._token\n        target_copy = self._pending_target\n        # Wait for terminal output to go quiet after the focus switch refresh.\n        self._debounce_task = asyncio.create_task(\n            self._emit_ready_after_delay(websocket, token, target_copy, delay_seconds=0.35, reason=\'settled\')\n        )\n\n    async def _emit_ready_after_delay(self, websocket, token: int, target: str, delay_seconds: float, reason: str):\n        try:\n            await asyncio.sleep(delay_seconds)\n            if token != self._token or target != self._pending_target or not target:\n                return\n            self._clear_pending()\n            await websocket.send(\n                json.dumps(\n                    {\n                        \'type\': \'focus-pane-ready\',\n                        \'target\': target,\n                        \'reason\': reason,\n                        \'ts\': int(time.time() * 1000),\n                    }\n                )\n            )\n        except asyncio.CancelledError:\n            return\n        except Exception as exc:\n            print(f\'[backend] focus-pane-ready emit failed: {exc}\')\n\n    def _clear_pending(self):\n        self._pending_target = \'\'\n        if self._debounce_task is not None:\n            self._debounce_task.cancel()\n            self._debounce_task = None\n        if self._fallback_task is not None:\n            self._fallback_task.cancel()\n            self._fallback_task = None\n\n    def cancel(self):\n        self._clear_pending()\n\n\nasync def emit_terminal_output(websocket, bridge: TmuxBridge, focus_ready: FocusPaneReadyCoordinator):\n    last_takeover_notice_at = 0.0\n    while True:\n        data = await bridge.read_output()\n        if not data:\n            break\n\n        # Check for OSC 9 notifications: \\x1b]9;MESSAGE\\x07 or \\x1b]9;MESSAGE\\x1b\\\n        # We handle simple case within a single chunk for now.\n        matches = NOTIFICATION_PATTERN.findall(data)\n        for match in matches:\n            try:\n                message = match.decode(\'utf-8\', errors=\'replace\')\n                # If message contains a pipe, split into title and body\n                if \'|\' in message:\n                    title, body = message.split(\'|\', 1)\n                else:\n                    title, body = \'ClawSSH\', message\n\n                await websocket.send(\n                    json.dumps(\n                        {\n                            \'type\': \'notification\',\n                            \'title\': title.strip(),\n                            \'body\': body.strip(),\n                            \'ts\': int(time.time() * 1000),\n                        }\n                    )\n                )\n            except Exception as e:\n                print(f"[backend] Failed to process notification: {e}")\n\n        takeover_target = bridge.get_recent_external_client_pane_target(max_age_seconds=3)\n        if takeover_target:\n            now_mono = time.monotonic()\n            if now_mono - last_takeover_notice_at >= 3.0:\n                await websocket.send(\n                    json.dumps(\n                        {\n                            \'type\': \'takeover\',\n                            \'message\': \'Terminal was taken over on another device\',\n                            \'target\': takeover_target,\n                            \'ts\': int(time.time() * 1000),\n                        }\n                    )\n                )\n                last_takeover_notice_at = now_mono\n        focus_ready.on_terminal_output(websocket)\n        await websocket.send(data)\n\n\nasync def monitor_external_takeover(bridge: TmuxBridge):\n    loop = asyncio.get_event_loop()\n    while True:\n        await asyncio.sleep(0.5)\n        takeover_target = await loop.run_in_executor(\n            None,\n            lambda: bridge.get_recent_external_client_pane_target(max_age_seconds=2),\n        )\n        if takeover_target:\n            await loop.run_in_executor(\n                None,\n                lambda: bridge.release_window_size_lock_for_target(takeover_target),\n            )\n\n\ndef _format_tmux_input_for_log(message) -> str:\n    if isinstance(message, str):\n        text = message\n    elif isinstance(message, memoryview):\n        text = message.tobytes().decode(\'utf-8\', errors=\'replace\')\n    elif isinstance(message, (bytes, bytearray)):\n        text = bytes(message).decode(\'utf-8\', errors=\'replace\')\n    else:\n        text = repr(message)\n\n    escaped = text.encode(\'unicode_escape\', errors=\'backslashreplace\').decode(\'ascii\', errors=\'replace\')\n    if len(escaped) > 240:\n        return f\'{escaped[:240]}...\'\n    return escaped\n\n\ndef _to_int_or_default(value, default=0) -> int:\n    try:\n        return int(str(value))\n    except (TypeError, ValueError):\n        return default\n\n\ndef _to_positive_int_or_default(value, default=1) -> int:\n    return max(1, _to_int_or_default(value, default))\n\n\ndef _is_reasonable_size(cols: int, rows: int) -> bool:\n    return 2 <= cols <= MAX_REASONABLE_COLS and 2 <= rows <= MAX_REASONABLE_ROWS\n\n\nasync def _run_bridge_call(loop, func, *args, **kwargs):\n    if args or kwargs:\n        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))\n    return await loop.run_in_executor(None, func)\n\n\ndef _bridge_process_is_alive(bridge: TmuxBridge) -> bool:\n    process = getattr(bridge, \'process\', None)\n    if process is None:\n        return True\n    try:\n        return process.poll() is None\n    except Exception:\n        return False\n\n\nasync def forward_client_input(websocket, bridge: TmuxBridge, focus_ready: FocusPaneReadyCoordinator):\n    control_types = {\n        \'pane-list\',\n        \'focus-pane\',\n        \'unzoom-pane\',\n        \'rename-session\',\n        \'create-pane\',\n        \'create-worktree\',\n        \'resize\',\n        \'resize-takeover\',\n        \'send-prefix\',\n        \'ui-event\',\n        \'request-snapshot\',\n    }\n    control_type_pattern = re.compile(r\'"type"\\s*:\\s*"([^"]+)"\')\n    resize_lock_until = 0.0\n    loop = asyncio.get_running_loop()\n\n    async for message in websocket:\n        raw_text = None\n        message_type = \'terminal-input\'\n        if isinstance(message, str):\n            raw_text = message\n        elif isinstance(message, memoryview):\n            try:\n                raw_text = message.tobytes().decode(\'utf-8\')\n            except UnicodeDecodeError:\n                raw_text = None\n        elif isinstance(message, (bytes, bytearray)):\n            try:\n                raw_text = message.decode(\'utf-8\')\n            except UnicodeDecodeError:\n                raw_text = None\n\n        if raw_text is not None:\n            stripped_preview = raw_text.lstrip()\n            match_preview = control_type_pattern.search(stripped_preview)\n            matched_type_preview = match_preview.group(1) if match_preview else None\n            if matched_type_preview:\n                message_type = matched_type_preview\n\n        try:\n            if raw_text is not None:\n                stripped = raw_text.lstrip()\n                match = control_type_pattern.search(stripped)\n                matched_type = match.group(1) if match else None\n                looks_like_control_json = matched_type in control_types\n                try:\n                    parsed = json.loads(stripped)\n                    msg_type = parsed.get(\'type\') if isinstance(parsed, dict) else None\n                    if msg_type == \'pane-list\':\n                        print(\'[tmux-input] control pane-list\')\n                        panes = await _run_bridge_call(loop, bridge.list_panes)\n                        await websocket.send(json.dumps({\'type\': \'pane-list\', \'panes\': panes, \'homePath\': bridge.home_dir}))\n                        continue\n                    if msg_type == \'focus-pane\':\n                        target = parsed.get(\'target\', \'\')\n                        if not target:\n                            await websocket.send(json.dumps({\'type\': \'pane-error\', \'message\': \'Missing pane target\'}))\n                            continue\n                        print(f\'[tmux-input] control focus-pane target={target}\')\n                        focus_ready.begin(websocket, target)\n                        await _run_bridge_call(loop, bridge.focus_pane, target)\n                        _maybe_force_control_repaint(bridge)\n                        resize_lock_until = time.monotonic() + RESIZE_TAKEOVER_TTL_SECONDS\n                        await websocket.send(json.dumps({\'type\': \'focus-pane\', \'ok\': True, \'target\': target}))\n                        continue\n                    if msg_type == \'unzoom-pane\':\n                        print(\'[tmux-input] control unzoom-pane\')\n                        await _run_bridge_call(loop, bridge.unzoom_last_focused_pane)\n                        _maybe_force_control_repaint(bridge)\n                        continue\n                    if msg_type == \'request-snapshot\':\n                        target = str(parsed.get(\'target\', \'\')).strip()\n                        if not target:\n                            await websocket.send(json.dumps({\'type\': \'pane-error\', \'message\': \'Missing pane target\'}))\n                            continue\n                        print(f\'[tmux-input] control request-snapshot target={target}\')\n                        snapshot_bytes = await _run_bridge_call(loop, bridge.capture_pane, target)\n                        if snapshot_bytes:\n                            await websocket.send(\n                                json.dumps(\n                                    {\n                                        \'type\': \'pane-snapshot\',\n                                        \'ok\': True,\n                                        \'target\': target,\n                                        \'encoding\': \'base64\',\n                                        \'data\': base64.b64encode(snapshot_bytes).decode(\'ascii\'),\n                                    }\n                                )\n                            )\n                        else:\n                            await websocket.send(\n                                json.dumps(\n                                    {\n                                        \'type\': \'pane-snapshot\',\n                                        \'ok\': False,\n                                        \'target\': target,\n                                        \'message\': \'Failed to capture pane snapshot\',\n                                    }\n                                )\n                            )\n                        continue\n                    if msg_type == \'rename-session\':\n                        old_name = str(parsed.get(\'oldName\', \'\')).strip()\n                        new_name = str(parsed.get(\'newName\', \'\')).strip()\n                        if not old_name or not new_name:\n                            await websocket.send(json.dumps({\'type\': \'pane-error\', \'message\': \'Missing session name(s)\'}))\n                            continue\n                        print(f\'[tmux-input] control rename-session old={old_name} new={new_name}\')\n                        await _run_bridge_call(loop, bridge.rename_session, old_name, new_name)\n                        panes = await _run_bridge_call(loop, bridge.list_panes)\n                        await websocket.send(json.dumps({\'type\': \'pane-list\', \'panes\': panes, \'homePath\': bridge.home_dir}))\n                        continue\n                    if msg_type == \'create-pane\':\n                        agent = str(parsed.get(\'agent\', \'\')).strip().lower()\n                        path = str(parsed.get(\'path\', \'\')).strip()\n                        command = str(parsed.get(\'command\', \'\')).strip()\n                        print(\n                            \'[tmux-input] control create-pane \'\n                            f\'agent={agent or "shell"} path={path or bridge.home_dir} session={bridge.session_name} \'\n                            f\'command={"set" if command else "none"}\'\n                        )\n                        target = await _run_bridge_call(\n                            loop,\n                            bridge.create_pane,\n                            agent,\n                            path,\n                            bridge.session_name,\n                            command=command,\n                        )\n                        await websocket.send(json.dumps({\'type\': \'create-pane\', \'ok\': True, \'target\': target}))\n                        panes = await _run_bridge_call(loop, bridge.list_panes)\n                        await websocket.send(json.dumps({\'type\': \'pane-list\', \'panes\': panes, \'homePath\': bridge.home_dir}))\n                        continue\n                    if msg_type == \'create-worktree\':\n                        repo_path = str(parsed.get(\'repoPath\', \'\')).strip()\n                        source_target = str(parsed.get(\'target\', \'\')).strip()\n                        worktree_name = str(parsed.get(\'worktreeName\', \'\')).strip()\n                        worktree_location = str(parsed.get(\'worktreeLocation\', \'\')).strip()\n                        if not repo_path:\n                            await websocket.send(json.dumps({\'type\': \'pane-error\', \'message\': \'Missing repository path\'}))\n                            continue\n                        if not worktree_name:\n                            await websocket.send(json.dumps({\'type\': \'pane-error\', \'message\': \'Missing worktree repo name\'}))\n                            continue\n                        if not worktree_location:\n                            await websocket.send(json.dumps({\'type\': \'pane-error\', \'message\': \'Missing worktree location\'}))\n                            continue\n                        print(\n                            \'[tmux-input] control create-worktree \'\n                            f\'repoPath={repo_path} target={source_target or "-"} \'\n                            f\'worktreeName={worktree_name} worktreeLocation={worktree_location}\'\n                        )\n                        created_path = await _run_bridge_call(\n                            loop,\n                            bridge.create_worktree,\n                            repo_path,\n                            worktree_name=worktree_name,\n                            worktree_location=worktree_location,\n                        )\n                        pane_target = await _run_bridge_call(loop, bridge.create_window, created_path, bridge.session_name)\n                        await websocket.send(\n                            json.dumps(\n                                {\n                                    \'type\': \'create-worktree\',\n                                    \'ok\': True,\n                                    \'repoPath\': repo_path,\n                                    \'worktreeName\': worktree_name,\n                                    \'worktreeLocation\': worktree_location,\n                                    \'createdPath\': created_path,\n                                    \'target\': pane_target,\n                                }\n                            )\n                        )\n                        panes = await _run_bridge_call(loop, bridge.list_panes)\n                        await websocket.send(json.dumps({\'type\': \'pane-list\', \'panes\': panes, \'homePath\': bridge.home_dir}))\n                        continue\n                    if msg_type == \'resize\':\n                        cols = _to_positive_int_or_default(parsed.get(\'cols\'), 80)\n                        rows = _to_positive_int_or_default(parsed.get(\'rows\'), 24)\n                        if not _is_reasonable_size(cols, rows):\n                            print(f\'[tmux-input] ignore resize out-of-range cols={cols} rows={rows}\')\n                            await websocket.send(\n                                json.dumps(\n                                    {\n                                        \'type\': \'control-event\',\n                                        \'event\': \'resize-ack\',\n                                        \'ok\': False,\n                                        \'applied\': False,\n                                        \'reason\': \'out-of-range\',\n                                        \'cols\': cols,\n                                        \'rows\': rows,\n                                    }\n                                )\n                            )\n                            continue\n                        bridge.resize(cols, rows)\n                        await websocket.send(\n                            json.dumps(\n                                {\n                                    \'type\': \'control-event\',\n                                    \'event\': \'resize-ack\',\n                                    \'ok\': True,\n                                    \'applied\': True,\n                                    \'cols\': cols,\n                                    \'rows\': rows,\n                                }\n                            )\n                        )\n                        continue\n                    if msg_type == \'resize-takeover\':\n                        cols = _to_positive_int_or_default(parsed.get(\'cols\'), 80)\n                        rows = _to_positive_int_or_default(parsed.get(\'rows\'), 24)\n                        if not _is_reasonable_size(cols, rows):\n                            print(f\'[tmux-input] ignore resize-takeover out-of-range cols={cols} rows={rows}\')\n                            continue\n                        requested_target = str(parsed.get(\'target\', \'\')).strip()\n\n                        resolved_target = requested_target\n                        if not resolved_target:\n                            # Fallback 1: recent external activity\n                            resolved_target = bridge.get_recent_external_client_pane_target(max_age_seconds=3)\n                            # Fallback 2: bridge\'s own last focused target\n                            if not resolved_target:\n                                resolved_target = bridge.last_focused_target\n\n                        print(\n                            \'[tmux-input] control resize-takeover \'\n                            f\'requested={requested_target or "-"} resolved={resolved_target or "-"} \'\n                            f\'cols={cols} rows={rows}\'\n                        )\n                        if resolved_target:\n                            await _run_bridge_call(loop, bridge.force_takeover_resize, resolved_target, cols, rows)\n                            # Force terminal repaint at reclaimed dimensions.\n                            _maybe_force_control_repaint(bridge)\n                        else:\n                            # Fallback keeps PTY/client state in sync even without a resolvable pane/window target.\n                            bridge.resize(cols, rows)\n                        resize_lock_until = time.monotonic() + RESIZE_TAKEOVER_TTL_SECONDS\n                        continue\n                    if msg_type == \'send-prefix\':\n                        try:\n                            bridge.send_tmux_prefix()\n                            await websocket.send(\n                                json.dumps(\n                                    {\n                                        \'type\': \'control-event\',\n                                        \'event\': \'send-prefix-ack\',\n                                        \'ok\': True,\n                                    }\n                                )\n                            )\n                        except Exception as exc:\n                            await websocket.send(\n                                json.dumps(\n                                    {\n                                        \'type\': \'control-event\',\n                                        \'event\': \'send-prefix-ack\',\n                                        \'ok\': False,\n                                        \'message\': str(exc),\n                                    }\n                                )\n                            )\n                        continue\n                    if msg_type == \'ui-event\':\n                        # UI telemetry events are intentionally ignored server-side.\n                        continue\n                except Exception as exc:\n                    # If this looks like a control packet, never leak it into the PTY stream.\n                    is_control_like = looks_like_control_json\n                    if is_control_like:\n                        await websocket.send(json.dumps({\'type\': \'pane-error\', \'message\': f\'Control message failed: {exc}\'}))\n                        continue\n                else:\n                    # Parsed JSON control/object frame that we don\'t handle should not be forwarded as terminal input.\n                    if isinstance(parsed, dict) and isinstance(parsed.get(\'type\'), str):\n                        continue\n\n                # Never forward JSON-like control envelopes into the shell input stream.\n                if looks_like_control_json:\n                    continue\n\n            print(f"[tmux-input] data { _format_tmux_input_for_log(message) }")\n            resize_lock_until = time.monotonic() + RESIZE_TAKEOVER_TTL_SECONDS\n            bridge.write_input(message)\n        finally:\n            pass\n\n\nasync def monitor_pane_changes(websocket, bridge: TmuxBridge):\n    """\n    Periodically polls the tmux pane list and pushes updates to the client\n    when changes are detected. This enables auto-refresh without a manual button.\n    """\n    last_panes_json = None\n    loop = asyncio.get_running_loop()\n    while True:\n        try:\n            # Poll every 2 seconds for external changes.\n            await asyncio.sleep(2.0)\n            if not _bridge_process_is_alive(bridge):\n                print(\'[backend] monitor_pane_changes stopping: bridge process is not alive\')\n                return\n\n            panes = await _run_bridge_call(loop, bridge.list_panes)\n            # We normalize the active status for comparison because it depends on the bridge\'s \n            # internal state (client_target) which might change without tmux changing.\n            current_panes_json = json.dumps(panes, sort_keys=True)\n            \n            if last_panes_json is not None and current_panes_json != last_panes_json:\n                await websocket.send(json.dumps({\n                    \'type\': \'pane-list\', \n                    \'panes\': panes, \n                    \'homePath\': bridge.home_dir\n                }))\n            \n            last_panes_json = current_panes_json\n        except Exception as e:\n            # Log and continue; we don\'t want to crash the session if tmux fails briefly.\n            print(f"[backend] monitor_pane_changes error: {e}")\n            await asyncio.sleep(5.0)\n\n\ndef _build_bridge():\n    return create_terminal_backend(backend_mode=\'control\')\n\n\ndef _string_attr_or_default(obj, attr_name: str, default: str) -> str:\n    value = getattr(obj, attr_name, default)\n    return value if isinstance(value, str) else default\n\n\ndef _remote_ip(remote_addr) -> str:\n    if isinstance(remote_addr, tuple) and remote_addr:\n        return str(remote_addr[0])\n    return str(remote_addr or \'unknown\')\n\n\nasync def bridge_client(websocket):\n    remote_addr = websocket.remote_address\n    remote_ip = _remote_ip(remote_addr)\n    print(f"--- New connection attempt from {remote_addr} ---")\n    if _enable_tcp_nodelay_for_websocket(websocket):\n        print(f"[backend] TCP_NODELAY enabled for {remote_addr}")\n\n    device_id = \'unknown\'\n    if TLS_ENABLED:\n        device, auth_reason = authorize_websocket_peer(websocket, STATE_DIR)\n        if device is None:\n            print(f"[auth] Authentication failed ts={int(time.time())} remote_ip={remote_ip} reason={auth_reason}")\n            await websocket.close(code=4003, reason=\'Authentication failed\')\n            return\n        device_id = str(device.get(\'client_id\', \'unknown\') or \'unknown\')\n        mark_device_connection(device_id, remote_ip, STATE_DIR)\n        print(f"[auth] Accepted ts={int(time.time())} remote_ip={remote_ip} device_id={device_id}")\n    elif ALLOW_UNSAFE_NO_AUTH:\n        print(f"[auth] WARNING: allowing unauthenticated connection ts={int(time.time())} remote_ip={remote_ip}")\n\n    bridge = _build_bridge()\n    try:\n        bridge.open()\n        mode_label = _string_attr_or_default(bridge, \'backend_mode_label\', \'tmux\')\n        print(f"[bridge] bridge opened for {remote_addr} device_id={device_id} mode={mode_label}")\n    except Exception as e:\n        print(f"[bridge] failed to open tmux bridge: {e}")\n        await websocket.close()\n        return\n\n    try:\n        await websocket.send(\n            json.dumps(\n                {\n                    \'type\': \'control-event\',\n                    \'event\': \'backend-version\',\n                    \'version\': BACKEND_VERSION,\n                    \'tmuxBackendMode\': _string_attr_or_default(bridge, \'backend_mode\', \'tmux_control\'),\n                    \'tmuxBackendModeLabel\': _string_attr_or_default(bridge, \'backend_mode_label\', \'tmux-control\'),\n                    \'tmuxControlStatus\': _string_attr_or_default(bridge, \'control_mode_status\', \'\'),\n                    \'tmuxPrefix\': bridge.get_tmux_prefix() if hasattr(bridge, \'get_tmux_prefix\') else \'C-b\',\n                    \'ts\': int(time.time() * 1000),\n                }\n            )\n        )\n        focus_ready = FocusPaneReadyCoordinator()\n        output_task = asyncio.create_task(emit_terminal_output(websocket, bridge, focus_ready))\n        takeover_task = asyncio.create_task(monitor_external_takeover(bridge))\n        input_task = asyncio.create_task(forward_client_input(websocket, bridge, focus_ready))\n        monitor_task = asyncio.create_task(monitor_pane_changes(websocket, bridge))\n\n        done, pending = await asyncio.wait(\n            [output_task, input_task, takeover_task, monitor_task],\n            return_when=asyncio.FIRST_COMPLETED,\n        )\n\n        for task in pending:\n            task.cancel()\n        for task in done:\n            try:\n                task.result()\n            except Exception as e:\n                print(f"[bridge] task result error for {remote_addr} device_id={device_id}: {e}")\n    except Exception as exc:\n        print(f\'[bridge] error for {remote_addr} device_id={device_id}: {exc}\')\n    finally:\n        if \'focus_ready\' in locals():\n            focus_ready.cancel()\n        print(f"--- Closing bridge for {remote_addr} device_id={device_id} ---")\n        bridge.close()\n\n\ndef _websocket_path(websocket) -> str:\n    request = getattr(websocket, \'request\', None)\n    path = getattr(request, \'path\', None)\n    if isinstance(path, str) and path:\n        return path\n    fallback = getattr(websocket, \'path\', \'\')\n    return fallback if isinstance(fallback, str) else \'\'\n\n\nasync def route_client(websocket):\n    await bridge_client(websocket)\n',
    'terminal_backend.py': "**********": str | None = None) -> str:\n    raw = (value or \'\').strip().lower()\n    if raw == \'control\':\n        return \'control\'\n    return \'control\'\n\n\n@runtime_checkable\nclass TerminalBackendAdapter(Protocol):\n    backend_mode: str\n    backend_mode_label: str\n\n    def open(self): ...\n    def close(self): ...\n    async def read_output(self): ...\n    def write_input(self, message): ...\n    def resize(self, cols: int, rows: int, force_refresh: bool = False): ...\n    def list_panes(self, session_name: str = \'\'): ...\n    def focus_pane(self, target: str): ...\n    def unzoom_last_focused_pane(self): ...\n    def rename_session(self, old_name: str, new_name: str): ...\n    def create_pane(\n        self,\n        agent: str = \'\',\n        path: str = \'\',\n        session_name: str = \'\',\n        command: str = \'\',\n        resize_window: bool = True,\n        split_target: str = \'\',\n    ) -> str: ...\n    def create_window(self, path: str = \'\', session_name: str = \'\', command: str = \'\') -> str: ...\n    def create_worktree(self, repo_path: str, worktree_name: str = \'\', worktree_location: str = \'\') -> str: ...\n    def force_takeover_resize(self, target: str, cols: int, rows: int) -> bool: ...\n    def get_tmux_prefix(self) -> str: ...\n    def send_tmux_prefix(self): ...\n    def get_recent_external_client_pane_target(self, max_age_seconds: int = 3) -> str: ...\n    def release_window_size_lock_for_target(self, target: str, min_interval_seconds: float = 1.0) -> bool: ...\n\n\nclass TmuxControlAdapter(TmuxBridge):\n    """\n    Phase-1/2 wiring for tmux control backend rollout.\n\n    This adapter is the default backend and uses a\n    PTY-backed tmux control-mode client (`tmux -C`) to stream `%output` events\n    instead of relying on the hidden attach-session redraw path.\n    """\n\n    backend_mode = \'tmux_control\'\n    backend_mode_label = \'tmux-control\'\n    _CONTROL_OUTPUT_RE = re.compile(r\'^%output\\s+(%\\d+)\\s+(.*)$\')\n    _CONTROL_EXTENDED_OUTPUT_RE = re.compile(r\'^%extended-output\\s+(%\\d+)\\s+.*?\\s(.*)$\')\n\n    def __init__(self, session_name: str = \'\'):\n        super().__init__(session_name=session_name)\n        self.control_mode_requested = True\n        self.control_mode_active = False\n        self.control_mode_status = \'not-open\'\n        self._control_read_buffer = bytearray()\n        self._control_pending_output_chunks: list[bytes] = []\n        self._pane_pipe_target = \'\'\n        self._pane_pipe_fifo_path = \'\'\n        self._pane_pipe_fd: int | None = None\n        self._last_output_source = \'control\'\n        self._pending_sendkeys_bytes = bytearray()\n        self._pending_sendkeys_flush_handle = None\n        self._tmux_prefix_bytes: bytes | None = None\n\n    def open(self):\n        self.master_fd, self.slave_fd = pty.openpty()\n\n        if not self.session_name:\n            self.session_name = self._resolve_or_create_session_name()\n        else:\n            try:\n                self._check_call(\n                    [\'tmux\', \'has-session\', \'-t\', self.session_name],\n                    stdout=subprocess.DEVNULL,\n                    stderr=subprocess.DEVNULL,\n                )\n            except subprocess.CalledProcessError:\n                self._check_call(\n                    [\'tmux\', \'new-session\', \'-d\', \'-s\', self.session_name],\n                    stdout=subprocess.DEVNULL,\n                    stderr=subprocess.DEVNULL,\n                )\n\n        # Keep sessions alive and reduce tmux-side delays.\n        try:\n            self._run(\n                [\'tmux\', \'set-option\', \'-t\', self.session_name, \'destroy-unattached\', \'off\'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            self._run(\n                [\'tmux\', \'set-option\', \'-s\', \'exit-unattached\', \'off\'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            self._run(\n                [\'tmux\', \'set-option\', \'-s\', \'escape-time\', \'0\'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            self._run(\n                [\'tmux\', \'set-option\', \'-s\', \'tty-update-time\', \'0\'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            self._run(\n                [\'tmux\', \'set-option\', \'-t\', self.session_name, \'status\', \'off\'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            self._run(\n                [\'tmux\', \'set-option\', \'-t\', self.session_name, \'status-interval\', \'0\'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n        except Exception:\n            pass\n\n        cols, rows = self._get_session_window_size()\n        self.last_client_size = (cols, rows)\n        size = struct.pack(\'HHHH\', rows, cols, 0, 0)\n        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, size)\n        fcntl.ioctl(self.slave_fd, termios.TIOCSWINSZ, size)\n\n        def _preexec():\n            os.setsid()\n            fcntl.ioctl(self.slave_fd, termios.TIOCSCTTY, 0)\n\n        self.process = subprocess.Popen(\n            [\'tmux\', \'-C\', \'attach-session\', \'-t\', self.session_name],\n            stdin=self.slave_fd,\n            stdout=self.slave_fd,\n            stderr=self.slave_fd,\n            preexec_fn=_preexec,\n            env=self.tmux_env,\n        )\n        self.control_mode_active = True\n        self.control_mode_status = \'active\'\n        self._initialize_control_focus_target()\n        if self.last_focused_target:\n            self._activate_pipe_pane_output_for_target(self.last_focused_target)\n\n    def _initialize_control_focus_target(self):\n        try:\n            panes = self.list_panes(self.session_name)\n        except Exception:\n            return\n        for pane in panes:\n            if pane.get(\'isActive\'):\n                target = str(pane.get(\'target\', \'\')).strip()\n                if target:\n                    self.last_focused_target = target\n                    return\n\n    @classmethod\n    def _decode_control_payload(cls, payload: str) -> bytes:\n        out = bytearray()\n        i = 0\n        length = len(payload)\n        while i < length:\n            ch = payload[i]\n            if ch != \'\\\\\':\n                out.extend(ch.encode(\'utf-8\', errors=\'replace\'))\n                i += 1\n                continue\n\n            i += 1\n            if i >= length:\n                out.append(ord(\'\\\\\'))\n                break\n\n            esc = payload[i]\n            if esc in \'01234567\':\n                digits = esc\n                i += 1\n                for _ in range(2):\n                    if i < length and payload[i] in \'01234567\':\n                        digits += payload[i]\n                        i += 1\n                    else:\n                        break\n                try:\n                    out.append(int(digits, 8))\n                except ValueError:\n                    out.extend(f\'\\\\{digits}\'.encode(\'utf-8\', errors=\'replace\'))\n                continue\n\n            simple = {\n                \'n\': b\'\\n\',\n                \'r\': b\'\\r\',\n                \'t\': b\'\\t\',\n                \'\\\\\': b\'\\\\\',\n            }.get(esc)\n            if simple is not None:\n                out.extend(simple)\n            else:\n                out.extend(esc.encode(\'utf-8\', errors=\'replace\'))\n            i += 1\n\n        return bytes(out)\n\n    def _extract_control_output_chunks_from_text(self, text: str) -> list[bytes]:\n        chunks: list[bytes] = []\n        for raw_line in text.splitlines():\n            line = raw_line.rstrip(\'\\r\')\n            match = self._CONTROL_OUTPUT_RE.match(line) or self._CONTROL_EXTENDED_OUTPUT_RE.match(line)\n            if not match:\n                continue\n            pane_target = match.group(1)\n            payload = match.group(2)\n            if self.last_focused_target and pane_target != self.last_focused_target:\n                # Phase 2 can multiplex panes; for now, keep single-pane semantics.\n                continue\n            decoded = self._decode_control_payload(payload)\n            if decoded:\n                chunks.append(decoded)\n        return chunks\n\n    def _read_control_raw(self) -> bytes:\n        return os.read(self.master_fd, 4096)\n\n    def _consume_control_raw_bytes(self, raw: bytes):\n        if not raw:\n            return\n        self._control_read_buffer.extend(raw)\n        while True:\n            newline_index = self._control_read_buffer.find(b\'\\n\')\n            if newline_index < 0:\n                break\n            line = bytes(self._control_read_buffer[: newline_index + 1])\n            del self._control_read_buffer[: newline_index + 1]\n            try:\n                text = line.decode(\'utf-8\', errors=\'replace\')\n            except Exception:\n                continue\n            self._control_pending_output_chunks.extend(self._extract_control_output_chunks_from_text(text))\n\n    def _drain_control_buffer_nowait(self) -> bytes:\n        if not self._control_pending_output_chunks:\n            return b\'\'\n        if len(self._control_pending_output_chunks) == 1:\n            return self._control_pending_output_chunks.pop(0)\n        merged = b\'\'.join(self._control_pending_output_chunks)\n        self._control_pending_output_chunks.clear()\n        return merged\n\n    def _ensure_pipe_fifo(self):\n        if self._pane_pipe_fifo_path and os.path.exists(self._pane_pipe_fifo_path):\n            return\n        self._pane_pipe_fifo_path = f\'/tmp/clawssh-tmux-control-{os.getpid()}-{id(self):x}.fifo\'\n        try:\n            if os.path.exists(self._pane_pipe_fifo_path):\n                os.unlink(self._pane_pipe_fifo_path)\n            os.mkfifo(self._pane_pipe_fifo_path, 0o600)\n        except FileExistsError:\n            pass\n\n    def _ensure_pipe_reader_open(self):\n        if self._pane_pipe_fd is not None:\n            return\n        self._ensure_pipe_fifo()\n        self._pane_pipe_fd = os.open(self._pane_pipe_fifo_path, os.O_RDONLY | os.O_NONBLOCK)\n\n    def _drain_control_responses_nowait(self, max_reads: int = 16):\n        if self.master_fd is None:\n            return\n        reads = 0\n        while reads < max_reads:\n            try:\n                readable, _, _ = select.select([self.master_fd], [], [], 0)\n            except Exception:\n                return\n            if self.master_fd not in readable:\n                return\n            try:\n                raw = os.read(self.master_fd, 4096)\n            except BlockingIOError:\n                return\n            except OSError:\n                return\n            if not raw:\n                return\n            reads += 1\n            self._consume_control_raw_bytes(raw)\n        if self._pane_pipe_fd is not None:\n            # With direct pane output enabled, tmux control redraw chunks are noise.\n            self._control_pending_output_chunks.clear()\n\n    def _disable_pipe_pane_output_for_target(self, target: str):\n        normalized = (target or \'\').strip()\n        if not normalized:\n            return\n        try:\n            self._run(\n                [\'tmux\', \'pipe-pane\', \'-t\', normalized],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n        except Exception:\n            pass\n\n    def _activate_pipe_pane_output_for_target(self, target: str):\n        normalized = (target or \'\').strip()\n        if not normalized:\n            return\n        if normalized == self._pane_pipe_target and self._pane_pipe_fd is not None:\n            return\n\n        if self._pane_pipe_target and self._pane_pipe_target != normalized:\n            self._disable_pipe_pane_output_for_target(self._pane_pipe_target)\n\n        self._ensure_pipe_reader_open()\n        fifo_cmd = f\'cat > {shlex.quote(self._pane_pipe_fifo_path)}\'\n        self._run(\n            [\'tmux\', \'pipe-pane\', \'-O\', \'-t\', normalized, fifo_cmd],\n            check=False,\n            stdout=subprocess.DEVNULL,\n            stderr=subprocess.DEVNULL,\n        )\n        self._pane_pipe_target = normalized\n        self._suppress_control_client_output_for_pane(normalized)\n\n    def _suppress_control_client_output_for_pane(self, pane_target: str):\n        client_target = \'\'\n        try:\n            client_target = self._get_client_target()\n        except Exception:\n            client_target = \'\'\n        if not client_target:\n            return\n        try:\n            self._run(\n                [\'tmux\', \'refresh-client\', \'-t\', client_target, \'-A\', f\'{pane_target}:off\'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n        except Exception:\n            pass\n\n    def _read_next_ready_stream_blocking(self) -> tuple[str, bytes]:\n        while not self._closed_event.is_set():\n            if self.process is not None and self.process.poll() is not None:\n                return (\'none\', b\'\')\n            fds = []\n            if self._pane_pipe_fd is not None:\n                fds.append(self._pane_pipe_fd)\n            if self.master_fd is not None:\n                fds.append(self.master_fd)\n            if not fds:\n                return (\'none\', b\'\')\n\n            try:\n                readable, _, _ = select.select(fds, [], [], 0.25)\n            except (OSError, ValueError):\n                return (\'none\', b\'\')\n            if not readable:\n                if self.process is not None and self.process.poll() is not None:\n                    return (\'none\', b\'\')\n                continue\n            if self.master_fd is not None and self.master_fd in readable:\n                try:\n                    raw = os.read(self.master_fd, 4096)\n                except BlockingIOError:\n                    raw = b\'\'\n                except OSError:\n                    return (\'control\', b\'\')\n                if raw:\n                    self._last_output_source = \'control\'\n                    self._consume_control_raw_bytes(raw)\n                elif self.master_fd is not None and self._pane_pipe_fd is None:\n                    return (\'control\', b\'\')\n            if self._pane_pipe_fd is not None and self._pane_pipe_fd in readable:\n                try:\n                    data = os.read(self._pane_pipe_fd, 4096)\n                except BlockingIOError:\n                    data = b\'\'\n                if data:\n                    self._drain_control_responses_nowait()\n                    self._last_output_source = \'pane-pipe\'\n                    return (\'pane-pipe\', data)\n                # Writer may have restarted/closed; keep reader and continue.\n            if self.master_fd is not None and self.master_fd in readable and self._pane_pipe_fd is None:\n                queued = self._drain_control_buffer_nowait()\n                if queued:\n                    self._last_output_source = \'control\'\n                    return (\'control-parsed\', queued)\n        return (\'none\', b\'\')\n\n    async def read_output(self):\n        if self._pane_pipe_fd is not None:\n            # Prefer direct pane output path for low-latency typing.\n            pass\n        queued = self._drain_control_buffer_nowait()\n        if queued and self._pane_pipe_fd is None:\n            return queued\n\n        loop = asyncio.get_event_loop()\n        while True:\n            stream_kind, raw = await loop.run_in_executor(None, self._read_next_ready_stream_blocking)\n            if stream_kind == \'pane-pipe\':\n                if raw:\n                    return raw\n                continue\n            if stream_kind == \'control-parsed\':\n                if raw:\n                    return raw\n                continue\n            if stream_kind == \'none\':\n                return b\'\'\n            if not raw:\n                return b\'\'\n            self._consume_control_raw_bytes(raw)\n\n            # If a pipe-pane stream is active, control-mode redraw output is intentionally\n            # discarded to avoid tmux client redraw batching/duplication.\n            if self._pane_pipe_fd is not None:\n                self._control_pending_output_chunks.clear()\n                continue\n\n            queued = self._drain_control_buffer_nowait()\n            if queued:\n                return queued\n\n    def focus_pane(self, target: str):\n        self._flush_pending_sendkeys()\n        super().focus_pane(target)\n        normalized = (target or \'\').strip()\n        if normalized:\n            self.last_focused_target = normalized\n            try:\n                self._activate_pipe_pane_output_for_target(normalized)\n            except Exception:\n                pass\n\n    def unzoom_last_focused_pane(self):\n        self._flush_pending_sendkeys()\n        return super().unzoom_last_focused_pane()\n\n    def close(self):\n        self._flush_pending_sendkeys()\n        try:\n            if self._pane_pipe_target:\n                self._disable_pipe_pane_output_for_target(self._pane_pipe_target)\n        except Exception:\n            pass\n        self._pane_pipe_target = \'\'\n\n        if self._pane_pipe_fd is not None:\n            try:\n                os.close(self._pane_pipe_fd)\n            except OSError:\n                pass\n            self._pane_pipe_fd = None\n\n        if self._pane_pipe_fifo_path:\n            try:\n                os.unlink(self._pane_pipe_fifo_path)\n            except OSError:\n                pass\n            self._pane_pipe_fifo_path = \'\'\n\n        super().close()\n\n    def _send_control_command(self, command: str):\n        if self.master_fd is None:\n            raise RuntimeError(\'tmux control client is not open\')\n        payload = command.rstrip(\'\\r\\n\') + \'\\n\'\n        os.write(self.master_fd, payload.encode(\'utf-8\', errors=\'replace\'))\n\n    def _send_input_bytes_now(self, data: bytes):\n        if not data:\n            return\n        target = (self.last_focused_target or \'\').strip() or \'\'\n        # send-keys -H injects raw bytes without relying on tmux client rendering.\n        # Chunk commands to avoid overlong control-mode command lines.\n        max_bytes_per_cmd = 128\n        for start in range(0, len(data), max_bytes_per_cmd):\n            chunk = data[start : start + max_bytes_per_cmd]\n            args = [\'send-keys\']\n            if target:\n                args.extend([\'-t\', target])\n            args.append(\'-H\')\n            args.extend([f\'{byte:02x}\' for byte in chunk])\n            self._send_control_command(\' \'.join(args))\n        # Prevent control-mode command responses from backing up and stalling tmux.\n        self._drain_control_responses_nowait()\n\n    @staticmethod\n    def _tmux_key_notation_to_bytes(value: str) -> bytes | None:\n        raw = (value or \'\').strip()\n        if not raw or raw.lower() == \'none\':\n            return None\n\n        alt = False\n        ctrl = False\n        token = raw\n        while True:\n            upper = token.upper()\n            if upper.startswith(\'M-\'):\n                alt = True\n                token = token[2:]\n                continue\n            if upper.startswith(\'C-\'):\n                ctrl = True\n                token = token[2:]\n                continue\n            break\n\n        if not token:\n            return None\n\n        if ctrl:\n            if len(token) == 1:\n                ch = token\n                code = ord(ch.upper())\n                if ord(\'A\') <= code <= ord(\'Z\'):\n                    base = bytes([code - ord(\'A\') + 1])\n                elif ch == \'[\':\n                    base = b\'\\x1b\'\n                elif ch == \'\\\\\':\n                    base = b\'\\x1c\'\n                elif ch == \']\':\n                    base = b\'\\x1d\'\n                elif ch == \'^\':\n                    base = b\'\\x1e\'\n                elif ch == \'_\':\n                    base = b\'\\x1f\'\n                elif ch in (\'@\', \'`\', \' \'):\n                    base = b\'\\x00\'\n                else:\n                    return None\n            else:\n                named = token.lower()\n                if named in {\'space\', \'spc\'}:\n                    base = b\'\\x00\'\n                elif named in {\'tab\'}:\n                    base = b\'\\x09\'\n                elif named in {\'enter\', \'return\'}:\n                    base = b\'\\x0d\'\n                else:\n                    return None\n        else:\n            named = token.lower()\n            if named in {\'escape\', \'esc\'}:\n                base = b\'\\x1b\'\n            elif len(token) == 1:\n                base = token.encode(\'utf-8\', errors=\'replace\')\n            else:\n                return None\n\n        if alt:\n            return b\'\\x1b\' + base\n        return base\n\n    def _get_tmux_prefix_bytes(self) -> bytes | None:\n        if self._tmux_prefix_bytes is not None:\n            return self._tmux_prefix_bytes\n        try:\n            prefix = self.get_tmux_prefix()\n        except Exception:\n            return None\n        self._tmux_prefix_bytes = self._tmux_key_notation_to_bytes(prefix)\n        return self._tmux_prefix_bytes\n\n    def _send_tmux_prefix_now(self):\n        target = (self.last_focused_target or \'\').strip() or \'\'\n        args = [\'send-prefix\']\n        if target:\n            args.extend([\'-t\', target])\n        self._send_control_command(\' \'.join(args))\n        self._drain_control_responses_nowait()\n\n    def send_tmux_prefix(self):\n        self._flush_pending_sendkeys()\n        self._send_tmux_prefix_now()\n\n    def _clear_pending_sendkeys_handle(self):\n        handle = self._pending_sendkeys_flush_handle\n        self._pending_sendkeys_flush_handle = None\n        if handle is None:\n            return\n        try:\n            handle.cancel()\n        except Exception:\n            pass\n\n    def _flush_pending_sendkeys(self):\n        self._clear_pending_sendkeys_handle()\n        if not self._pending_sendkeys_bytes:\n            return\n        data = bytes(self._pending_sendkeys_bytes)\n        self._pending_sendkeys_bytes.clear()\n        self._send_input_bytes_now(data)\n\n    def _flush_pending_sendkeys_from_timer(self):\n        self._pending_sendkeys_flush_handle = None\n        self._flush_pending_sendkeys()\n\n    @staticmethod\n    def _is_batchable_sendkeys_payload(data: bytes) -> bool:\n        if not data:\n            return False\n        # Batch only plain printable ASCII text; flush control keys and escape\n        # sequences immediately to preserve terminal responsiveness.\n        return all(0x20 <= byte <= 0x7E for byte in data)\n\n    def _queue_sendkeys_microbatch(self, data: bytes):\n        self._pending_sendkeys_bytes.extend(data)\n        if len(self._pending_sendkeys_bytes) >= 128:\n            self._flush_pending_sendkeys()\n            return\n        if self._pending_sendkeys_flush_handle is not None:\n            return\n        try:\n            loop = asyncio.get_running_loop()\n        except RuntimeError:\n            self._flush_pending_sendkeys()\n            return\n        self._pending_sendkeys_flush_handle = loop.call_later(0.006, self._flush_pending_sendkeys_from_timer)\n\n    def write_input(self, message):\n        if isinstance(message, str):\n            data = message.encode(\'utf-8\', errors=\'replace\')\n        elif isinstance(message, memoryview):\n            data = message.tobytes()\n        else:\n            data = bytes(message)\n\n        if not data:\n            return\n\n        prefix_bytes = self._get_tmux_prefix_bytes()\n        if prefix_bytes and data == prefix_bytes:\n            self._flush_pending_sendkeys()\n            self._send_tmux_prefix_now()\n            return\n\n        if self._is_batchable_sendkeys_payload(data):\n            self._queue_sendkeys_microbatch(data)\n            return\n        self._flush_pending_sendkeys()\n        self._send_input_bytes_now(data)\n\n    def resize(self, cols: int, rows: int, force_refresh: bool = False):\n        self._flush_pending_sendkeys()\n        cols = max(2, int(cols))\n        rows = max(2, int(rows))\n        if cols > self.MAX_REASONABLE_COLS or rows > self.MAX_REASONABLE_ROWS:\n            return\n        if (cols, rows) == self.last_client_size and not force_refresh:\n            return\n        self.last_client_size = (cols, rows)\n\n        if self.master_fd is not None and self.slave_fd is not None:\n            size = struct.pack(\'HHHH\', rows, cols, 0, 0)\n            try:\n                fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, size)\n                fcntl.ioctl(self.slave_fd, termios.TIOCSWINSZ, size)\n            except OSError:\n                pass\n\n        if self.process is not None and self.process.poll() is None:\n            try:\n                os.killpg(os.getpgid(self.process.pid), signal.SIGWINCH)\n            except OSError:\n                pass\n            try:\n                client_target = self._get_client_target()\n                if client_target:\n                    self._run(\n                        [\'tmux\', \'refresh-client\', \'-C\', f\'{cols}x{rows}\', \'-t\', client_target],\n                        check=False,\n                        stdout=subprocess.DEVNULL,\n                        stderr=subprocess.DEVNULL,\n                    )\n                    if force_refresh:\n                        self._run(\n                            [\'tmux\', \'refresh-client\', \'-t\', client_target],\n                            check=False,\n                            stdout=subprocess.DEVNULL,\n                            stderr=subprocess.DEVNULL,\n                        )\n            except _SUBPROCESS_FAILURES:\n                pass\n\n\n\ndef create_terminal_backend(session_name: str = \'\', backend_mode: str | None = None) -> TerminalBackendAdapter:\n    mode = resolve_tmux_backend_mode(backend_mode)\n    if mode == \'control\':\n        return TmuxControlAdapter(session_name=session_name)\n    raise RuntimeError(f\'Unsupported backend mode: {mode}\')\n',
    'tmux_bridge.py': "import asyncio\nimport fcntl\nimport os\nimport pty\nimport re\nimport select\nimport signal\nimport struct\nimport subprocess\nimport termios\nimport time\nimport threading\nfrom typing import Dict, List, Set\n\n_SUBPROCESS_FAILURES = (\n    subprocess.CalledProcessError,\n    FileNotFoundError,\n    subprocess.TimeoutExpired,\n)\n\n\nclass TmuxBridge:\n    TAKEOVER_RECLAIM_HOLD_SECONDS = 3.0\n    MAX_REASONABLE_COLS = 1000\n    MAX_REASONABLE_ROWS = 400\n    TMUX_QUERY_TIMEOUT_SECONDS = 2.0\n    TMUX_MUTATION_TIMEOUT_SECONDS = 5.0\n    GIT_TIMEOUT_SECONDS = 15.0\n    PROCESS_QUERY_TIMEOUT_SECONDS = 2.0\n\n    def __init__(self, session_name: str = ''):\n        self.session_name = (session_name or '').strip()\n        self.home_dir = os.path.expanduser('~')\n        self.master_fd = None\n        self.slave_fd = None\n        self.process = None\n        self._closed_event = threading.Event()\n        self.last_focused_target = ''\n        self.saved_layout_by_window: Dict[str, str] = {}\n        self.saved_size_by_window: Dict[str, tuple[int, int]] = {}\n        self.last_external_release_by_window: Dict[str, float] = {}\n        self.takeover_reclaim_hold_until_by_window: Dict[str, float] = {}\n        self.last_client_size: tuple[int, int] = (80, 24)\n        self.tmux_env = os.environ.copy()\n        self.tmux_env.pop('TMUX', None)\n        self.tmux_env.pop('TMUX_PANE', None)\n        if not self.tmux_env.get('TERM') or self.tmux_env.get('TERM') == 'dumb':\n            self.tmux_env['TERM'] = 'xterm-256color'\n\n    def _command_timeout(self, args: list[str] | tuple[str, ...]) -> float:\n        if not args:\n            return self.TMUX_MUTATION_TIMEOUT_SECONDS\n        command = args[0]\n        if command == 'git':\n            return self.GIT_TIMEOUT_SECONDS\n        if command == 'ps':\n            return self.PROCESS_QUERY_TIMEOUT_SECONDS\n        if command != 'tmux' or len(args) < 2:\n            return self.TMUX_MUTATION_TIMEOUT_SECONDS\n        tmux_cmd = args[1]\n        if tmux_cmd in {\n            'display-message',\n            'list-clients',\n            'list-panes',\n            'list-sessions',\n            'list-windows',\n            'show-options',\n            'show-window-options',\n            'has-session',\n            'capture-pane',\n        }:\n            return self.TMUX_QUERY_TIMEOUT_SECONDS\n        return self.TMUX_MUTATION_TIMEOUT_SECONDS\n\n    def _run(\n        self,\n        args: list[str],\n        *,\n        check: bool = False,\n        text: bool = False,\n        stdout=None,\n        stderr=None,\n        timeout: float | None = None,\n    ):\n        return subprocess.run(\n            args,\n            env=self.tmux_env,\n            check=check,\n            text=text,\n            stdout=stdout,\n            stderr=stderr,\n            timeout=self._command_timeout(args) if timeout is None else timeout,\n        )\n\n    def _check_call(self, args: list[str], *, stdout=None, stderr=None, timeout: float | None = None):\n        return subprocess.check_call(\n            args,\n            env=self.tmux_env,\n            stdout=stdout,\n            stderr=stderr,\n            timeout=self._command_timeout(args) if timeout is None else timeout,\n        )\n\n    def _check_output(\n        self,\n        args: list[str],\n        *,\n        text: bool = False,\n        stderr=None,\n        timeout: float | None = None,\n    ):\n        return subprocess.check_output(\n            args,\n            env=self.tmux_env,\n            text=text,\n            stderr=stderr,\n            timeout=self._command_timeout(args) if timeout is None else timeout,\n        )\n\n    def is_process_alive(self) -> bool:\n        if self.process is None:\n            return True\n        try:\n            return self.process.poll() is None\n        except Exception:\n            return False\n\n    def open(self):\n        self._closed_event.clear()\n        self.master_fd, self.slave_fd = pty.openpty()\n\n        if not self.session_name:\n            self.session_name = self._resolve_or_create_session_name()\n        else:\n            try:\n                self._check_call(\n                    ['tmux', 'has-session', '-t', self.session_name],\n                    stdout=subprocess.DEVNULL,\n                    stderr=subprocess.DEVNULL,\n                )\n            except subprocess.CalledProcessError:\n                self._check_call(\n                    ['tmux', 'new-session', '-d', '-s', self.session_name],\n                    stdout=subprocess.DEVNULL,\n                    stderr=subprocess.DEVNULL,\n                )\n\n        # Keep the session alive when the app disconnects, even if the host\n        # tmux config enables aggressive cleanup of unattached sessions.\n        try:\n            self._run(\n                ['tmux', 'set-option', '-t', self.session_name, 'destroy-unattached', 'off'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            self._run(\n                ['tmux', 'set-option', '-s', 'exit-unattached', 'off'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            # Reduce redraw/input latency for the hidden tmux client attached to our PTY.\n            self._run(\n                ['tmux', 'set-option', '-s', 'tty-update-time', '0'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            self._run(\n                ['tmux', 'set-option', '-s', 'escape-time', '0'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            # Experiment: reduce tmux redraw churn for the hidden backend-attached client.\n            self._run(\n                ['tmux', 'set-option', '-t', self.session_name, 'status', 'off'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            self._run(\n                ['tmux', 'set-option', '-t', self.session_name, 'status-interval', '0'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            self._run(\n                ['tmux', 'set-option', '-t', self.session_name, 'mouse', 'off'],\n                check=False,\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n        except Exception:\n            pass\n\n        cols, rows = self._get_session_window_size()\n        self.last_client_size = (cols, rows)\n        initial_size = struct.pack('HHHH', rows, cols, 0, 0)\n        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, initial_size)\n        fcntl.ioctl(self.slave_fd, termios.TIOCSWINSZ, initial_size)\n\n        def _preexec():\n            os.setsid()\n            # Make the PTY slave the controlling terminal for the tmux client process.\n            fcntl.ioctl(self.slave_fd, termios.TIOCSCTTY, 0)\n\n        self.process = subprocess.Popen(\n            ['tmux', 'attach-session', '-t', self.session_name],\n            stdin=self.slave_fd,\n            stdout=self.slave_fd,\n            stderr=self.slave_fd,\n            preexec_fn=_preexec,\n            env=self.tmux_env,\n        )\n\n    def _resolve_or_create_session_name(self) -> str:\n        try:\n            output = self._check_output(\n                ['tmux', 'list-sessions', '-F', '#{session_name}'],\n                text=True,\n                stderr=subprocess.DEVNULL,\n            )\n            for line in output.splitlines():\n                candidate = line.strip()\n                if candidate:\n                    return candidate\n        except _SUBPROCESS_FAILURES:\n            pass\n\n        fallback = 'main'\n        self._check_call(\n            ['tmux', 'new-session', '-d', '-s', fallback],\n            stdout=subprocess.DEVNULL,\n            stderr=subprocess.DEVNULL,\n        )\n        return fallback\n\n    def _get_session_window_size(self) -> tuple[int, int]:\n        try:\n            output = self._check_output(\n                ['tmux', 'display-message', '-p', '-t', self.session_name, '#{window_width}\\t#{window_height}'],\n                text=True,\n            ).strip()\n            if not output:\n                return 80, 24\n            parts = output.split('\\t')\n            if len(parts) != 2:\n                return 80, 24\n            cols = int(parts[0])\n            rows = int(parts[1])\n            if (\n                cols < 2\n                or rows < 2\n                or cols > self.MAX_REASONABLE_COLS\n                or rows > self.MAX_REASONABLE_ROWS\n            ):\n                return 80, 24\n            return cols, rows\n        except (_SUBPROCESS_FAILURES, ValueError):\n            return 80, 24\n\n    def get_tmux_prefix(self) -> str:\n        if self.session_name:\n            try:\n                output = self._check_output(\n                    ['tmux', 'show-options', '-t', self.session_name, 'prefix'],\n                    text=True,\n                    stderr=subprocess.DEVNULL,\n                ).strip()\n                if output.startswith('prefix '):\n                    return output.split(' ', 1)[1]\n            except _SUBPROCESS_FAILURES:\n                pass\n\n        try:\n            output = self._check_output(\n                ['tmux', 'show-options', '-g', 'prefix'],\n                text=True,\n                stderr=subprocess.DEVNULL,\n            ).strip()\n            if output.startswith('prefix '):\n                return output.split(' ', 1)[1]\n        except _SUBPROCESS_FAILURES:\n            pass\n\n        return 'C-b'\n\n    async def read_output(self):\n        loop = asyncio.get_event_loop()\n        return await loop.run_in_executor(None, self._read_output_blocking)\n\n    def _read_output_blocking(self) -> bytes:\n        while not self._closed_event.is_set():\n            if self.master_fd is None:\n                return b''\n            if not self.is_process_alive():\n                return b''\n            try:\n                readable, _, _ = select.select([self.master_fd], [], [], 0.25)\n            except (OSError, ValueError):\n                return b''\n            if self.master_fd not in readable:\n                continue\n            try:\n                return os.read(self.master_fd, 4096)\n            except (BlockingIOError, InterruptedError):\n                continue\n            except OSError:\n                return b''\n        return b''\n\n    def write_input(self, message):\n        if isinstance(message, str):\n            os.write(self.master_fd, message.encode())\n        else:\n            os.write(self.master_fd, message)\n\n    def resize(self, cols: int, rows: int, force_refresh: bool = False):\n        cols = max(2, int(cols))\n        rows = max(2, int(rows))\n        if cols > self.MAX_REASONABLE_COLS or rows > self.MAX_REASONABLE_ROWS:\n            return\n        if (cols, rows) == self.last_client_size and not force_refresh:\n            return\n        self.last_client_size = (cols, rows)\n        size = struct.pack('HHHH', rows, cols, 0, 0)\n        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, size)\n        fcntl.ioctl(self.slave_fd, termios.TIOCSWINSZ, size)\n        if self.process is not None and self.process.poll() is None:\n            try:\n                os.killpg(os.getpgid(self.process.pid), signal.SIGWINCH)\n            except OSError:\n                pass\n            try:\n                client_target = self._get_client_target()\n                if client_target:\n                    self._run(\n                        ['tmux', 'refresh-client', '-C', f'{cols}x{rows}', '-t', client_target],\n                        check=False,\n                        stdout=subprocess.DEVNULL,\n                        stderr=subprocess.DEVNULL,\n                    )\n                    self._run(\n                        ['tmux', 'refresh-client', '-t', client_target],\n                        check=False,\n                        stdout=subprocess.DEVNULL,\n                        stderr=subprocess.DEVNULL,\n                    )\n\n                # After takeover/zoom flows, keep the focused window geometry\n                # aligned on subsequent plain resizes to avoid scrambled redraws\n                # in full-screen terminal UIs.\n                focused_target = (self.last_focused_target or '').strip()\n                if focused_target:\n                    window_target = self._check_output(\n                        ['tmux', 'display-message', '-p', '-t', focused_target, '#{window_id}'],\n                        text=True,\n                    ).strip()\n                    if window_target:\n                        self._run(\n                            ['tmux', 'resize-window', '-t', window_target, '-x', str(cols), '-y', str(rows)],\n                            check=False,\n                            stdout=subprocess.DEVNULL,\n                            stderr=subprocess.DEVNULL,\n                        )\n            except _SUBPROCESS_FAILURES:\n                pass\n\n    def force_takeover_resize(self, target: str, cols: int, rows: int) -> bool:\n        normalized_target = (target or '').strip()\n        if not normalized_target:\n            return False\n\n        effective_target = normalized_target\n        try:\n            self._check_output(\n                ['tmux', 'display-message', '-p', '-t', effective_target, '#{pane_id}'],\n                text=True,\n            )\n        except _SUBPROCESS_FAILURES:\n            fallback_target = (self.last_focused_target or '').strip()\n            if fallback_target:\n                try:\n                    self._check_output(\n                        ['tmux', 'display-message', '-p', '-t', fallback_target, '#{pane_id}'],\n                        text=True,\n                    )\n                    effective_target = fallback_target\n                except _SUBPROCESS_FAILURES:\n                    return False\n            else:\n                return False\n\n        try:\n            window_target = self._check_output(\n                ['tmux', 'display-message', '-p', '-t', effective_target, '#{window_id}'],\n                text=True,\n            ).strip()\n        except _SUBPROCESS_FAILURES:\n            return False\n\n        if not window_target:\n            return False\n\n        cols = max(2, int(cols))\n        rows = max(2, int(rows))\n        resized_window = False\n        self.takeover_reclaim_hold_until_by_window[window_target] = (\n            time.monotonic() + self.TAKEOVER_RECLAIM_HOLD_SECONDS\n        )\n\n        try:\n            self._check_call(\n                ['tmux', 'set-window-option', '-t', window_target, 'window-size', 'manual'],\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            self._check_call(\n                ['tmux', 'resize-window', '-t', window_target, '-x', str(cols), '-y', str(rows)],\n                stdout=subprocess.DEVNULL,\n                stderr=subprocess.DEVNULL,\n            )\n            resized_window = True\n        except _SUBPROCESS_FAILURES:\n            resized_window = False\n\n        # Ensure takeover target is rendered as a single zoomed pane.\n        try:\n            zoom_flag = self._check_output(\n                ['tmux', 'display-message', '-p', '-t', effective_target, '#{window_zoomed_flag}'],\n                text=True,\n            ).strip()\n            if zoom_flag != '1':\n                self._check_call(\n                    ['tmux', 'resize-pane', '-Z', '-t', effective_target],\n                    stdout=subprocess.DEVNULL,\n                    stderr=subprocess.DEVNULL,\n                )\n        except _SUBPROCESS_FAILURES:\n            pass\n\n        # Keep PTY dimensions/client refresh aligned with requested takeover size.\n        # Force refresh even when dimensions are unchanged so tmux redraws after\n        # ownership changes caused by external clients.\n        self.resize(cols, rows, force_refresh=True)\n        self.saved_size_by_window.pop(window_target, None)\n        return resized_window\n\n    def list_panes(self, session_name: str = '') -> List[Dict[str, str]]:\n        selected_session = (session_name or '').strip()\n        separator = ':::__TELETERM_PANE_SEP__:::'\n        pane_format = separator.join(\n            [\n                '#{pane_id}',\n                '#{session_name}',\n                '#{window_index}',\n                '#{window_name}',\n                '#{pane_index}',\n                '#{pane_title}',\n                '#{pane_current_command}',\n                '#{pane_active}',\n                '#{pane_current_path}',\n                '#{pane_pid}',\n                '#{pane_left}',\n                '#{pane_top}',\n                '#{pane_width}',\n                '#{pane_height}',\n            ]\n        )\n\n        # Default to listing all panes across all sessions for a comprehensive dashboard.\n        if not selected_session or selected_session == '*':\n            list_panes_args = ['tmux', 'list-panes', '-a', '-F', pane_format]\n        else:\n            list_panes_args = ['tmux', 'list-panes', '-t', selected_session, '-F', pane_format]\n\n        output = self._check_output(\n            list_panes_args,\n            text=True,\n        )\n\n        # Identify the pane currently visible to our specific tmux client.\n        client_pane_id = ''\n        client_target = self._get_client_target()\n        if client_target:\n            try:\n                client_pane_id = self._check_output(\n                    ['tmux', 'display-message', '-p', '-t', client_target, '#{pane_id}'],\n                    text=True,\n                ).strip()\n            except _SUBPROCESS_FAILURES:\n                pass\n\n        process_snapshot = self._get_process_snapshot()\n\n        panes: List[Dict[str, str]] = []\n        for line in output.splitlines():\n            if not line.strip():\n                continue\n\n            parts = line.split(separator)\n            if len(parts) != 14:\n                continue\n\n            pane_id = parts[0]\n            current_path = parts[8]\n            pane_pid = parts[9]\n            repo_basename, pane_title = self._build_repo_context(current_path)\n            worktree_created_at = self._get_worktree_created_at(current_path)\n            pane_agent = self._detect_agent_for_pane_pid(pane_pid, process_snapshot)\n\n            # A pane is truly active only if it is the one currently rendered by the bridge's client.\n            # Fallback to #{pane_active} (active in its window) only if we can't resolve the client's current pane.\n            is_active = (pane_id == client_pane_id) if client_pane_id else (parts[7] == '1')\n\n            if is_active:\n                self.last_focused_target = pane_id\n\n            panes.append(\n                {\n                    'paneId': pane_id,\n                    'sessionName': parts[1],\n                    'windowIndex': parts[2],\n                    'windowName': parts[3],\n                    'paneIndex': parts[4],\n                    'paneTitle': pane_title,\n                    'currentCommand': parts[6],\n                    'isActive': is_active,\n                    'target': pane_id,\n                    'label': f'{parts[1]}:{parts[2]}.{parts[4]}',\n                    'agent': pane_agent,\n                    'repoBasename': repo_basename,\n                    'currentPath': current_path,\n                    'worktreeCreatedAt': worktree_created_at,\n                    'paneLeft': parts[10],\n                    'paneTop': parts[11],\n                    'paneWidth': parts[12],\n                    'paneHeight': parts[13],\n                }\n            )\n\n        def _try_int(val, default=0):\n            try:\n                return int(val)\n            except (TypeError, ValueError):\n                return default\n\n        panes.sort(\n            key=lambda pane: (\n                str(pane.get('repoBasename', '')).lower(),\n                int(pane.get('worktreeCreatedAt', 0)),\n                str(pane.get('sessionName', '')).lower(),\n                _try_int(pane.get('windowIndex')),\n                _try_int(pane.get('paneIndex')),\n            )\n        )\n        return panes\n\n    def _build_repo_context(self, current_path: str) -> tuple[str, str]:\n        if not current_path:\n            return '', ''\n\n        repo_basename = ''\n        try:\n            repo_root = self._check_output(\n                ['git', '-C', current_path, 'rev-parse', '--show-toplevel'],\n                text=True,\n                stderr=subprocess.DEVNULL,\n            ).strip()\n            if repo_root:\n                repo_basename = os.path.basename(repo_root.rstrip('/'))\n        except _SUBPROCESS_FAILURES:\n            repo_basename = ''\n\n        if repo_basename:\n            return repo_basename, f'{repo_basename} | {current_path}'\n        return '', current_path\n\n    def _get_worktree_created_at(self, current_path: str) -> int:\n        if not current_path:\n            return 0\n\n        try:\n            git_dir = self._check_output(\n                ['git', '-C', current_path, 'rev-parse', '--git-dir'],\n                text=True,\n                stderr=subprocess.DEVNULL,\n            ).strip()\n        except _SUBPROCESS_FAILURES:\n            return 0\n\n        if not git_dir:\n            return 0\n\n        if os.path.isabs(git_dir):\n            git_dir_path = git_dir\n        else:\n            git_dir_path = os.path.abspath(os.path.join(current_path, git_dir))\n\n        try:\n            # Linux has no portable birth-time; ctime is the closest available signal.\n            return int(os.stat(git_dir_path).st_ctime)\n        except OSError:\n            return 0\n\n    def _get_process_snapshot(self) -> Dict[str, Dict[int, object]]:\n        try:\n            output = self._check_output(\n                ['ps', '-eo', 'pid=,ppid=,args='],\n                text=True,\n            )\n        except _SUBPROCESS_FAILURES:\n            return {'args_by_pid': {}, 'children_by_ppid': {}}\n\n        args_by_pid: Dict[int, str] = {}\n        children_by_ppid: Dict[int, List[int]] = {}\n        for line in output.splitlines():\n            stripped = line.strip()\n            if not stripped:\n                continue\n\n            parts = stripped.split(None, 2)\n            if len(parts) < 2:\n                continue\n\n            try:\n                pid = int(parts[0])\n                ppid = int(parts[1])\n            except ValueError:\n                continue\n\n            args = parts[2] if len(parts) == 3 else ''\n            args_by_pid[pid] = args\n            children_by_ppid.setdefault(ppid, []).append(pid)\n\n        return {'args_by_pid': args_by_pid, 'children_by_ppid': children_by_ppid}\n\n    def _detect_agent_for_pane_pid(self, pane_pid: str, process_snapshot: Dict[str, Dict[int, object]]) -> str:\n        try:\n            root_pid = int(pane_pid)\n        except (TypeError, ValueError):\n            return ''\n\n        args_by_pid = process_snapshot.get('args_by_pid', {})\n        children_by_ppid = process_snapshot.get('children_by_ppid', {})\n        if not isinstance(args_by_pid, dict) or not isinstance(children_by_ppid, dict):\n            return ''\n\n        queue: List[int] = [root_pid]\n        seen: Set[int] = set()\n        aggregated_args: List[str] = []\n        while queue:\n            pid = queue.pop(0)\n            if pid in seen:\n                continue\n            seen.add(pid)\n\n            cmdline = args_by_pid.get(pid)\n            if isinstance(cmdline, str):\n                aggregated_args.append(cmdline.lower())\n\n            for child in children_by_ppid.get(pid, []):\n                if isinstance(child, int) and child not in seen:\n                    queue.append(child)\n\n        detected = self._match_agent_name(aggregated_args)\n        return detected or ''\n\n    def _match_agent_name(self, command_lines: List[str]) -> str:\n        patterns = (\n            ('codex', re.compile(r'(^|[\\s/])codex($|[\\s])')),\n            ('gemini', re.compile(r'(^|[\\s/])gemini($|[\\s])')),\n            ('claude', re.compile(r'(^|[\\s/])claude($|[\\s])')),\n            ('opencode', re.compile(r'(^|[\\s/])opencode($|[\\s])')),\n        )\n\n        for command_line in command_lines:\n            for agent, pattern in patterns:\n                if pattern.search(command_line):\n                    return agent\n        return ''\n\n    def _get_client_target(self) -> str:\n        if self.process is None:\n            return ''\n\n        try:\n            output = self._check_output(\n                ['tmux', 'list-clients', '-F', '#{client_pid}\\t#{client_tty}'],\n                text=True,\n            )\n        except _SUBPROCESS_FAILURES:\n            return ''\n        process_pid = str(self.process.pid)\n        for line in output.splitlines():\n            parts = line.split('\\t')\n            if len(parts) != 2:\n                continue\n            if parts[0] == process_pid:\n                return parts[1]\n\n        return ''\n\n    def has_recent_external_client_activity(self, max_age_seconds: int = 3) -> bool:\n        if self.process is None:\n            return False\n\n        try:\n            output = self._check_output(\n                ['tmux', 'list-clients', '-F', '#{client_pid}\\t#{client_activity}'],\n                text=True,\n            )\n        except _SUBPROCESS_FAILURES:\n            return False\n\n        own_pid = str(self.process.pid)\n        now = int(time.time())\n        for line in output.splitlines():\n            parts = line.split('\\t')\n            if len(parts) != 2:\n                continue\n            pid, activity = parts\n            if pid == own_pid:\n                continue\n            try:\n                activity_ts = int(activity)\n            except ValueError:\n                continue\n            if now - activity_ts <= max(1, int(max_age_seconds)):\n                return True\n        return False\n\n    def get_recent_external_client_pane_target(self, max_age_seconds: int = 3) -> str:\n        if self.process is None:\n            return ''\n\n        try:\n            output = self._check_output(\n                ['tmux', 'list-clients', '-F', '#{client_pid}\\t#{client_activity}\\t#{pane_id}'],\n                text=True,\n            )\n        except _SUBPROCESS_FAILURES:\n            return ''\n\n        own_pid = str(self.process.pid)\n        now = int(time.time())\n        newest_activity = -1\n        newest_target = ''\n        max_age = max(1, int(max_age_seconds))\n        for line in output.splitlines():\n            parts = line.split('\\t')\n            if len(parts) != 3:\n                continue\n            pid, activity, target = parts\n            if pid == own_pid:\n                continue\n            try:\n                activity_ts = int(activity)\n            except ValueError:\n                continue\n            if now - activity_ts > max_age:\n                continue\n            if not target.startswith(f'{self.session_name}:'):\n                continue\n            if activity_ts > newest_activity:\n                newest_activity = activity_ts\n                newest_target = target\n\n        return newest_target\n\n    def release_window_size_lock_for_target(self, target: str, min_interval_seconds: float = 1.0) -> bool:\n        normalized_target = (target or '').strip()\n        if not normalized_target:\n            return False\n        window_target = normalized_target.rsplit('.', 1)[0] if '.' in normalized_target else normalized_target\n        if not window_target:\n            return False\n\n        now_mono = time.monotonic()\n        reclaim_hold_until = self.takeover_reclaim_hold_until_by_window.get(window_target, 0.0)\n        if now_mono < reclaim_hold_until:\n            return False\n        last_release = self.last_external_release_by_window.get(window_target, 0.0)\n        if now_mono - last_release < max(0.1, float(min_interval_seconds)):\n            return False\n\n        self._run(\n            ['tmux', 'set-window-option', '-t', window_target, 'window-size', 'largest'],\n            check=False,\n            stdout=subprocess.DEVNULL,\n            stderr=subprocess.DEVNULL,\n        )\n        # Apply the new window-size policy immediately so external desktop\n        # clients reclaim dimensions without waiting for a later redraw.\n        self._run(\n            ['tmux', 'resize-window', '-A', '-t', window_target],\n            check=False,\n            stdout=subprocess.DEVNULL,\n            stderr=subprocess.DEVNULL,\n        )\n        self.last_external_release_by_window[window_target] = now_mono\n\n        # Drop stale snapshot once desktop/manual takeover happens.\n        self.saved_size_by_window.pop(window_target, None)\n        return True\n\n    def focus_pane(self, target: str):\n        # Resolve full session:window context from the target (which can now be a pane_id like %123).\n        try:\n            context = self._check_output(\n                ['tmux', 'display-message', '-p', '-t', target, '#{session_name}\\t#{session_name}:#{window_index}'],\n                text=True,\n            ).strip().split('\\t')\n            if len(context) == 2:\n                session_target = context[0]\n                window_target = context[1]\n            else:\n                session_target = self.session_name\n                window_target = target\n        except _SUBPROCESS_FAILURES:\n            session_target = self.session_name\n            window_target = target\n\n        client_target = self._get_client_target()\n\n        if client_target:\n            self._check_call(['tmux', 'switch-client', '-c', client_target, '-t', session_target])\n\n        self._check_call(['tmux', 'select-window', '-t', window_target])\n        self._check_call(['tmux', 'select-pane', '-t', target])\n\n        zoom_flag = self._check_output(\n            ['tmux', 'display-message', '-p', '-t', target, '#{window_zoomed_flag}'],\n            text=True,\n        ).strip()\n        if zoom_flag == '0':\n            current_layout = self._check_output(\n                ['tmux', 'display-message', '-p', '-t', target, '#{window_layout}'],\n                text=True,\n            ).strip()\n            size_output = self._check_output(\n                ['tmux', 'display-message', '-p', '-t', target, '#{window_width}\\t#{window_height}'],\n                text=True,\n            ).strip()\n            if current_layout:\n                self.saved_layout_by_window[window_target] = current_layout\n            if size_output:\n                parts = size_output.split('\\t')\n                if len(parts) == 2:\n                    try:\n                        self.saved_size_by_window[window_target] = (max(2, int(parts[0])), max(2, int(parts[1])))\n                    except ValueError:\n                        pass\n            self._check_call(['tmux', 'resize-pane', '-Z', '-t', target])\n        self.last_focused_target = target\n\n        # Resize the window to match the mobile client's viewport so that\n        # content immediately flows at the correct column width. Without this,\n        # tmux renders the zoomed pane at the window's previous (often wider)\n        # dimensions, causing right-side clipping on smaller screens.\n        cols, rows = self.last_client_size\n        try:\n            self._check_call(\n                ['tmux', 'resize-window', '-t', window_target, '-x', str(cols), '-y', str(rows)],\n            )\n        except _SUBPROCESS_FAILURES:\n            pass\n        self.resize(cols, rows)\n\n        if client_target:\n            self._check_call(['tmux', 'refresh-client', '-t', client_target])\n\n    def unzoom_last_focused_pane(self):\n        client_target = self._get_client_target()\n        session_name = self._get_client_session_name(client_target)\n        target = self.last_focused_target or self._find_zoomed_pane_target(session_name=session_name)\n        if not target:\n            return\n\n        zoom_flag = self._check_output(\n            ['tmux', 'display-message', '-p', '-t', target, '#{window_zoomed_flag}'],\n            text=True,\n        ).strip()\n        if zoom_flag != '1':\n            target = self._find_zoomed_pane_target(session_name=session_name)\n            if not target:\n                return\n            zoom_flag = self._check_output(\n                ['tmux', 'display-message', '-p', '-t', target, '#{window_zoomed_flag}'],\n                text=True,\n            ).strip()\n            if zoom_flag != '1':\n                return\n\n        window_target = target.rsplit('.', 1)[0] if '.' in target else target\n\n        self._check_call(['tmux', 'resize-pane', '-Z', '-t', target])\n\n        # Prefer restoring the pre-zoom dimensions we captured so external clients\n        # return to the prior layout immediately after mobile unzoom.\n        saved_size = self.saved_size_by_window.get(window_target)\n        if saved_size:\n            cols, rows = saved_size\n            self._check_call(\n                ['tmux', 'resize-window', '-t', window_target, '-x', str(cols), '-y', str(rows)],\n            )\n        else:\n            # Fallback when no snapshot exists (e.g. zoom state predated this bridge).\n            self._run(\n                ['tmux', 'set-window-option', '-t', window_target, 'window-size', 'largest'],\n                check=False,\n            )\n\n        saved_layout = self.saved_layout_by_window.get(window_target, '')\n        if saved_layout:\n            self._check_call(['tmux', 'select-layout', '-t', window_target, saved_layout])\n\n        if client_target:\n            self._check_call(['tmux', 'refresh-client', '-t', client_target])\n\n    def rename_session(self, old_name: str, new_name: str):\n        old_name = (old_name or '').strip()\n        new_name = (new_name or '').strip()\n        if not old_name or not new_name:\n            raise ValueError('Both old_name and new_name are required')\n        if old_name == new_name:\n            return\n\n        self._check_call(['tmux', 'rename-session', '-t', old_name, new_name])\n        if self.session_name == old_name:\n            self.session_name = new_name\n\n    def create_pane(\n        self,\n        agent: str = '',\n        path: str = '',\n        session_name: str = '',\n        command: str = '',\n        resize_window: bool = True,\n        split_target: str = '',\n    ) -> str:\n        normalized_agent = (agent or '').strip().lower()\n        if normalized_agent not in {'', 'codex', 'claude', 'gemini', 'opencode'}:\n            raise ValueError(f'Unsupported agent: {normalized_agent}')\n\n        selected_path = (path or '').strip()\n        if selected_path:\n            selected_path = os.path.expanduser(selected_path)\n        else:\n            selected_path = self.home_dir\n\n        requested_session = (session_name or '').strip()\n        selected_command = (command or '').strip()\n        if not selected_command and normalized_agent:\n            selected_command = f'exec {normalized_agent}'\n\n        target_session = requested_session or self.session_name\n        return self.create_window(\n            path=selected_path,\n            session_name=target_session,\n            command=selected_command,\n        )\n\n    def create_window(self, path: str = '', session_name: str = '', command: str = '') -> str:\n        selected_path = (path or '').strip()\n        if selected_path:\n            selected_path = os.path.expanduser(selected_path)\n        else:\n            selected_path = self.home_dir\n\n        selected_session_name = (session_name or '').strip() or self.session_name\n        args = [\n            'tmux',\n            'new-window',\n            '-c',\n            selected_path,\n            '-t',\n            selected_session_name,\n            '-P',\n            '-F',\n            '#{pane_id}',\n        ]\n        selected_command = (command or '').strip()\n        if selected_command:\n            args.append(selected_command)\n\n        target = self._check_output(\n            args,\n            text=True,\n        ).strip()\n        if not target:\n            raise RuntimeError('Failed to create window')\n        return target\n\n    def create_worktree(self, repo_path: str, worktree_name: str = '', worktree_location: str = '') -> str:\n        selected_repo_path = (repo_path or '').strip()\n        if not selected_repo_path:\n            raise ValueError('repo_path is required')\n\n        selected_repo_path = os.path.abspath(os.path.expanduser(selected_repo_path))\n        if not os.path.isdir(selected_repo_path):\n            raise ValueError(f'Repository path does not exist: {selected_repo_path}')\n\n        selected_worktree_name = (worktree_name or '').strip()\n        selected_worktree_location = (worktree_location or '').strip()\n        if selected_worktree_location:\n            selected_worktree_location = os.path.abspath(os.path.expanduser(selected_worktree_location))\n            if not os.path.isdir(selected_worktree_location):\n                raise ValueError(f'Worktree location does not exist: {selected_worktree_location}')\n\n        if selected_worktree_name and (os.sep in selected_worktree_name or (os.altsep and os.altsep in selected_worktree_name)):\n            raise ValueError('Worktree repo name must not contain path separators')\n\n        if selected_worktree_name:\n            parent_path = selected_worktree_location or os.path.dirname(selected_repo_path.rstrip(os.sep)) or os.sep\n            worktree_path = os.path.join(parent_path, selected_worktree_name)\n            if os.path.exists(worktree_path):\n                raise ValueError(f'Worktree path already exists: {worktree_path}')\n        else:\n            parent_path = selected_worktree_location or os.path.dirname(selected_repo_path.rstrip(os.sep)) or os.sep\n            repo_name = os.path.basename(selected_repo_path.rstrip(os.sep)) or 'repo'\n            timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())\n            base_path = os.path.join(parent_path, f'{repo_name}-wt-{timestamp}')\n            worktree_path = base_path\n            suffix = 1\n            while os.path.exists(worktree_path):\n                worktree_path = f'{base_path}-{suffix}'\n                suffix += 1\n\n        self._check_call(\n            ['git', '-C', selected_repo_path, 'worktree', 'add', worktree_path],\n        )\n        return worktree_path\n\n    def _resize_window_target(self, window_target: str):\n        cols, rows = self.last_client_size\n        try:\n            self._check_call(\n                ['tmux', 'resize-window', '-t', window_target, '-x', str(cols), '-y', str(rows)],\n            )\n        except _SUBPROCESS_FAILURES:\n            pass\n\n    def _get_default_pane_target_for_session(self, session_name: str) -> str:\n        normalized_session = (session_name or '').strip()\n        if not normalized_session:\n            return ''\n\n        try:\n            output = self._check_output(\n                ['tmux', 'list-panes', '-t', normalized_session, '-F', '#{pane_id}'],\n                text=True,\n            )\n        except _SUBPROCESS_FAILURES:\n            return ''\n        for line in output.splitlines():\n            target = line.strip()\n            if target.startswith('%'):\n                return target\n        return ''\n\n    def _get_client_session_name(self, client_target: str) -> str:\n        if not client_target:\n            return ''\n        try:\n            return self._check_output(\n                ['tmux', 'display-message', '-p', '-t', client_target, '#{session_name}'],\n                text=True,\n            ).strip()\n        except _SUBPROCESS_FAILURES:\n            return ''\n\n    def _find_zoomed_pane_target(self, session_name: str = '') -> str:\n        try:\n            output = self._check_output(\n                ['tmux', 'list-windows', '-a', '-F', '#{pane_id}\\t#{window_zoomed_flag}\\t#{session_name}'],\n                text=True,\n            )\n        except _SUBPROCESS_FAILURES:\n            return ''\n        for line in output.splitlines():\n            parts = line.split('\\t')\n            if len(parts) != 3:\n                continue\n            target, zoomed, s_name = parts\n            if session_name:\n                if s_name != session_name:\n                    continue\n            elif s_name != self.session_name:\n                continue\n            if zoomed == '1':\n                return target\n        return ''\n\n    def capture_pane(self, target: str) -> bytes:\n        normalized_target = (target or '').strip()\n        if not normalized_target:\n            return b''\n\n        try:\n            # -e includes escape sequences (colors, etc), -p outputs to stdout\n            output = self._check_output(\n                ['tmux', 'capture-pane', '-ep', '-t', normalized_target],\n            )\n            return output\n        except _SUBPROCESS_FAILURES:\n            return b''\n\n    def close(self):\n        self._closed_event.set()\n        # Restore all windows we resized or zoomed\n        targets = set(self.saved_size_by_window.keys()) | set(self.saved_layout_by_window.keys())\n        for window_target in targets:\n            try:\n                # 1. Restore the pre-mobile dimensions when available.\n                saved_size = self.saved_size_by_window.get(window_target)\n                if saved_size:\n                    cols, rows = saved_size\n                    self._run(\n                        ['tmux', 'resize-window', '-t', window_target, '-x', str(cols), '-y', str(rows)],\n                        check=False,\n                        stdout=subprocess.DEVNULL,\n                        stderr=subprocess.DEVNULL,\n                    )\n                else:\n                    # Fallback to automatic sizing if we do not have a snapshot.\n                    self._run(\n                        ['tmux', 'set-window-option', '-t', window_target, 'window-size', 'largest'],\n                        check=False,\n                        stdout=subprocess.DEVNULL,\n                        stderr=subprocess.DEVNULL,\n                    )\n\n                # 2. If any pane is zoomed in this window, unzoom it\n                try:\n                    output = self._check_output(\n                        ['tmux', 'list-panes', '-t', window_target, '-F', '#{pane_id}\\t#{window_zoomed_flag}'],\n                        text=True,\n                        stderr=subprocess.DEVNULL,\n                    )\n                    for line in output.splitlines():\n                        parts = line.split('\\t')\n                        if len(parts) == 2 and parts[1] == '1':\n                            self._run(\n                                ['tmux', 'resize-pane', '-Z', '-t', parts[0]],\n                                check=False,\n                                stdout=subprocess.DEVNULL,\n                                stderr=subprocess.DEVNULL,\n                            )\n                except _SUBPROCESS_FAILURES:\n                    pass\n\n                # 3. Restore saved layout if we have one\n                saved_layout = self.saved_layout_by_window.get(window_target)\n                if saved_layout:\n                    self._run(\n                        ['tmux', 'select-layout', '-t', window_target, saved_layout],\n                        check=False,\n                        stdout=subprocess.DEVNULL,\n                        stderr=subprocess.DEVNULL,\n                    )\n            except Exception:\n                pass\n\n        if self.master_fd is not None:\n            try:\n                os.close(self.master_fd)\n            except OSError:\n                pass\n            self.master_fd = None\n\n        if self.slave_fd is not None:\n            try:\n                os.close(self.slave_fd)\n            except OSError:\n                pass\n            self.slave_fd = None\n\n        if self.process is not None:\n            try:\n                # Try to terminate gracefully first\n                self.process.terminate()\n                # Give it a tiny bit of time\n                time.sleep(0.05)\n                if self.process.poll() is None:\n                    self.process.kill()\n            except OSError:\n                pass\n            self.process = None\n",
}
BACKEND_MODULE_FILES = (
    'auth.py',
    'config.py',
    'server.py',
    'session.py',
    'terminal_backend.py',
    'tmux_bridge.py',
)


def script_path() -> Path:
    return Path(__file__).resolve()


def script_dir() -> Path:
    return script_path().parent


def embedded_backend_dir() -> Path:
    return state_dir() / '.embedded_backend'


def ensure_embedded_backend_file(filename: str) -> Path:
    source = EMBEDDED_BACKEND_MODULES.get(filename)
    if source is None:
        return script_dir() / filename

    target_dir = embedded_backend_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(target_dir, 0o700)
    target_path = target_dir / filename
    if not target_path.exists() or target_path.read_text(encoding='utf-8') != source:
        target_path.write_text(source, encoding='utf-8')
    os.chmod(target_path, 0o600)
    return target_path


def import_backend_module(name: str):
    # Monolithic installs embed sibling backend modules and materialize them on demand.
    ensure_embedded_backend_file(f'{name}.py')
    module_parent = str(embedded_backend_dir() if EMBEDDED_BACKEND_MODULES else script_dir())
    if module_parent not in sys.path:
        sys.path.insert(0, module_parent)
    return importlib.import_module(name)


def default_state_dir() -> Path:
    return Path(os.getenv('CLAWSSH_STATE_DIR', '~/.clawssh')).expanduser()


def state_dir() -> Path:
    return default_state_dir()


def port_file() -> Path:
    return Path(os.getenv('CLAWSSH_PORT_FILE', str(state_dir() / 'backend_port'))).expanduser()


def ca_cert_file() -> Path:
    return Path(os.getenv('CLAWSSH_TLS_CA_CERT_FILE', str(state_dir() / 'ca.crt'))).expanduser()


def certificate_fingerprint_from_pem_file(cert_path: str | os.PathLike[str]) -> str:
    import hashlib
    import ssl

    pem_text = Path(cert_path).read_text(encoding='utf-8')
    certificate_der = ssl.PEM_cert_to_DER_cert(pem_text)
    digest = hashlib.sha256(certificate_der).hexdigest().upper()
    return ':'.join(digest[i:i + 2] for i in range(0, len(digest), 2))


def ca_key_file() -> Path:
    return state_dir() / 'ca.key'


def server_cert_file() -> Path:
    return Path(os.getenv('CLAWSSH_TLS_CERT_FILE', str(state_dir() / 'server.crt'))).expanduser()


def server_key_file() -> Path:
    return Path(os.getenv('CLAWSSH_TLS_KEY_FILE', str(state_dir() / 'server.key'))).expanduser()


def device_dir() -> Path:
    return state_dir() / 'devices'


def _set_default_port_from_state() -> None:
    if os.getenv('CLAWSSH_PORT'):
        return
    os.environ['CLAWSSH_PORT'] = '8765'
    saved_port_path = port_file()
    if not saved_port_path.is_file():
        return
    candidate = saved_port_path.read_text(encoding='utf-8').strip()
    if candidate.isdigit() and 1 <= int(candidate) <= 65535:
        os.environ['CLAWSSH_PORT'] = candidate


def current_port() -> int:
    _set_default_port_from_state()
    return int(os.environ['CLAWSSH_PORT'])


def ensure_python_package(package_name: str, import_name: str | None = None) -> None:
    module_name = import_name or package_name
    if importlib.util.find_spec(module_name) is not None:
        return

    print(f'Installing missing Python package: {package_name}')
    pip_command = ['python3', '-m', 'pip', 'install', '--user', package_name]
    result = subprocess.run(pip_command, check=False)
    if result.returncode == 0 and importlib.util.find_spec(module_name) is not None:
        return

    ensurepip_command = ['python3', '-m', 'ensurepip', '--user']
    ensurepip_result = subprocess.run(ensurepip_command, check=False)
    if ensurepip_result.returncode != 0:
        raise RuntimeError(
            f'Unable to install required Python package {package_name}. '
            f'Please run: python3 -m pip install --user {package_name}'
        )

    retry_result = subprocess.run(pip_command, check=False)
    if retry_result.returncode != 0 or importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            f'Unable to install required Python package {package_name}. '
            f'Please run: python3 -m pip install --user {package_name}'
        )


def auth_cli_path() -> Path:
    return ensure_embedded_backend_file('auth.py')


def _build_standalone_source() -> str:
    current_source = script_path().read_text(encoding='utf-8')
    if EMBEDDED_BACKEND_MODULES:
        return current_source

    embedded_modules = {}
    for filename in BACKEND_MODULE_FILES:
        embedded_modules[filename] = (script_dir() / filename).read_text(encoding='utf-8')

    replacement = 'EMBEDDED_BACKEND_MODULES = {\n'
    for filename, module_source in embedded_modules.items():
        replacement += f"    {filename!r}: {module_source!r},\n"
    replacement += '}\n'
    marker = 'EMBEDDED_BACKEND_MODULES = {}\n'
    if marker not in current_source:
        raise RuntimeError('Expected EMBEDDED_BACKEND_MODULES marker in clawssh.py')
    return current_source.replace(marker, replacement, 1)


def run_auth_subprocess(*args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    command = [
        'python3',
        str(auth_cli_path()),
        '--state-dir',
        str(state_dir()),
        *args,
    ]
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def detect_tailscale_cert_domain() -> str:
    if shutil.which('tailscale') is None:
        return ''
    command = (
        "import json, sys\n"
        "try:\n"
        "    data = json.load(sys.stdin)\n"
        "except Exception:\n"
        "    raise SystemExit(1)\n"
        "domains = data.get('CertDomains') or []\n"
        "if domains:\n"
        "    print(str(domains[0]).rstrip('.'))\n"
        "    raise SystemExit(0)\n"
        "self_obj = data.get('Self') or {}\n"
        "dns_name = (self_obj.get('DNSName') or '').rstrip('.')\n"
        "if dns_name:\n"
        "    print(dns_name)\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    result = subprocess.run(
        ['tailscale', 'status', '--json'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ''
    parsed = subprocess.run(
        [sys.executable, '-c', command],
        input=result.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    if parsed.returncode != 0:
        return ''
    return parsed.stdout.strip()


def detect_bind_host() -> str:
    if os.getenv('CLAWSSH_HOST', '').strip():
        return os.environ['CLAWSSH_HOST'].strip()
    if os.getenv('CLAWSSH_TAILSCALE_IP', '').strip():
        return os.environ['CLAWSSH_TAILSCALE_IP'].strip()
    if shutil.which('tailscale') is not None:
        result = subprocess.run(
            ['tailscale', 'ip', '-4'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line:
                    return line
    if shutil.which('ip') is not None:
        result = subprocess.run(
            ['ip', '-o', '-4', 'addr', 'show', 'dev', 'tailscale0'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"r "**********"e "**********"s "**********"u "**********"l "**********"t "**********". "**********"s "**********"t "**********"d "**********"o "**********"u "**********"t "**********". "**********"s "**********"p "**********"l "**********"i "**********"t "**********"( "**********") "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********". "**********"c "**********"o "**********"u "**********"n "**********"t "**********"( "**********"' "**********". "**********"' "**********") "**********"  "**********"= "**********"= "**********"  "**********"3 "**********"  "**********"a "**********"n "**********"d "**********"  "**********"' "**********"/ "**********"' "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
                    return token.split('/', 1)[0]
    return ''


 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"( "**********") "**********"  "**********"- "**********"> "**********"  "**********"s "**********"t "**********"r "**********": "**********"
    return secrets.token_urlsafe(16)


def print_invite_qr(invite_link: str, output_png: Path | None = None) -> None:
    if not invite_link:
        return
    require_qrencode()
    terminal_result = subprocess.run(
        ['qrencode', '-t', 'UTF8', '-m', '1', '-l', 'L', invite_link],
        check=False,
        capture_output=True,
        text=True,
    )
    if terminal_result.returncode == 0 and terminal_result.stdout.strip():
        print('Enrollment QR code:')
        print(terminal_result.stdout.rstrip())
        print()
    else:
        print('Note: Terminal QR code generation failed.')
    if output_png:
        result = subprocess.run(
            ['qrencode', '-t', 'PNG', '-o', str(output_png), '-s', '6', '-m', '2', '-l', 'L', invite_link],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0 and output_png.exists():
            os.chmod(output_png, 0o600)
            print(f'Generated enrollment QR code: {output_png}')
            if sys.stdin.isatty():
                answer = input('Open QR code image now? [y/N]: ').strip().lower()
                if answer in {'y', 'yes'}:
                    opener = 'open' if os.uname().sysname == 'Darwin' else 'xdg-open'
                    if shutil.which(opener):
                        subprocess.run([opener, str(output_png)], check=False)
                    else:
                        print(f'Note: Could not find {opener} to open the image automatically.')
        elif result.returncode == 0:
            print('Note: qrencode did not create the requested PNG output file.')
        else:
            print('Note: PNG QR code generation failed (check if libpng is available for qrencode).')
    print('If the QR code is hard to scan, use the enrollment link shown above instead.')
    print()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('', 0))
        return int(sock.getsockname()[1])


def ensure_state_dir_initialized() -> None:
    state_path = state_dir()
    device_path = device_dir()
    state_path.mkdir(parents=True, exist_ok=True)
    device_path.mkdir(parents=True, exist_ok=True)
    os.chmod(state_path, 0o700)
    os.chmod(device_path, 0o700)
    run_auth_subprocess('ensure-state')
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"l "**********"e "**********"g "**********"a "**********"c "**********"y "**********"_ "**********"n "**********"a "**********"m "**********"e "**********"  "**********"i "**********"n "**********"  "**********"( "**********"' "**********"b "**********"a "**********"c "**********"k "**********"e "**********"n "**********"d "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"' "**********", "**********"  "**********"' "**********"b "**********"a "**********"c "**********"k "**********"e "**********"n "**********"d "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********". "**********"p "**********"b "**********"k "**********"d "**********"f "**********"2 "**********"' "**********") "**********": "**********"
        legacy_path = state_path / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()


def ensure_ca() -> None:
    cert_path = ca_cert_file()
    key_path = ca_key_file()
    if cert_path.is_file() and key_path.is_file():
        os.chmod(cert_path, 0o600)
        os.chmod(key_path, 0o600)
        return

    require_system_command(
        'openssl',
        package_name='openssl',
        context='manage ClawSSH certificates',
    )
    ensure_state_dir_initialized()
    # Using EC prime256v1 (P-256) for a much smaller certificate footprint in QR codes
    subprocess.run(
        [
            'openssl',
            'ecparam',
            '-name',
            'prime256v1',
            '-genkey',
            '-noout',
            '-out',
            str(key_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.chmod(key_path, 0o600)
    subprocess.run(
        [
            'openssl',
            'req',
            '-x509',
            '-new',
            '-nodes',
            '-key',
            str(key_path),
            '-sha256',
            '-days',
            '3650',
            '-out',
            str(cert_path),
            '-subj',
            '/CN=ClawSSH Local CA',
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.chmod(cert_path, 0o600)
    print('Generated local ClawSSH CA:')
    print(f'  {cert_path}')


def _server_extfile_contents(bind_host: str, connect_host: str) -> str:
    lines = [
        'basicConstraints=CA:FALSE',
        'keyUsage=digitalSignature,keyEncipherment',
        'extendedKeyUsage=serverAuth',
        'subjectKeyIdentifier=hash',
        'authorityKeyIdentifier=keyid,issuer',
        'subjectAltName=@alt_names',
        '[alt_names]',
    ]
    alt_index = 1
    for host in (connect_host, bind_host):
        if not host:
            continue
        if host == bind_host and host == connect_host:
            continue
        prefix = 'DNS' if any(char for char in host if not (char.isdigit() or char == '.')) else 'IP'
        lines.append(f'{prefix}.{alt_index}={host}')
        alt_index += 1
    if alt_index == 1 and bind_host:
        prefix = 'DNS' if any(char for char in bind_host if not (char.isdigit() or char == '.')) else 'IP'
        lines.append(f'{prefix}.1={bind_host}')
    return '\n'.join(lines) + '\n'


def generate_server_cert(bind_host: str, connect_host: str, force_refresh: bool = False) -> None:
    cert_path = server_cert_file()
    key_path = server_key_file()
    if not force_refresh and cert_path.is_file() and key_path.is_file():
        os.chmod(cert_path, 0o600)
        os.chmod(key_path, 0o600)
        return

    require_system_command(
        'openssl',
        package_name='openssl',
        context='generate ClawSSH server certificates',
    )
    ensure_ca()
    work_dir = Path(tempfile.mkdtemp(prefix='server-cert.', dir=state_dir()))
    csr_file = work_dir / 'server.csr'
    ext_file = work_dir / 'server.ext'
    try:
        subprocess.run(['openssl', 'genrsa', '-out', str(key_path), '2048'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chmod(key_path, 0o600)
        subprocess.run(
            [
                'openssl',
                'req',
                '-new',
                '-key',
                str(key_path),
                '-out',
                str(csr_file),
                '-subj',
                f'/CN={connect_host or bind_host}',
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ext_file.write_text(_server_extfile_contents(bind_host, connect_host), encoding='utf-8')
        subprocess.run(
            [
                'openssl',
                'x509',
                '-req',
                '-in',
                str(csr_file),
                '-CA',
                str(ca_cert_file()),
                '-CAkey',
                str(ca_key_file()),
                '-CAcreateserial',
                '-out',
                str(cert_path),
                '-days',
                '825',
                '-sha256',
                '-extfile',
                str(ext_file),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.chmod(cert_path, 0o600)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    print('Refreshed ClawSSH server certificate:')
    print(f'  {cert_path}')


def print_connection_info(bind_host: str, connect_host: str) -> None:
    host = connect_host or bind_host
    if not host:
        return
    print('=== ClawSSH Manual Connect ===')
    print(f'Connect URL: wss://{host}:{current_port()}')
    print(f'CA certificate: {ca_cert_file()}')
    print('Client auth: mutual TLS only')
    print('Legacy URL passwords are ignored.')
    print('==============================')


def prompt_port() -> int:
    while True:
        answer = input(f'Enter backend port [current: {current_port()}, default: 8765]: ').strip()
        if not answer:
            return current_port()
        if not answer.isdigit():
            print('Port must be a number (1-65535).', file=sys.stderr)
            continue
        port = int(answer)
        if 1 <= port <= 65535:
            return port
        print('Port must be between 1 and 65535.', file=sys.stderr)


def detect_package_install_command(package_name: str) -> str:
    system_name = os.uname().sysname
    if system_name == 'Darwin' and shutil.which('brew'):
        return f'brew install {package_name}'
    if system_name == 'Linux':
        for command_name, command_value in (
            ('apt-get', f'sudo apt-get update && sudo apt-get install -y {package_name}'),
            ('dnf', f'sudo dnf install -y {package_name}'),
            ('yum', f'sudo yum install -y {package_name}'),
            ('pacman', f'sudo pacman -Sy --noconfirm {package_name}'),
            ('zypper', f'sudo zypper --non-interactive install {package_name}'),
            ('apk', f'sudo apk add {package_name}'),
        ):
            if shutil.which(command_name):
                return command_value
    return ''


def require_system_command(command_name: str, *, package_name: str | None = None, context: str) -> None:
    if shutil.which(command_name) is not None:
        return
    install_label = package_name or command_name
    print(f'{command_name} is required to {context}.')
    install_cmd = detect_package_install_command(install_label)
    if not install_cmd:
        raise RuntimeError(
            f'{command_name} is not installed and no automatic installer is available for OS: {os.uname().sysname}'
        )
    if not sys.stdin.isatty():
        raise RuntimeError(
            f'{command_name} is not installed and this session is non-interactive.\n'
            f'Install it manually, then retry:\n  {install_cmd}'
        )
    print(f'{command_name} was not found.')
    print('Install command:')
    print(f'  {install_cmd}')
    answer = input(f'Install {command_name} automatically now? [Y/n]: ').strip()
    if answer.lower() in {'n', 'no'}:
        raise RuntimeError(f'{command_name} installation skipped.')
    result = subprocess.run(['/bin/bash', '-lc', install_cmd], check=False)
    if result.returncode == 0 and shutil.which(command_name) is not None:
        print(f'{command_name} installed successfully.')
        return
    raise RuntimeError(
        f'{command_name} installation did not complete successfully.\n'
        f'Please install {command_name} manually, then retry.\n  {install_cmd}'
    )


def require_tmux(context: str = 'run ClawSSH backend') -> None:
    require_system_command('tmux', package_name='tmux', context=context)


def require_qrencode(context: str = 'print enrollment QR codes') -> None:
    require_system_command('qrencode', package_name='qrencode', context=context)


def install_backend_runtime() -> None:
    runtime_script = state_dir() / 'clawssh.py'

    state_dir().mkdir(parents=True, exist_ok=True)
    runtime_script.write_text(_build_standalone_source(), encoding='utf-8')
    os.chmod(runtime_script, 0o700)


def install_autostart(bind_host: str, connect_host: str) -> None:
    system_name = os.uname().sysname
    if system_name == 'Darwin':
        install_autostart_launchd()
        return
    if system_name == 'Linux':
        install_autostart_systemd_user()
        return
    raise RuntimeError(f'Auto-start setup is not implemented for OS: {system_name}')


def install_autostart_systemd_user() -> None:
    unit_dir = Path(os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))).expanduser() / 'systemd' / 'user'
    unit_file = unit_dir / 'clawssh.service'
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_file.write_text(
        '\n'.join(
            [
                '[Unit]',
                'Description=ClawSSH Backend Server',
                'After=network-online.target',
                'Wants=network-online.target',
                '',
                '[Service]',
                'Type=simple',
                f'WorkingDirectory={state_dir()}',
                f'ExecStart={state_dir() / "clawssh.py"} --server',
                'Restart=always',
                'RestartSec=3',
                'Environment=PATH=/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
                '',
                '[Install]',
                'WantedBy=default.target',
                '',
            ]
        ),
        encoding='utf-8',
    )
    if shutil.which('systemctl') is None:
        print(f'Installed systemd user unit: {unit_file}')
        print('systemctl not found; enable it manually when available:')
        print('  systemctl --user daemon-reload')
        print('  systemctl --user enable --now clawssh.service')
        return
    daemon_reload = subprocess.run(['systemctl', '--user', 'daemon-reload'], check=False)
    if daemon_reload.returncode != 0:
        print(f'Installed systemd user unit: {unit_file}')
        print("Could not run 'systemctl --user'. Enable manually:")
        print('  systemctl --user daemon-reload')
        print('  systemctl --user enable --now clawssh.service')
        return
    enable_now = subprocess.run(['systemctl', '--user', 'enable', '--now', 'clawssh.service'], check=False)
    if enable_now.returncode == 0:
        print('Auto-start enabled via systemd user service: clawssh.service')
        return
    print(f'Installed systemd user unit: {unit_file}')
    print('Enable/start manually:')
    print('  systemctl --user enable --now clawssh.service')


def install_autostart_launchd() -> None:
    agents_dir = Path.home() / 'Library' / 'LaunchAgents'
    plist_file = agents_dir / 'com.clawssh.server.plist'
    label = 'com.clawssh.server'
    log_dir = state_dir() / 'logs'
    agents_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plist_file.write_text(
        '\n'.join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
                '<plist version="1.0">',
                '<dict>',
                '  <key>Label</key>',
                f'  <string>{label}</string>',
                '  <key>ProgramArguments</key>',
                '  <array>',
                f'    <string>{state_dir() / "clawssh.py"}</string>',
                '    <string>--server</string>',
                '  </array>',
                '  <key>WorkingDirectory</key>',
                f'  <string>{state_dir()}</string>',
                '  <key>RunAtLoad</key>',
                '  <true/>',
                '  <key>KeepAlive</key>',
                '  <true/>',
                '  <key>StandardOutPath</key>',
                f'  <string>{log_dir / "launchd.out.log"}</string>',
                '  <key>StandardErrorPath</key>',
                f'  <string>{log_dir / "launchd.err.log"}</string>',
                '  <key>EnvironmentVariables</key>',
                '  <dict>',
                '    <key>PATH</key>',
                '    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>',
                '  </dict>',
                '</dict>',
                '</plist>',
                '',
            ]
        ),
        encoding='utf-8',
    )
    if shutil.which('launchctl') is None:
        print(f'Installed LaunchAgent: {plist_file}')
        print('launchctl not found; load manually in a macOS user session:')
        print(f'  launchctl bootstrap gui/{os.getuid()} "{plist_file}"')
        return
    subprocess.run(['launchctl', 'bootout', f'gui/{os.getuid()}/{label}'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bootstrap = subprocess.run(['launchctl', 'bootstrap', f'gui/{os.getuid()}', str(plist_file)], check=False)
    if bootstrap.returncode == 0:
        subprocess.run(['launchctl', 'enable', f'gui/{os.getuid()}/{label}'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['launchctl', 'kickstart', '-k', f'gui/{os.getuid()}/{label}'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f'Auto-start enabled via launchd LaunchAgent: {label}')
        print(f'Logs: {log_dir / "launchd.out.log"} and {log_dir / "launchd.err.log"}')
        return
    print(f'Installed LaunchAgent: {plist_file}')
    print('Load manually in a macOS user session:')
    print(f'  launchctl bootstrap gui/{os.getuid()} "{plist_file}"')


def prompt_install_finish_action() -> tuple[bool, bool]:
    print()
    print('Choose how to run ClawSSH:')
    print('1) Run directly now')
    print('2) Install auto-start on this machine')
    print('3) Finish without starting')
    choice = input('Select an option [1-3]: ').strip()
    if choice in {'', '1'}:
        return True, False
    if choice == '2':
        return False, True
    if choice in {'3', 'n', 'N', 'no', 'NO', 'later', 'LATER'}:
        return False, False
    raise RuntimeError(f'Invalid selection: {choice}')


def prompt_enroll_after_install() -> str:
    if not sys.stdin.isatty():
        return ''
    print()
    answer = input('Enroll a device now? [y/N]: ').strip()
    if answer.lower() not in {'y', 'yes'}:
        return ''
    return prompt_device_id()


def prompt_device_id() -> str:
    while True:
        answer = ''.join(char for char in input('Enter device ID: ') if char.isalnum() or char in '._-').strip()
        if answer:
            return answer
        print('Device ID cannot be empty.', file=sys.stderr)


def list_devices() -> None:
    run_auth_subprocess('list-devices')


def revoke_device(device_id: str) -> None:
    if not device_id:
        raise RuntimeError('revoke-device requires a device ID.')
    run_auth_subprocess('revoke-device', '--device-id', device_id)
    print(f'Revoked device: {device_id}')


def enroll_device(device_id: str, bind_host: str, connect_host: str) -> None:
    invitation_code = "**********"
    enrollment_port = find_free_port()
    ensure_ca()
    generate_server_cert(bind_host, connect_host, force_refresh=False)
    ws_url = f'wss://{connect_host or bind_host}:{current_port()}'
    enroll_url = f'https://{connect_host or bind_host}:{enrollment_port}/enroll'

    ca_fingerprint = certificate_fingerprint_from_pem_file(ca_cert_file())

    # Use 1-letter keys to minimize JSON overhead
    invite_payload = {
        't': 'c-e',  # type: clawssh-enroll
        'w': ws_url, # wsUrl
        'e': enroll_url, # enrollmentUrl
        'c': invitation_code, # invitationCode
        'f': ca_fingerprint, # caFingerprint
    }
    if device_id:
        invite_payload['l'] = device_id # deviceLabel

    invite_json = json.dumps(invite_payload, separators=(',', ':'))
    invite_link = 'clawssh://enroll?p=' + base64.urlsafe_b64encode(invite_json.encode('utf-8')).decode('ascii').rstrip('=')

    print('Enrollment invite (compact):')
    print(invite_json[:100] + '...')
    print('Enrollment invite link:')
    print(invite_link[:100] + '...')
    print()

    invite_qr_path = state_dir() / 'enroll_invite.png'
    try:
        print_invite_qr(invite_json, output_png=invite_qr_path)

        print('Open the enrollment link on your device, or scan the QR code in the ClawSSH app.')
        print(f'Waiting for your device to connect to {enroll_url} ...')
        result = run_auth_subprocess(
            'serve-enrollment',
            '--bind-host',
            bind_host,
            '--port',
            str(enrollment_port),
            '--invitation-code',
            invitation_code,
            '--ws-url',
            ws_url,
            '--suggested-device-id',
            device_id,
            '--timeout-seconds',
            '300',
            capture_output=True,
        )
        if result.stdout:
            print(result.stdout.strip())
    finally:
        if invite_qr_path.exists():
            invite_qr_path.unlink()
            print('Cleaned up enrollment QR code image.')


def show_setup_menu() -> tuple[str, str, bool, bool]:
    print('=== ClawSSH Setup Menu ===')
    print('1) Fresh install (initialize CA, server cert, and runtime)')
    print('2) Enroll device')
    print('3) List devices')
    print('4) Revoke device')
    print(f'5) Rotate server certificate')
    print(f'6) Define backend port (current: {current_port()})')
    print('7) Set up auto-start on this machine')
    print('8) Quit')
    choice = input('Select an option [1-8]: ').strip()
    if choice in {'', '1'}:
        return 'fresh-install', '', True, False
    if choice == '2':
        return 'enroll-device', '', False, False
    if choice == '3':
        return 'list-devices', '', False, False
    if choice == '4':
        return 'revoke-device', '', False, False
    if choice == '5':
        return 'rotate-server-cert', '', False, False
    if choice == '6':
        return 'set-port', '', True, False
    if choice == '7':
        return 'install-autostart', '', False, True
    if choice in {'8', 'q', 'Q', 'quit', 'QUIT'}:
        print('Cancelled.')
        raise SystemExit(0)
    raise RuntimeError(f'Invalid selection: {choice}')


def export_runtime_env(bind_host: str) -> None:
    os.environ['CLAWSSH_HOST'] = bind_host
    os.environ['CLAWSSH_STATE_DIR'] = str(state_dir())
    os.environ['CLAWSSH_TLS_CA_CERT_FILE'] = str(ca_cert_file())
    os.environ['CLAWSSH_TLS_CERT_FILE'] = str(server_cert_file())
    os.environ['CLAWSSH_TLS_KEY_FILE'] = str(server_key_file())


def save_port(port: int) -> None:
    os.environ['CLAWSSH_PORT'] = str(port)
    port_path = port_file()
    port_path.write_text(f'{port}\n', encoding='utf-8')
    os.chmod(port_path, 0o600)
    print(f'Saved ClawSSH backend port: {port_path}')


def run_server() -> int:
    ensure_python_package('websockets')
    server = import_backend_module('server')
    return asyncio_run(server.main())


def launch_server_runtime() -> int:
    runtime_script = state_dir() / 'clawssh.py'
    completed = subprocess.run(['python3', str(runtime_script), '--server-runtime'], check=False)
    return int(completed.returncode)


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--server-runtime', action='store_true')
    parser.add_argument('--help', '-h', action='store_true')
    parser.add_argument('command', nargs='?')
    parser.add_argument('command_arg', nargs='?')
    args, extras = parser.parse_known_args(argv)
    if extras:
        raise RuntimeError(f'Unknown option or command: {extras[0]}')
    if args.command not in {None, 'enroll-device', 'list-devices', 'revoke-device', 'rotate-server-cert'}:
        raise RuntimeError(f'Unknown option or command: {args.command}')
    return args


def print_help() -> None:
    print('Usage: clawssh.py [options|command]')
    print()
    print('Options:')
    print('  (no args)                 Show interactive setup menu')
    print('  --server                  Start backend server')
    print('  --help, -h                Show this help')
    print()
    print('Commands:')
    print('  enroll-device [device-id] Generate a client certificate and add it to the authorized device list')
    print('  list-devices              Print the enrolled device table')
    print('  revoke-device <device-id> Revoke a device immediately')
    print('  rotate-server-cert        Refresh the server certificate while keeping the CA intact')


def main(argv: list[str] | None = None) -> int:
    _set_default_port_from_state()
    args = parse_args(argv or sys.argv[1:])
    if args.help:
        print_help()
        return 0

    command = args.command or ''
    command_arg = args.command_arg or ''
    start_backend = bool(args.server)
    server_runtime = bool(args.server_runtime)
    do_install_autostart = False
    autostart_configured = False
    prompt_set_port = False
    enroll_after_install_device_id = ''

    bind_host = detect_bind_host()
    connect_host = detect_tailscale_cert_domain() or bind_host
    if not bind_host:
        raise RuntimeError('Unable to determine backend bind host automatically.\nSet CLAWSSH_HOST or CLAWSSH_TAILSCALE_IP explicitly.')
    export_runtime_env(bind_host)

    if server_runtime:
        return run_server()

    if not command and not start_backend:
        if not sys.stdin.isatty():
            raise RuntimeError('No TTY available for setup menu.\nUse one of: --server, enroll-device, list-devices, revoke-device, rotate-server-cert')
        command, command_arg, prompt_set_port, do_install_autostart = show_setup_menu()

    ensure_state_dir_initialized()

    if prompt_set_port:
        save_port(prompt_port())

    if command == 'fresh-install':
        ensure_ca()
        generate_server_cert(bind_host, connect_host, force_refresh=True)
        install_backend_runtime()
        print_connection_info(bind_host, connect_host)
        start_backend, do_install_autostart = prompt_install_finish_action()
        enroll_after_install_device_id = prompt_enroll_after_install()
    elif command == 'list-devices':
        list_devices()
        return 0
    elif command == 'revoke-device':
        if not command_arg and sys.stdin.isatty():
            command_arg = prompt_device_id()
        revoke_device(command_arg)
        return 0
    elif command == 'enroll-device':
        enroll_device(command_arg, bind_host, connect_host)
        return 0
    elif command == 'rotate-server-cert':
        generate_server_cert(bind_host, connect_host, force_refresh=True)
        return 0
    elif command == 'set-port':
        print_connection_info(bind_host, connect_host)
        return 0
    elif command == 'install-autostart':
        ensure_ca()
        generate_server_cert(bind_host, connect_host, force_refresh=False)
        install_backend_runtime()
        enroll_after_install_device_id = prompt_enroll_after_install()

    if do_install_autostart:
        install_autostart(bind_host, connect_host)
        autostart_configured = True

    if enroll_after_install_device_id:
        enroll_device(enroll_after_install_device_id, bind_host, connect_host)

    if not start_backend:
        if not autostart_configured and command not in {'fresh-install', 'install-autostart'}:
            print_connection_info(bind_host, connect_host)
        if not autostart_configured:
            print('Backend not started.')
            print('To run it directly later:')
            print(f'  {state_dir() / "clawssh.py"} --server')
        return 0

    require_tmux('run ClawSSH backend')
    ensure_ca()
    generate_server_cert(bind_host, connect_host, force_refresh=False)
    install_backend_runtime()
    print_connection_info(bind_host, connect_host)
    return launch_server_runtime()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr if 'Unknown option or command' in str(exc) else sys.stdout)
        raise SystemExit(1)
