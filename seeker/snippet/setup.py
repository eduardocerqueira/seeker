#date: 2026-03-03T17:31:36Z
#url: https://api.github.com/gists/7e9b4fcbb93045c072b61a39f1e2a398
#owner: https://api.github.com/users/rcarmo

#!/usr/bin/env python3
"""
Provision a Cloudflare Tunnel and install a systemd service.

Usage:
  sudo python3 setup.py \
      --tunnel-name piclaw-vm \
      --hostname piclaw.example.com \
      --service-url http://localhost:3000 \
      --user agent
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import textwrap
from shutil import which

HOME = pathlib.Path.home()
CLOUDFLARE_DIR = HOME / ".cloudflared"
CERT_PATH = CLOUDFLARE_DIR / "cert.pem"


def run(cmd, *, capture=False, check=True):
    kwargs = {}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["text"] = True
    result = subprocess.run(cmd, check=check, **kwargs)
    return result.stdout.strip() if capture else None


def ensure_cloudflared_path():
    path = which("cloudflared")
    if not path:
        sys.exit("cloudflared binary not found in PATH.")
    return path


def ensure_certificate():
    if not CERT_PATH.exists():
        sys.exit(f"Cloudflare cert not found at {CERT_PATH} (run `cloudflared tunnel login`).")


def tunnel_exists(name):
    output = run(["cloudflared", "tunnel", "list", "--output", "json"], capture=True)
    tunnels = json.loads(output or "[]")
    return any(t.get("name") == name or t.get("id") == name for t in tunnels)


def get_tunnel_id(name):
    output = run(["cloudflared", "tunnel", "list", "--output", "json"], capture=True)
    tunnels = json.loads(output or "[]")
    for tunnel in tunnels:
        tunnel_name = tunnel.get("name")
        tunnel_id = tunnel.get("id")
        if tunnel_name == name or tunnel_id == name:
            if tunnel_id:
                return tunnel_id
            break
    sys.exit(
        f"Tunnel '{name}' not found. Create it first with `cloudflared tunnel create {name}` (same user as the cert)."
    )


def write_config(config_path, tunnel_name, tunnel_id, hostname, service_url):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    yaml = textwrap.dedent(
        f"""\
        tunnel: {tunnel_name}
        credentials-file: {CLOUDFLARE_DIR}/{tunnel_id}.json

        ingress:
          - hostname: {hostname}
            service: {service_url}
          - service: http_status:404
        """
    )
    config_path.write_text(yaml)
    print(f"Wrote {config_path}")


def write_systemd_unit(unit_path, cloudflared_path, config_path, user):
    unit_body = textwrap.dedent(
        f"""\
        [Unit]
        Description=Cloudflare Tunnel ({config_path.name})
        After=network-online.target
        Wants=network-online.target

        [Service]
        User={user}
        Group={user}
        ExecStart={cloudflared_path} --config {config_path} --no-autoupdate tunnel run
        Restart=on-failure
        RestartSec=5s

        [Install]
        WantedBy=multi-user.target
        """
    )
    unit_path.write_text(unit_body)
    print(f"Wrote {unit_path}")
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "enable", unit_path.name])
    run(["systemctl", "restart", unit_path.name])
    print(f"Enabled and started {unit_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Set up Cloudflare Tunnel as a systemd service.")
    parser.add_argument("--tunnel-name", required=True)
    parser.add_argument("--hostname", required=True, help="Public hostname (in your Cloudflare zone)")
    parser.add_argument("--service-url", default="http://localhost:3000", help="Local service URL")
    parser.add_argument("--config-path", default=str(CLOUDFLARE_DIR / "config.yml"))
    parser.add_argument("--user", default="agent", help="System user to run cloudflared as")
    args = parser.parse_args()

    ensure_certificate()
    cloudflared_path = ensure_cloudflared_path()

    if not tunnel_exists(args.tunnel_name):
        print(f"Creating tunnel {args.tunnel_name}…")
        run(["cloudflared", "tunnel", "create", args.tunnel_name])

    tunnel_id = get_tunnel_id(args.tunnel_name)
    credentials = CLOUDFLARE_DIR / f"{tunnel_id}.json"
    if not credentials.exists():
        sys.exit(f"Credentials file {credentials} not found (create command should produce it).")

    config_path = pathlib.Path(args.config_path).expanduser()
    write_config(config_path, args.tunnel_name, tunnel_id, args.hostname, args.service_url)

    print("Registering DNS route…")
    run(["cloudflared", "tunnel", "route", "dns", args.tunnel_name, args.hostname])

    unit_name = f"cloudflared-{args.tunnel_name}.service"
    unit_path = pathlib.Path("/etc/systemd/system") / unit_name
    write_systemd_unit(unit_path, cloudflared_path, config_path, args.user)

    print(f"Setup complete. Check status with `systemctl status {unit_name}`")


if __name__ == "__main__":
    main()