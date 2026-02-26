#date: 2026-02-26T17:40:57Z
#url: https://api.github.com/gists/1edab88ecc7953c4d3cb0f7624a70889
#owner: https://api.github.com/users/zeta987

#!/usr/bin/env python3
"""Single-file Linode-like speed test service for Zeabur on LN k3s."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
DISPLAY_LABEL = os.getenv("DISPLAY_LABEL", "Tokyo, JP")
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

<link rel="shortcut icon" href="https://assets.linode.com/icons/favicon.ico">
<title>LN k3s Ping Test</title>

<link rel="preload" href="https://assets.linode.com/fonts/source-sans-pro-v14-latin-600.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="https://assets.linode.com/fonts/source-sans-pro-v14-latin-regular.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="https://assets.linode.com/fonts/oswald-v35-latin-300.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="https://assets.linode.com/fonts/oswald-v35-latin-regular.woff2" as="font" type="font/woff2" crossorigin>

<style type="text/css">
@font-face {
  font-family: "Source Sans Pro";
  font-style: normal;
  font-weight: 400;
  src:
    local(""),
    url("https://assets.linode.com/fonts/source-sans-pro-v14-latin-regular.woff2") format("woff2"),
    url("https://assets.linode.com/fonts/source-sans-pro-v14-latin-regular.woff") format("woff");
}
@font-face {
  font-family: "Source Sans Pro";
  font-style: normal;
  font-weight: 600;
  src:
    local(""),
    url("https://assets.linode.com/fonts/source-sans-pro-v14-latin-600.woff2") format("woff2"),
    url("https://assets.linode.com/fonts/source-sans-pro-v14-latin-600.woff") format("woff");
}
@font-face {
  font-family: "Oswald";
  font-style: normal;
  font-weight: 300;
  src:
    local(""),
    url("https://assets.linode.com/fonts/oswald-v35-latin-300.woff2") format("woff2"),
    url("https://assets.linode.com/fonts/oswald-v35-latin-300.woff") format("woff");
}
@font-face {
  font-family: "Oswald";
  font-style: normal;
  font-weight: 400;
  src:
    local(""),
    url("https://assets.linode.com/fonts/oswald-v35-latin-regular.woff2") format("woff2"),
    url("https://assets.linode.com/fonts/oswald-v35-latin-regular.woff") format("woff");
}
:root {
  --dark-gray: #32363b;
  --grass-green: #02b159;
  --sky-blue: #0ac9f7;
  --gap: 2rem;
  --site-header-height: 76px;
}
body {
  -moz-osx-font-smoothing: grayscale;
  -webkit-font-smoothing: antialiased;
  background-color: #fafafc;
  box-sizing: border-box;
  color: var(--dark-gray);
  display: flex;
  flex-direction: column;
  font-family: "Source Sans Pro", sans-serif;
  font-weight: 400;
  letter-spacing: 0;
  line-height: 1.15;
  margin: 0;
  min-height: 100vh;
  min-height: 100dvh;
  padding: 0;
  text-align: center;
}
*, *::before, *::after {
  box-sizing: inherit;
}
header {
  align-items: center;
  background-color: white;
  border-bottom: 1px solid #ededf4;
  display: flex;
  height: calc(var(--site-header-height) - 1px);
  justify-content: space-between;
  padding-inline: 16px;
}
main {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: var(--gap);
  justify-content: center;
  padding-block: var(--gap);
  padding-inline: 16px;
}
footer {
  background-color: white;
  border-top: 1px solid #ededf4;
  display: flex;
  justify-content: center;
  padding: 16px;
}
#pageTitle {
  font-size: 4em;
  letter-spacing: -2px;
  margin: 0;
}
#pageTitle .eyebrow {
  display: block;
  font-family: "Oswald", sans-serif;
  font-size: 1rem;
  font-weight: 400;
  letter-spacing: 2px;
  line-height: 1.5;
  margin-bottom: 1em;
  text-transform: uppercase;
  width: 100%;
}
#startStopBtn {
  background-color: var(--grass-green);
  border-radius: 0.25rem;
  border: 0.125rem solid var(--grass-green);
  color: white;
  cursor: pointer;
  display: inline-block;
  font-weight: 600;
  margin-inline: auto;
  padding: 0.8888888889em 1.666rem;
  text-decoration: none;
  user-select: none;
}
#startStopBtn:hover,
#startStopBtn.running {
  background-color: var(--dark-gray);
  border-color: var(--dark-gray);
  color: white;
}
#startStopBtn::before {
  content: "Start Ping";
}
#startStopBtn.running::before {
  content: "Stop Ping";
}
.testGroup {
  align-items: flex-start;
  display: flex;
  justify-content: center;
  margin-inline: auto;
  max-width: 520px;
}
.testSlot {
  flex: 0 1 420px;
  width: min(100%, 420px);
}
.testCard {
  aspect-ratio: 1 / 1;
  display: grid;
  font-size: min(21.5992px, calc(100vw * 21.5992 / (16 + 520 + 16)));
  grid: "full" 1fr / 1fr;
  margin: 0;
}
.testCard > * {
  grid-area: full;
}
.testName {
  align-self: end;
  font-size: max(1em, 12px);
  font-weight: 600;
  z-index: 9;
}
.meterText {
  align-self: center;
  z-index: 9;
}
.meterValue {
  display: block;
  font-family: "Oswald", sans-serif;
  font-size: 4em;
  font-weight: 300;
  line-height: 1;
  margin-bottom: .1em;
}
.meterValue:empty::before {
  content: "0.00";
}
.meterUnit {
  display: block;
  font-family: "Oswald", sans-serif;
  font-size: max(1.5em, 12px);
  font-weight: 400;
  letter-spacing: 0.5px;
  line-height: 1;
}
.testCard canvas {
  height: 100%;
  width: 100%;
  z-index: 1;
}
#ipArea {
  margin-top: var(--gap);
}
#tip {
  color: #757b84;
  font-size: 0.95rem;
}
@media all and (max-width: 61.99em) {
  #pageTitle {
    font-size: 3rem;
  }
  #pageTitle .eyebrow {
    font-size: 0.875rem;
    margin-bottom: 0.333em;
  }
}
@media all and (max-width: 47.99em) {
  #pageTitle {
    font-size: 2rem;
  }
}
</style>

<script src="/dc_loc.js"></script>
<script type="text/javascript">
function I(id) { return document.getElementById(id); }

const meterBk = "#ebecf0";
const pingColor = "#02b159";
const progColor = "#0ac9f7";
const ABORT_ERR = "ABORT";

const cfg = {
  pingGapMs: 500
};

const state = {
  running: false,
  abort: false,
  ping: 0,
  jitter: 0,
  pingProgress: 0,
  sampleCount: 0,
  clientIp: ""
};

function setLocation() {
  if (typeof display_label !== "string" || !display_label) return;
  const locationText = " : " + display_label;
  I("pageTitle").appendChild(document.createTextNode(locationText));
  document.title += locationText;
}

function drawMeter(c, amount, bk, fg, progress, prog) {
  const ctx = c.getContext("2d");
  const dp = window.devicePixelRatio || 1;
  const cw = c.clientWidth * dp;
  const ch = c.clientHeight * dp;
  const sizScale = ch * 0.0055;
  if (c.width === cw && c.height === ch) {
    ctx.clearRect(0, 0, cw, ch);
  } else {
    c.width = cw;
    c.height = ch;
  }

  ctx.beginPath();
  ctx.strokeStyle = bk;
  ctx.lineCap = "round";
  ctx.lineWidth = 4 * sizScale;
  ctx.arc(c.width / 2, c.height / 2, c.height / 2 - ctx.lineWidth, -Math.PI * 1.2, Math.PI * 0.2);
  ctx.stroke();

  ctx.beginPath();
  ctx.strokeStyle = fg;
  ctx.lineCap = amount > 0 ? "round" : "butt";
  ctx.lineWidth = 4 * sizScale;
  ctx.arc(c.width / 2, c.height / 2, c.height / 2 - ctx.lineWidth, -Math.PI * 1.2, amount * Math.PI * 1.2 - Math.PI * 1.2);
  ctx.stroke();

  ctx.fillStyle = prog;
  ctx.fillRect(c.width * 0.3, c.height - 28 * sizScale, c.width * 0.4 * progress, 4 * sizScale);
}

function msToAmount(s) {
  return 1 - (1 / Math.pow(1.08, Math.sqrt(Math.max(s, 0))));
}

function oscillate() {
  return 1 + 0.02 * Math.sin(Date.now() / 100);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ensureNotAborted() {
  if (state.abort) throw new Error(ABORT_ERR);
}

async function getIp() {
  const res = await fetch("/getIP.php?isp=true&distance=km&r=" + Math.random(), { cache: "no-store" });
  if (!res.ok) throw new Error("getIP failed");
  const payload = await res.json();
  state.clientIp = payload.processedString || "";
}

async function runPingLoop() {
  let previousSample = 0;
  while (!state.abort) {
    const t0 = performance.now();
    const res = await fetch("/empty.php?r=" + Math.random(), { cache: "no-store" });
    if (!res.ok) throw new Error("ping endpoint failed");
    const sample = Math.max(performance.now() - t0, 1);

    if (state.sampleCount === 0) {
      state.ping = sample;
      state.jitter = 0;
    } else {
      const instJitter = Math.abs(sample - previousSample);
      state.ping = sample < state.ping ? sample : 0.8 * state.ping + 0.2 * sample;
      state.jitter = state.sampleCount === 1 ? instJitter : (state.jitter < instJitter ? 0.3 * state.jitter + 0.7 * instJitter : 0.8 * state.jitter + 0.2 * instJitter);
    }

    previousSample = sample;
    state.sampleCount += 1;
    state.pingProgress = (state.sampleCount % 20) / 20;
    await sleep(cfg.pingGapMs);
  }
}

function updateUI() {
  const pingDisplay = state.running && state.sampleCount === 0 ? "..." : (state.ping > 0 ? state.ping.toFixed(2) : "");
  I("pingText").textContent = pingDisplay;
  I("ip").textContent = state.clientIp;
  drawMeter(
    I("pingMeter"),
    msToAmount(state.ping * (state.running ? oscillate() : 1)),
    meterBk,
    pingColor,
    state.pingProgress,
    progColor
  );
}

function initUI(resetIp) {
  state.ping = 0;
  state.jitter = 0;
  state.pingProgress = 0;
  state.sampleCount = 0;
  if (resetIp) {
    state.clientIp = "";
  }
  updateUI();
}

async function runAll() {
  state.running = true;
  state.abort = false;
  I("startStopBtn").className = "running";
  initUI(false);
  try {
    if (!state.clientIp) {
      await getIp();
      ensureNotAborted();
    }
    await runPingLoop();
  } catch (error) {
    if (!error || error.message !== ABORT_ERR) {
      console.error(error);
      if (!I("pingText").textContent) I("pingText").textContent = "Fail";
    }
  } finally {
    state.running = false;
    state.abort = false;
    I("startStopBtn").className = "";
  }
}

function startStop() {
  if (state.running) {
    state.abort = true;
    return;
  }
  runAll();
}

function frame() {
  updateUI();
  window.requestAnimationFrame(frame);
}

window.requestAnimationFrame = window.requestAnimationFrame || window.webkitRequestAnimationFrame || window.mozRequestAnimationFrame || window.msRequestAnimationFrame || function (callback) { setTimeout(callback, 1000 / 60); };
window.addEventListener("load", function () {
  setLocation();
  initUI(true);
  frame();
});
</script>
</head>
<body>
  <header>
    <img width="113.84" height="50" alt="Akamai Logo" src="https://assets.linode.com/akamai-logo.svg">
    <span id="tip">Continuous HTTP latency probe</span>
  </header>
  <main>
    <h1 id="pageTitle">
      <span class="eyebrow">Linode style on Zeabur</span>
      Ping Test
    </h1>
    <div id="test">
      <div class="testGroup">
        <div class="testSlot">
          <div class="testCard">
            <div class="testName">Ping</div>
            <canvas id="pingMeter" class="meter"></canvas>
            <div class="meterText">
              <span class="meterValue" id="pingText"></span>
              <span class="meterUnit">ms</span>
            </div>
          </div>
        </div>
      </div>
      <div id="ipArea">
        IP Address: <span id="ip"></span>
      </div>
    </div>
    <div id="startStopBtn" onclick="startStop()"></div>
  </main>
  <footer>
    <a target="_blank" href="https://www.linode.com/global-infrastructure/">Explore global infrastructure</a>
  </footer>
</body>
</html>
"""


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class SpeedtestHandler(BaseHTTPRequestHandler):
    server_version = "LNK3SSpeedtest/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format_str: str, *args: object) -> None:
        print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format_str % args))

    def do_HEAD(self) -> None:
        self._route(head_only=True)

    def do_GET(self) -> None:
        self._route(head_only=False)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/empty.php":
            self._send_text(404, "not found\n", head_only=False)
            return
        self._drain_body()
        self._send_text(200, "ok", "text/plain; charset=utf-8", head_only=False)

    def _route(self, head_only: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/":
            self._send_text(200, HTML_PAGE, "text/html; charset=utf-8", head_only=head_only)
            return
        if path == "/healthz":
            self._send_text(200, "ok\n", "text/plain; charset=utf-8", head_only=head_only)
            return
        if path == "/dc_loc.js":
            content = "var display_label = %s;\n" % json.dumps(DISPLAY_LABEL)
            self._send_text(200, content, "application/javascript; charset=utf-8", head_only=head_only)
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/empty.php":
            self._send_text(200, "", "text/plain; charset=utf-8", head_only=head_only)
            return
        if path == "/getIP.php":
            self._handle_get_ip(query, head_only=head_only)
            return
        if path == "/telemetry/telemetry.php":
            self._send_text(200, "id local", "text/plain; charset=utf-8", head_only=head_only)
            return
        self._send_text(404, "not found\n", "text/plain; charset=utf-8", head_only=head_only)

    def _handle_get_ip(self, query: dict[str, list[str]], head_only: bool) -> None:
        ip_addr = self._client_ip()
        if query.get("isp", ["false"])[0].lower() == "true":
            payload = {
                "processedString": ip_addr,
                "rawIspInfo": {
                    "org": "",
                    "asn": "",
                    "country": "",
                    "distance": query.get("distance", ["km"])[0],
                },
            }
            self._send_json(200, payload, head_only=head_only)
            return
        self._send_text(200, ip_addr, "text/plain; charset=utf-8", head_only=head_only)

    def _send_json(self, status_code: int, payload: dict[str, object], *, head_only: bool) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_text(
        self,
        status_code: int,
        content: str,
        content_type: str = "text/plain; charset=utf-8",
        *,
        head_only: bool,
    ) -> None:
        body = content.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _drain_body(self) -> None:
        total = _safe_int(self.headers.get("Content-Length", "0"), 0)
        remaining = max(total, 0)
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def _client_ip(self) -> str:
        forwarded = self.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = self.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        cf_ip = self.headers.get("CF-Connecting-IP")
        if cf_ip:
            return cf_ip.strip()
        return self.client_address[0]


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), SpeedtestHandler)
    print(f"Serving Linode-like speedtest at http://{HOST}:{PORT}")
    print(f"Display label: {DISPLAY_LABEL}")
    server.serve_forever()


if __name__ == "__main__":
    main()
