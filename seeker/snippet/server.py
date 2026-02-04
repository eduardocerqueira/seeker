#date: 2026-02-04T17:29:14Z
#url: https://api.github.com/gists/53db9f09bb8e4bf21d22cefc9a04cd52
#owner: https://api.github.com/users/sunapi386

#!/usr/bin/env python3
"""Flask server that wraps the Claude CLI as a localhost REST API."""

import argparse
import json
import os
import shutil
import subprocess
import sys

from flask import Flask, Response, jsonify, request, stream_with_context

app = Flask(__name__)

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", shutil.which("claude") or "claude")
DEFAULT_TIMEOUT = 300  # 5 minutes


def _build_cmd(prompt_opts):
    """Build the claude CLI command from request options."""
    cmd = [CLAUDE_BIN, "-p"]

    if prompt_opts.get("session_id"):
        cmd += ["--resume", prompt_opts["session_id"]]

    if prompt_opts.get("model"):
        cmd += ["--model", prompt_opts["model"]]

    if prompt_opts.get("system_prompt"):
        cmd += ["--system-prompt", prompt_opts["system_prompt"]]

    if prompt_opts.get("max_budget_usd") is not None:
        cmd += ["--max-budget-usd", str(prompt_opts["max_budget_usd"])]

    if prompt_opts.get("permission_mode"):
        cmd += ["--permission-mode", prompt_opts["permission_mode"]]

    if prompt_opts.get("allowed_tools"):
        for tool in prompt_opts["allowed_tools"]:
            cmd += ["--allowedTools", tool]

    return cmd


@app.route("/v1/health", methods=["GET"])
def health():
    """Health check — returns status and claude CLI version."""
    try:
        result = subprocess.run(
            [CLAUDE_BIN, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.strip()
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 503

    return jsonify({"status": "ok", "claude_version": version})


@app.route("/v1/chat", methods=["POST"])
def chat():
    """Non-streaming chat — runs claude -p --output-format json."""
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    cmd = _build_cmd(data) + ["--output-format", "json"]
    cwd = data.get("cwd")

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if result.returncode != 0:
        return jsonify({
            "error": "claude CLI failed",
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }), 502

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return jsonify({
            "error": "Failed to parse CLI output",
            "raw_output": result.stdout[:2000],
        }), 502

    return jsonify({
        "result": parsed.get("result", ""),
        "session_id": parsed.get("session_id", ""),
        "cost_usd": parsed.get("cost_usd"),
        "usage": parsed.get("usage", {}),
        "model": parsed.get("model", ""),
    })


@app.route("/v1/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming chat — runs claude with stream-json output, returns SSE."""
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    cmd = _build_cmd(data) + [
        "--output-format", "stream-json",
        "--verbose",
    ]
    cwd = data.get("cwd")

    def generate():
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
        proc.stdin.write(prompt)
        proc.stdin.close()

        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = event.get("type", "")

                if msg_type == "system":
                    sse = json.dumps({
                        "type": "start",
                        "session_id": event.get("session_id", ""),
                        "model": event.get("model", ""),
                    })
                    yield f"data: {sse}\n\n"

                elif msg_type == "assistant" and "content" in event:
                    for block in event.get("content", []):
                        if block.get("type") == "text":
                            sse = json.dumps({
                                "type": "delta",
                                "text": block["text"],
                            })
                            yield f"data: {sse}\n\n"

                elif msg_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        sse = json.dumps({
                            "type": "delta",
                            "text": delta["text"],
                        })
                        yield f"data: {sse}\n\n"

                elif msg_type == "result":
                    sse = json.dumps({
                        "type": "done",
                        "result": event.get("result", ""),
                        "cost_usd": event.get("cost_usd"),
                        "usage": event.get("usage", {}),
                        "session_id": event.get("session_id", ""),
                    })
                    yield f"data: {sse}\n\n"

        finally:
            proc.stdout.close()
            proc.stderr.close()
            proc.wait()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude CLI API Server")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"Starting Claude API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
