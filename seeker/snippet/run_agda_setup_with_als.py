#date: 2025-07-29T16:54:37Z
#url: https://api.github.com/gists/58ba3752125f4956928fea83252f0027
#owner: https://api.github.com/users/andy0130tw

#!/usr/bin/env python

import os
from pathlib import Path
import json
import subprocess
from typing import IO, Dict
import re

dirname = Path(__file__).resolve().parent

def rpcenc(content):
    content['jsonrpc'] = '2.0'
    payload = json.dumps(content)
    sz = len(payload)
    return f'Content-Length: {sz}\r\n\r\n{payload}'

def reqdict(idx, method, params):
    return {
        'id': idx,
        'method': method,
        'params': params,
    }

def notidict(method, params):
    return {
        'method': method,
        'params': params,
    }

def respdict(idx, params = None):
    return {
        'id': idx,
        'result': params,
    }

def iotcmreqs(iotcms, n):
    def _makereq(pair):
        idx, cmd = pair
        return reqdict(n + idx, 'agda', { "tag": "CmdReq", "contents": cmd })
    return map(_makereq, enumerate(iotcms))

def send(f: IO[bytes], msg: Dict):
    f.write(rpcenc(msg).encode())
    f.flush()

#############################################################################

iotcms = []
with open('file-list', 'r') as f:
    for ent in f:
        ent = ent.strip()
        if not ent or ent[0] == '#': continue
        entrepr = json.dumps(str((dirname / ent).resolve()))
        cmdpre = f'IOTCM {entrepr} None Indirect'
        iotcms.append(f'{cmdpre} (Cmd_load {entrepr} [])')
        iotcms.append(f'{cmdpre} (Cmd_no_metas)')

tosend = [
    reqdict(0, 'initialize', {
        'processId': None,
        'rootUri': '',
        'capabilities': {},
        'trace': 'verbose',
    }),
    notidict('initialized', {}),
    *iotcmreqs(iotcms, 1),
]

res = subprocess.Popen(
    ['node', '/home/qbane/agda-project/wasm-run/wasm-run.mjs', '/home/qbane/Downloads/als-local-opt.wasm'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    env={
        'Agda_datadir': str((dirname / '..' / '..').resolve()),
        'PATH': os.environ.get('PATH', ''),
    })

stdin: IO[bytes] | None = res.stdin
assert stdin is not None
stdout: IO[bytes] | None = res.stdout
assert stdout is not None


for idx, outmsg in enumerate(tosend):
    print(f'---------------------- Sending msg {idx + 1}/{len(tosend)} ---------------------- ')
    send(stdin, outmsg)

    if 'id' not in outmsg:
        continue

    ack = False

    buf = []
    while True:
        data = stdout.readline()
        nbytes = -1
        if mat := re.match(br'Content-Length:\s+(\d+)', data, re.IGNORECASE):
            nbytes = int(mat.group(1))
        nl = stdout.read(2)
        assert nl == b'\r\n'
        raw = stdout.read(nbytes)
        inmsg = json.loads(raw)

        if 'result' in inmsg:
            print('--> RESP', inmsg['id'], inmsg)
            if inmsg['id'] == outmsg['id']:
                ack = True
                if inmsg['result'].get('contents'):
                    print(inmsg['result'].get('contents'))
                    raise Exception('Encounter ERROR')

            if inmsg['id'] == 0:
                # init result
                break
        elif 'id' not in inmsg:
            print('--> NOTI', inmsg['method'], inmsg['params'])
        else:
            print('--> REQ', inmsg['id'], inmsg['params'])
            send(stdin, respdict(inmsg['id']))
            if inmsg['params'].get('tag') == 'ResponseEnd':
                break

    assert ack

    print('--- DONE ---')

print('All done!')