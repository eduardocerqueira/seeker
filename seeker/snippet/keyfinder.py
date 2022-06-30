#date: 2022-06-30T17:08:11Z
#url: https://api.github.com/gists/0b57eb4098381533183198ce6fee3198
#owner: https://api.github.com/users/blackle

#!/usr/bin/env python3
import random
from math import floor, sin
def printen(strr):
	print(strr, end='')

drawers = [
["https://staging.cohostcdn.org/attachment/d305d6ee-247d-4a2c-8554-08d1769271ca/0007.png",
"https://staging.cohostcdn.org/attachment/41f6e39e-a678-47dc-93f7-750045bf7004/0009.png",
"https://staging.cohostcdn.org/attachment/25d5915d-4757-4526-b303-996be4e24d1e/0008.png"],
["https://staging.cohostcdn.org/attachment/ab3742a0-7f06-4162-a09d-cd5a7facca99/0006.png",
"https://staging.cohostcdn.org/attachment/b7ffa0c5-0854-403d-9fe0-184791305ca1/0004.png",
"https://staging.cohostcdn.org/attachment/723902da-f9e0-4ae9-9839-7cb3ceaba082/0005.png"],
["https://staging.cohostcdn.org/attachment/af1e639f-c4a6-42d0-93e7-d242abb848b4/0001.png",
"https://staging.cohostcdn.org/attachment/40054e93-b82a-4e47-8484-92697342b30b/0002.png",
"https://staging.cohostcdn.org/attachment/fb9a608c-0995-43e0-ab25-1dc014120e90/0003.png"]
]

slats = [
"https://staging.cohostcdn.org/attachment/44cbe87b-465b-4c77-9b59-ff9f9534da32/bottom.jpg",
"https://staging.cohostcdn.org/attachment/b3f2a828-d841-4a75-9d29-31a2beb1431b/middle.jpg",
"https://staging.cohostcdn.org/attachment/a2d0f009-df38-4c08-891e-2fa09ae953b4/top.jpg"
]

mover = '''<img src="https://staging.cohostcdn.org/attachment/07d006ca-906f-4499-a89a-4528009f39e3/mover.png" style="position:absolute;pointer-events:none;right:0;bottom:0;margin:0;user-select:none;">'''

def print_drawer(url, row, col):
	left = 33*col
	top = 10*(2-row)
	printen(f'''<div style="position: absolute; left: {left}%; top: {top}%; width: 33%; height: 10%; background-image: url('{url}'); background-repeat: no-repeat; background-position: 50% 100%; background-size: 100% auto; overflow: auto; resize: vertical; max-height: 55%; min-height: 10%;">{mover}''')
	printen('''<div style="position: absolute;width: 100%;padding-bottom: 170%;bottom: 0;pointer-events:none;">''')
	if "0005.png" in url:
		printen('''<details style="position: absolute; inset: 0;">''')
		printen('''<summary style="cursor:pointer;position: absolute;right: 13%;top: 32%;height: 16%;left: 58%;background: rgba(0,0,0,.01);font-size: 0;pointer-events:auto;"></summary>''')
		printen('''<div style="position: absolute; right: 16%;top: 32%;height: 16%; background-color:#111;color:white;font-weight:bold;font-size:min(3vw,60%);padding:0em 2em;border-radius:15px;display:flex;align-items: center;z-index:1;">good job!</div>''')

		printen('''</details>''')


	lastwidth = 0
	for i in range(4):
		width = floor(sin(random.random()*100)*20+60)
		if abs(width-lastwidth)<10:
			width = floor(sin(random.random()*100)*20+60)
		printen(f'''<div style="position: absolute; left: 5%; top: {i*18+10}%; height: 18%; background-image: url('https://staging.cohostcdn.org/attachment/5ed9b6da-9f0c-4e34-ad5a-fa371307129e/ducky.png'); background-repeat: no-repeat; background-position: 100% 50%; background-size: auto 100%; overflow: auto; resize: horizontal; max-width: 80%; min-width: 40%; width:{width}%;pointer-events:auto;user-select:none;">{mover}</div>''')
		lastwidth = width


	printen('''</div>''')
	printen('''</div>''')


def print_slat(url, row):
	top = max(0,(2-row)*10-1)
	printen(f'''<div style="position: absolute; left: 0; top:{top}%; width: 100%; height: 2%; background-image: url('{url}');background-size:100% auto; background-position:50% 0%;pointer-events:none;background-repeat:no-repeat;"></div>''')

printen('''<div style="position:relative;width:100%;padding-bottom:100%;background-image:url('https://staging.cohostcdn.org/attachment/ecf59987-c39c-4724-98be-0e386bf71357/background.jpg');background-size:100% 100%;">''')
for i in range(3):
	for j in range(3):
		drawer = drawers[i][j]
		print_drawer(drawer, i, j)
	print_slat(slats[i], i)

printen('''<div style="position: absolute; right: 8%; bottom:8%; height: 10%; background-color:#111;color:white;font-weight:bold;font-size:min(3.5vw,120%);padding:0em 2em;border-radius:15px;display:flex;align-items: center;">can anyone help me find my keys?</div>''')
printen('''</div>''')