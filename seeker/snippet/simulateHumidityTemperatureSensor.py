#date: 2021-09-03T16:59:28Z
#url: https://api.github.com/gists/85aaa325890b215c80ada869fea1900b
#owner: https://api.github.com/users/abhishekvp

import requests
import random
import decimal
import time

headers = { 'Content-Type': 'application/json' }

i=0

while i<10:

	temp = decimal.Decimal(random.randrange(300, 350))/10
	hum = decimal.Decimal(random.randrange(800, 850))/10
	data = '{"temperature": '+str(temp)+', "humidity": '+str(hum)+' }'

	response = requests.post('https://demo.thingsboard.io/api/v1/<Your Access Token>/telemetry', headers=headers, data=data)
	print("I am a simulated DHT11 sensor. I sent the following data to ThingsBoard Server - "+str(data))
	print("===")
	time.sleep(5)