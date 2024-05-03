#date: 2024-05-03T16:54:43Z
#url: https://api.github.com/gists/2587012b2cd506bed708e8c734a78bef
#owner: https://api.github.com/users/fabioafreitas

import requests, random
from datetime import datetime, timedelta
import math
import random

def get_random_arbitrary(min_value, max_value):
    return random.uniform(min_value, max_value)

def generate_sin_function(date, amplitude, offset):
    hours = date.hour + date.minute / 60
    angle = ((hours - 12) / 12) * math.pi
    sin_value = math.sin(angle) * amplitude + offset
    return sin_value

def generate_oxigenio(date):
    gen_value = generate_sin_function(date, 2.5, 6)
    randomize_val = random.uniform(gen_value - 0.05, gen_value + 0.05)
    return randomize_val

def generate_temperatura(date):
    gen_value = generate_sin_function(date, 1.5, 27)
    randomize_val = random.uniform(gen_value - 0.01, gen_value + 0.01)
    return randomize_val

def generate_ph(date):
    gen_value = generate_sin_function(date, 0.5, 7)
    randomize_val = random.uniform(gen_value - 0.01, gen_value + 0.01)
    return randomize_val



if __name__ == '__main__':
    date = datetime.now()
    print(generate_oxigenio(date))
    print(generate_temperatura(date))
    print(generate_ph(date))


    # dt_now = int(datetime.timestamp(datetime.now())*1000)

    lista = []  # {"ts":1451649600512, "values":{"key1":"value1", "key2":"value2"}}
    count = 1

    time_between_telemetry_sending = 5 #minutes
    iterations = int(12*30*24*60/time_between_telemetry_sending) # number of 5 minutes durations within 360 days
    for i in range(iterations):
        unixTsMillis = int(datetime.timestamp(date)*1000)
        lista.append({"ts":unixTsMillis,"values":{
            "temperatura":round(generate_temperatura(date),3),
            "ph":round(generate_ph(date),3),
            "oxigenio":round(generate_oxigenio(date),3)
        }})

        date -= timedelta(minutes=time_between_telemetry_sending)


    chunk_size = 20
    for i in range(0, len(lista), chunk_size):
        chunk = lista[i:i+chunk_size]
        print(f"Chunk {i//chunk_size + 1}, len {len(chunk)}")
    

        accessToken = "**********"
        res = requests.post(
            url = f'https: "**********"
            headers={
                "Content-Type":"application/json"
            },
            json=chunk
        )
        print(res.status_code) json=chunk
        )
        print(res.status_code)