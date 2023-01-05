#date: 2023-01-05T16:54:09Z
#url: https://api.github.com/gists/3c96980bc735eb3e26f7425c46c18e61
#owner: https://api.github.com/users/Akasurde

from api_helper import ShoonyaApiPy, get_time
import logging
import yaml
import pandas as pd
import pyotp
import asyncio
import json
import websockets


# sample
logging.basicConfig(level=logging.DEBUG)

# start of our program
api = ShoonyaApiPy()



# yaml for parameters
with open("cred.yml") as f:
    cred = yaml.load(f, Loader=yaml.FullLoader)
    print(cred)

otp_token = "**********"

cred['factor2'] = "**********"
print(f"OTP generated : {cred['factor2']}")


ret = api.login(
    userid=cred["user"],
    password= "**********"
    twoFA=cred["factor2"],
    vendor_code=cred["vc"],
    api_secret= "**********"
    imei=cred["imei"],
)

# ret = api.get_holdings()
print(ret)



# Set the trailing stop loss threshold (in percent)
stop_loss_threshold = 5

# Set the initial stop loss price to None
stop_loss_price = None

async def get_market_data():
  # Connect to the Shoonya Websocket API
  async with websockets.connect('wss://api.shoonya.com/NorenWSTP/') as websocket:
    # Send a request to subscribe to market data
    await websocket.send('{"type": "subscribe", "symbol": "NSE:INFY"}')
    # Continuously receive and process messages
    async for message in websocket:
      # Parse the message
      data = json.loads(message)
      # If the message is a trade update, update the stop loss price
      if data['type'] == 'trade':
        # Calculate the current stop loss price
        current_stop_loss_price = data['price'] * (1 - stop_loss_threshold / 100)
        # If the stop loss price is not set or the current price is higher than the stop loss price, update the stop loss price
        if stop_loss_price is None or data['price'] > stop_loss_price:
          stop_loss_price = current_stop_loss_price
        # If the current price falls below the stop loss price, place a stop loss order
        elif data['price'] < stop_loss_price:
          # Send a stop loss order to the Shoonya Websocket API
          await websocket.send(f'{"type": "stop_loss", "symbol": "NSE:INFY", "price": {stop_loss_price}}')
          # Reset the stop loss price
          stop_loss_price = None

asyncio.get_event_loop().run_until_complete(get_market_data())


market_data())


