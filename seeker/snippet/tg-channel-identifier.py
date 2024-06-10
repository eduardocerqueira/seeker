#date: 2024-06-10T17:01:32Z
#url: https://api.github.com/gists/fa9c06b09bc8cd9de55a9ce1223b73a3
#owner: https://api.github.com/users/erseco

"""
Telegram Channel Identifier

Este script obtiene el ID de un canal de Telegram especificado por su título.
Las variables de configuración se leen desde un archivo `.env`.

### Instrucciones:

1. Instala las dependencias necesarias:
    pip install telethon python-dotenv

2. Crea un archivo `.env` en el mismo directorio que este script con el siguiente contenido:
    API_ID=your_api_id
    API_HASH=your_api_hash
    PHONE_NUMBER=your_phone_number

3. Ejecuta el script:
    python tg-channel-identifier.py "Título del Canal"

Este script utiliza el modo sincrónico de Telethon.
"""

import os
import sys
from telethon.sync import TelegramClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read configuration from environment variables
api_id = int(os.getenv('API_ID'))
api_hash = os.getenv('API_HASH')
phone = os.getenv('PHONE_NUMBER')

# Initialize the Telegram client
client = TelegramClient('session_name', api_id, api_hash)

# Function to get the ID of a channel by its title
def get_channel_id(channel_title):
    with client:
        for dialog in client.iter_dialogs():
            if dialog.name == channel_title:
                return dialog.entity.id
    return None

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python tg-channel-identifier.py \"Channel Title\"")
    else:
        channel_title = sys.argv[1]
        channel_id = get_channel_id(channel_title)
        if channel_id:
            print(f'Channel ID: {channel_id}')
        else:
            print('Channel not found')
