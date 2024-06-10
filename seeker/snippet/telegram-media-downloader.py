#date: 2024-06-10T16:54:01Z
#url: https://api.github.com/gists/9ea3052e3b2e441606a13de093eabe22
#owner: https://api.github.com/users/erseco

"""
Telegram Channel Media Downloader

Este script descarga archivos multimedia de un canal de Telegram especificado y evita descargar archivos ya existentes.
Las variables de configuración se leen desde un archivo `.env`.

### Instrucciones:

1. Instala las dependencias necesarias:
    pip install telethon python-dotenv

2. Crea un archivo `.env` en el mismo directorio que este script con el siguiente contenido:
    API_ID=your_api_id
    API_HASH=your_api_hash
    PHONE_NUMBER=your_phone_number
    CHANNEL_ID=your_channel_id
    SAVE_DIR=downloads
    LOG_FILE=log.txt

3. Ejecuta el script:
    python tg_downloader.py

Este script utiliza el modo sincrónico de Telethon y verifica si los archivos ya existen antes de descargarlos.
También registra todas las descargas y errores en un archivo log.
"""
import os
from telethon.sync import TelegramClient
from telethon.errors import ChannelPrivateError, ChannelInvalidError, ChatAdminRequiredError
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read configuration from environment variables
api_id = int(os.getenv('API_ID'))
api_hash = os.getenv('API_HASH')
phone = os.getenv('PHONE_NUMBER')
channel_id = int(os.getenv('CHANNEL_ID'))
save_dir = os.getenv('SAVE_DIR', 'downloads')
log_file = os.getenv('LOG_FILE', 'log.txt')

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the Telegram client
client = TelegramClient('session_name', api_id, api_hash)

# Function to log messages
def log_message(message):
    with open(log_file, 'a') as log:
        log.write(f"{datetime.now()}: {message}\n")

# Function to download media from a message
def download_media(message):
    if message.media:
        filename = message.file.name or f"{message.id}_{message.date.strftime('%Y-%m-%d_%H-%M-%S')}.unknown"
        file_path = os.path.join(save_dir, filename)
        if not os.path.exists(file_path):
            try:
                client.download_media(message.media, file_path)
                print(f"Downloaded: {file_path}")
                log_message(f"Downloaded: {file_path}")
            except Exception as e:
                error_message = f"Failed to download message ID {message.id} ({file_path}): {e}"
                print(error_message)
                log_message(error_message)
        else:
            print(f"Skipped (already exists): {file_path}")
            log_message(f"Skipped (already exists): {file_path}")

# Main function to download all media from the channel
def main():
    client.start()
    try:
        entity = client.get_entity(channel_id)
        for message in client.iter_messages(entity):
            print(f"Processing message ID {message.id}")
            download_media(message)
    except (ChannelPrivateError, ChannelInvalidError, ChatAdminRequiredError) as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        log_message(error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        log_message(error_message)
    finally:
        client.disconnect()

if __name__ == '__main__':
    main()
