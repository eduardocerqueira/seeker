#date: 2021-11-11T17:03:17Z
#url: https://api.github.com/gists/92f7cb2c34d11cc69a6d5b58f57ac40f
#owner: https://api.github.com/users/heylouiz

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import os
import random
import re
import tempfile

import youtube_dl

import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, RegexHandler, Filters, CallbackQueryHandler
from telegram.ext.dispatcher import run_async
from telegram.utils.helpers import escape_markdown

# Load config
with open('config.json') as config_file:
    CONFIGURATION = json.load(config_file)

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Globals
logger = logging.getLogger(__name__)

START_MESSAGE = "Oi! Eu baixo vídeos, só me mandar o link e eu vou tentar baixar"

HELP_MESSAGE = "Help!"

URL_REGEX = r'((?:http(?:s)?:\/\/|(?:www\.))(?:www\.)?.*?)\s'

# Command functions
def start(bot, update):
    """Send a message when the command /start is issued."""
    update.message.reply_text(START_MESSAGE)


def help(bot, update):
    """Send a message when the command /help is issued."""
    update.message.reply_text(HELP_MESSAGE)


def send_video(bot, update, url):
    filename = '/tmp/%s%s.mp4' % (update.message.chat.id, update.message.message_id)
    ydl_opts = {'outtmpl': filename, 'format': 'mp4'}
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except youtube_dl.utils.UnsupportedError:
        update.message.reply_text('Link não suportado.')
        return
    except youtube_dl.utils.DownloadError:
        update.message.reply_text('Falha ao baixar vídeo')
        return
    try:
        with open(filename, 'rb') as video:
            update.message.reply_video(video=video, timeout=99999)
    except telegram.error.BadRequest:
            update.message.reply_text('Falha ao baixar vídeo')
    os.remove(filename)


@run_async
def message_handler(bot, update):
    print(update.message)
    url = re.search(r'(http.*)', update.message.text)
    if url:
        send_video(bot, update, url.group(1))

def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    """Start the bot."""
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(CONFIGURATION["telegram_token"])

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Add command handlers
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # Add message handlers
    dp.add_handler(MessageHandler(Filters.text, message_handler))

    # Add error handler
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
