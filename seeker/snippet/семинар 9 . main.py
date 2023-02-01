#date: 2023-02-01T16:55:52Z
#url: https://api.github.com/gists/9cdc8b3114da23d2b75ef313f6172056
#owner: https://api.github.com/users/sybery1980

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes,CallbackQueryHandler,MessageHandler
from bot_commands import *
import emoji
import datetime

app = ApplicationBuilder().token("5834723753: "**********"
print("Server start")


app.add_handler(CommandHandler("hi", hi_command))
app.add_handler(CommandHandler("time", time_command))
app.add_handler(CommandHandler("help", help_command))
app.add_handler(CommandHandler("sum", sum_command))
app.add_handler(CommandHandler("days2NY", day2newYear)) 
app.add_handler(CommandHandler("play", play_command))
app.add_handler(MessageHandler())


print(emoji.emojize(f'Привет :thumbs_up:'))

app.run_polling()
print("Server stop")n_polling()
print("Server stop")