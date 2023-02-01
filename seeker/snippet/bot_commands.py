#date: 2023-02-01T16:55:52Z
#url: https://api.github.com/gists/9cdc8b3114da23d2b75ef313f6172056
#owner: https://api.github.com/users/sybery1980

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import datetime
import play
async def hi_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Привет противный {update.effective_user.first_name}')
    
    

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'/hi(Приветствие)\n/time(Время)\n/help(Вызов команд)\n/days2NY(Сколько осталось до Нового года)\n')


async def time_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'{datetime.datetime.now().time()}')
  
  
async def sum_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mess = update.message.text
    items = mess.split()
    a = int(items[1])
    b = int(items[2])
    await update.message.reply_text(a+b)
        

def days2NY():
    now = datetime.datetime.today()
    NY = datetime.datetime(now.year + 1, 1, 1)
    d = NY-now              
    mm, ss = divmod(d.seconds, 60)
    hh, mm = divmod(mm, 60)
    return('До нового года: {} дней {} часа {} мин {} сек.'.format(d.days, hh, mm, ss))

async def day2newYear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'{days2NY()}')
    
async def play_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mess = update.message.text.split()
    
  
    # await update.message.reply_text(play.showMatrix()) 
    await update.message.reply_text(mess) 
    await update.message.reply_text(play.player(mess)) 
    