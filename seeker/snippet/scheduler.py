#date: 2023-03-09T16:44:15Z
#url: https://api.github.com/gists/9ba72baef3c828de8d9f26c72802080b
#owner: https://api.github.com/users/ihorfusion

from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler


if __name__ == '__main__':
    now = datetime.now()
    datetime_now = datetime(year=now.year, month=now.month, day=now.day, 
                            hour=now.hour, second=5)

    scheduler = AsyncIOScheduler()
    scheduler.add_job(send_signal, 
                      trigger='interval', 
                      start_date=datetime_now,
                      minutes=60,
                      seconds=5,
                      kwargs={'bot': bot})
    scheduler.start()

    executor.start_polling(dp, skip_updates=True)