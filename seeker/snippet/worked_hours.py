#date: 2022-05-02T16:47:32Z
#url: https://api.github.com/gists/35a15b0bbc9339eefadb890c82c191b3
#owner: https://api.github.com/users/macndesign

from datetime import timedelta
import time


def save_file(total_work_formatted_hours, start_lunch_time, end_lunch_time):
    filename = input('year-month-day: ')
    with open('date-{}.txt'.format(filename), "w") as f:
        report = 'Work duration: {}\nLunch time: {} - {}'.format(
            total_work_formatted_hours, start_lunch_time, end_lunch_time)
        f.write(report)
        return report


def worked_hours(start_work, end_work, start_lunch, end_lunch):
    start_work = timedelta(hours=start_work[0], minutes=start_work[1])
    end_work = timedelta(hours=end_work[0], minutes=end_work[1])
    start_lunch = timedelta(hours=start_lunch[0], minutes=start_lunch[1])
    end_lunch = timedelta(hours=end_lunch[0], minutes=end_lunch[1])

    total_work_seconds = end_work.seconds - start_work.seconds
    total_lunch_seconds = end_lunch.seconds - start_lunch.seconds

    total_seconds = total_work_seconds - total_lunch_seconds

    total_work_formatted_hours = time.strftime(
        '%H:%M', time.gmtime(total_seconds))

    start_lunch_time = time.strftime('%H:%M', time.gmtime(start_lunch.seconds))
    end_lunch_time = time.strftime('%H:%M', time.gmtime(end_lunch.seconds))

    return save_file(
        total_work_formatted_hours,
        start_lunch_time, end_lunch_time
    )


start_work_hour = int(input('Start work hour: '))
start_work_minute = int(input('Start work minute: '))
print('---')
start_lunch_hour = int(input('Start lunch hour: '))
start_lunch_minute = int(input('Start lunch minute: '))
print('---')
end_lunch_hour = int(input('End lunch hour: '))
end_lunch_minute = int(input('End lunch minute: '))
print('---')
end_work_hour = int(input('End work hour: '))
end_work_minute = int(input('End work minute: '))

print(
    worked_hours(
        [start_work_hour, start_work_minute],
        [end_work_hour, end_work_minute],
        [start_lunch_hour, start_lunch_minute],
        [end_lunch_hour, end_lunch_minute]
    )
)
