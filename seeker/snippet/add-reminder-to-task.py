#date: 2022-04-28T17:08:22Z
#url: https://api.github.com/gists/e671798a702d48e70070f7558eb156d9
#owner: https://api.github.com/users/aspose-com-gists

from aspose.email import MapiTask, TaskSaveFormat
import datetime as dt

# Create a new task
task = MapiTask("To Do", "Just click and type to add new task", dt.datetime(2018, 6, 1, 21, 30, 0), dt.datetime(2018, 6, 4, 21, 30, 0))

# Set task reminder
task.reminder_set = True
task.reminder_time = dt.datetime(2018, 6, 1, 21, 30, 0)
task.reminder_file_parameter ="file://Alarm01.wav"

# Save task
task.save("task.msg", TaskSaveFormat.MSG)