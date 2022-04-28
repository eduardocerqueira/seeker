#date: 2022-04-28T17:08:22Z
#url: https://api.github.com/gists/e671798a702d48e70070f7558eb156d9
#owner: https://api.github.com/users/aspose-com-gists

from aspose.email import MapiTask, MapiCalendarDailyRecurrencePattern, MapiCalendarRecurrencePatternType, TaskSaveFormat
import datetime as dt

# Create a new task
task = MapiTask("To Do", "Just click and type to add new task", dt.datetime(2018, 6, 1, 21, 30, 0), dt.datetime(2018, 6, 4, 21, 30, 0))

# Set the weekly recurrence
rec = MapiCalendarDailyRecurrencePattern()
rec.pattern_type = MapiCalendarRecurrencePatternType.DAY
rec.period = 1
rec.week_start_day = 0 #0 is for Sunday and so on. WeekStartDay=0
rec.occurrence_count = 0
task.recurrence = rec

# Save task
task.save("task.msg", TaskSaveFormat.MSG)