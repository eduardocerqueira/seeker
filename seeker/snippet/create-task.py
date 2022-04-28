#date: 2022-04-28T17:08:22Z
#url: https://api.github.com/gists/e671798a702d48e70070f7558eb156d9
#owner: https://api.github.com/users/aspose-com-gists

from aspose.email import MapiTask, MapiTaskHistory, MapiTaskOwnership, MapiSensitivity, MapiTaskStatus, TaskSaveFormat
import datetime as dt

# Create a new task
task = MapiTask("To Do", "Just click and type to add new task", dt.datetime(2018, 6, 1, 21, 30, 0), dt.datetime(2018, 6, 4, 21, 30, 0))

# Set task properties
task.percent_complete = 20
task.estimated_effort = 2000
task.actual_effort = 20
task.history = MapiTaskHistory.ASSIGNED
task.last_update = dt.datetime(2018, 6, 1, 21, 30, 0)
task.users.owner = "Darius"
task.users.last_assigner = "Harkness"
task.users.last_delegate = "Harkness"
task.users.ownership = MapiTaskOwnership.ASSIGNERS_COPY
task.companies = [ "company1", "company2", "company3" ]
task.categories = [ "category1", "category2", "category3" ]
task.mileage = "Some test mileage"
task.billing = "Test billing information"
task.users.delegator = "Test Delegator"
task.sensitivity = MapiSensitivity.PERSONAL
task.status = MapiTaskStatus.COMPLETE
task.estimated_effort = 5

# Save task
task.save("task.msg", TaskSaveFormat.MSG)