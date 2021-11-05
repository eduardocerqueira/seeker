#date: 2021-11-05T17:01:02Z
#url: https://api.github.com/gists/7312f53fb975d720104e0d3767013742
#owner: https://api.github.com/users/muditlambda

import datetime
import win32com.client
 
scheduler = win32com.client.Dispatch('Schedule.Service')
scheduler.Connect()
root_folder = scheduler.GetFolder('\\')
task_def = scheduler.NewTask(0)
 
# Start time of script
start_time = datetime.datetime.now() + datetime.timedelta(minutes=1)
 
# for running it one time
TASK_TRIGGER_DAILY = 1
trigger = task_def.Triggers.Create(TASK_TRIGGER_DAILY)
 
#Repeat for 10 day
num_of_days = 10
trigger.Repetition.Duration = "P"+str(num_of_days)+"D"
 
#For every 6 hour
trigger.Repetition.Interval = "PT6H" 
trigger.StartBoundary = start_time.isoformat()
 
# Create action
TASK_ACTION_EXEC = 0
action = task_def.Actions.Create(TASK_ACTION_EXEC)
action.ID = 'TRIGGER BATCH'
action.Path = r'C:\Users\vinayak\selenium_test\env\Scripts\pytest.exe'
action.Arguments = r'C:\Users\vinayak\selenium_test\main.py'
 
# Set parameters
task_def.RegistrationInfo.Description = 'Test Task'
task_def.Settings.Enabled = True
task_def.Settings.StopIfGoingOnBatteries = False
 
# Register task
# If task already running, it will be updated
TASK_CREATE_OR_UPDATE = 6
TASK_LOGON_NONE = 0
root_folder.RegisterTaskDefinition(
   'Test Task',  # Task name
   task_def,
   TASK_CREATE_OR_UPDATE,
   '',  # No user
   '',  # No password
   TASK_LOGON_NONE
)
