#date: 2022-01-03T16:53:56Z
#url: https://api.github.com/gists/c9256a0c0c4a0f888ac68cec07d5b480
#owner: https://api.github.com/users/danfunk

ready_tasks = workflow.get_ready_user_tasks()
while len(ready_tasks) > 0:
  for task in ready_tasks:
    if isinstance(task.task_spec, UserTask):
      show_form(task) # We'll get to this in just a second
      workflow.complete_task_from_id(task.id)
   workflow.do_engine_steps()
   ready_tasks = workflow.get_ready_user_tasks()