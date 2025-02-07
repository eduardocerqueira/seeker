#date: 2025-02-07T17:02:18Z
#url: https://api.github.com/gists/16496b6791d5f1deaacc9b8b745c9ec9
#owner: https://api.github.com/users/aspose-com-kb

import aspose.tasks as tasks

# Create a Project file
mppProject = tasks.Project()

# Add task and sub task
task = mppProject.root_task.children.add("Summary1")
subtask = task.children.add("Subtask1")

# Save output MPP file
mppProject.save("CreateMPP.mpp", tasks.saving.SaveFileFormat.MPP)