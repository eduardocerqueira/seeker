#date: 2023-09-07T17:03:40Z
#url: https://api.github.com/gists/f7b6771cb278d03d3bc7a55645862cf7
#owner: https://api.github.com/users/harrietdede

import boto3
import time

# Replace these variables with your specific values
instance_name = "your-notebook-instance-name"
max_retries = 5

sagemaker = boto3.client("sagemaker")

for i in range(max_retries):
    try:
        response = sagemaker.start_notebook_instance(
            NotebookInstanceName=instance_name
        )
        print(f"Attempt {i + 1}: Started SageMaker notebook instance {instance_name}")
        break
    except Exception as e:
        print(f"Attempt {i + 1}: Failed to start instance: {str(e)}")
        if i < max_retries - 1:
            print("Retrying in 30 seconds...")
            time.sleep(30)
        else:
            print("Maximum retry attempts reached. Instance could not be started.")
