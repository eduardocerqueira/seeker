#date: 2024-02-21T17:02:19Z
#url: https://api.github.com/gists/f73fe9adb8e369f46a886a7ef3b73b29
#owner: https://api.github.com/users/elliottcordo

from time import sleep
import boto3
client = boto3.client('athena')

calculation_response = client.start_session(
    Description='job_session',
    WorkGroup='aws-meetup',
    EngineConfiguration={
        'CoordinatorDpuSize': 1,
        'DefaultExecutorDpuSize': 1,
        'MaxConcurrentDpus': 60        }
    )

sleep(5) # give time for the session to complete
session_id = calculation_response.get('SessionId')
session_state =calculation_response.get('State')
print(session_id)
print(session_state)

with open("url_parse.py","r") as f:
    noteboook_code = f.read()
    
execution_response = client.start_calculation_execution(
    SessionId=session_id,
    Description='daily job',
    CodeBlock=noteboook_code,
)

calc_exec_id = execution_response.get('CalculationExecutionId')
print(calc_exec_id)

while True:
    exec_status_response = client.get_calculation_execution_status(
        CalculationExecutionId=calc_exec_id
    )
    exec_state = exec_status_response.get('Status').get('State')
    print(exec_state)
    if exec_state in ['CANCELED','COMPLETED','FAILED']:
        print(exec_status_response)
        break
    sleep(5)

client.terminate_session(SessionId=session_id)
