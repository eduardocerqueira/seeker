#date: 2021-10-27T17:02:04Z
#url: https://api.github.com/gists/8052d9f023328a610b6f8d3a9b893ef0
#owner: https://api.github.com/users/omairaasim

import airflow
from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import requests

args = {
    'owner': 'airflow',
}

dag_branch = DAG(
    dag_id = "BranchPythonOperator_demo",
    default_args=args,
    # schedule_interval='0 0 * * *',
    schedule_interval='@once',    
    dagrun_timeout=timedelta(minutes=60),
    description='use case of branch operator in airflow',
    start_date = airflow.utils.dates.days_ago(1))






def make_request():
    
    response = requests.get('https://ghibliapi.herokuapp.com/films/')
    

    if response.status_code == 200:
        return 'conn_success'
       
        data = response.json()
    elif response.status_code == 404:
         return 'not_reachable'
    else:
        print("Unable to connect API or retrieve data.")




make_request = BranchPythonOperator(
        task_id='make_request',
        python_callable= make_request,
dag=dag_branch
    )
response = DummyOperator(
task_id='conn_success',
dag=dag_branch
)
noresponse = DummyOperator(
task_id='not_reachable',
dag=dag_branch
    )
   
make_request  >> [response , noresponse]


if __name__ == '__main__ ':
  dag_branch.cli()

