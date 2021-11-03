#date: 2021-11-03T17:04:32Z
#url: https://api.github.com/gists/b3b4468b92126bc138d9a3de4df818df
#owner: https://api.github.com/users/natashadsilva

import os
import getpass

import datetime
from streamsx.topology.context import ContextTypes, JobConfig
from streamsx.topology import context
from streamsx.topology.topology import Topology

import random, time
import typing
import streamsx.topology.context
import streamsx.topology.composite

import streamsx.database as db


## Change these as needed
DRIVER_NAME = "oracle.jdbc.driver.OracleDriver"
DRIVER_PATH = "/home/wsuser/oracledriver.jar"

db2credentials = {
     "username": "your-db-user-name",
      "password": "your-db-password",
      "jdbcurl": "jdbc:oracle:thin:YOURURL"
    }
def submit_topology(topo):

    import os

    from streamsx.topology import context
    
    username = "USERNAME"
    password = "PASSWORD"
    CP4D_URL = "URL"
    STREAMS_INSTANCE_ID = "sample-streams" # Set instance name
    
    os.environ["STREAMS_USERNAME"] = username
    os.environ["STREAMS_PASSWORD"] = password
    os.environ["STREAMS_INSTANCE_ID"] = STREAMS_INSTANCE_ID
    os.environ["CP4D_URL"] = CP4D_URL
    # Disable SSL certificate verification if necessary
    cfg= {}
    cfg[context.ConfigParams.SSL_VERIFY] = False
 
    # Topology wil be deployed as a distributed app
 
 ## To enable trace, comment out these two lines
  #  job_config = JobConfig(tracing='trace')
   # job_config.add(cfg)
    
    contextType = context.ContextTypes.DISTRIBUTED
    submission_result= context.submit (contextType, topo, config = cfg)
    print(submission_result)
    # The submission_result object contains information about the running application, or job
    if submission_result.job:
        streams_job = submission_result.job
        print ("JobId: ", streams_job.id , "\nJob name: ", streams_job.name)
    return submission_result


#create topology...change the name to something more meaningful

topo = Topology(name="BasicTemplate", namespace="sample")


table_name = 'RUN_SAMPLE_DEMO'

# SQL statements
sql_drop   = 'DROP TABLE ' + table_name
sql_create = 'CREATE TABLE ' + table_name + ' (ID INT, NAME CHAR(30), AGE INT)'


from streamsx.topology.schema import CommonSchema, StreamSchema
# The crt_table is a Stream containing the two SQL statements: sql_drop and sql_create
crt_table = topo.source([sql_drop, sql_create]).as_string()
# drop the table if exist and create a new table in database
stmt = db.JDBCStatement(credentials=db2credentials)


stmt.jdbc_driver_class=DRIVER_NAME

stmt.jdbc_driver_lib= DRIVER_PATH
crt_table.map(stmt, name='CREATE_TABLE', schema=CommonSchema.String)



# The submission_result object contains information about the running application, or job
print("Submitting Topology to Streams for execution..")
submission_result = submit_topology(topo)
