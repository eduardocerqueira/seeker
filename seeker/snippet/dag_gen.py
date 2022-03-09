#date: 2022-03-09T17:02:48Z
#url: https://api.github.com/gists/df0b4779bea4b1d0e8cdfb9bf867a328
#owner: https://api.github.com/users/milimetric

projectview_ready = HiveTriggeredHQLTaskFactory(
    'run_hql_and_arhive',
    default_args=default_args,
    ...
)

archive = ArchiveTaskFactory(...)

projectview_ready.sensors() >> projectview_ready.etl() >> archive()


class HiveTriggeredHQLTaskFactory(...):
    def __init__(sources, ...):
        self.sources = ...
        
    def sensors(self):
        return self._build_sensors(self.sources, ...)
    
    def etl(self):
        # I looked and it seems that dag= is not required
        # https://airflow.apache.org/docs/apache-airflow/1.10.6/_api/airflow/models/index.html#airflow.models.BaseOperator
        return SparkSqlOperator(...)

cass ArchiveTaskFactory(...):
    def __init__(...):
        ...
    
    def __call__(...):
        return ArchiveOperator(...)