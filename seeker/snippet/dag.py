#date: 2023-03-01T17:03:42Z
#url: https://api.github.com/gists/878f6c9595f93759f482a3793dbf5794
#owner: https://api.github.com/users/edkeeble

@dag(
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
)
def my_dag():
    @task()
    def add(**kwargs) -> Dict:
        """
        Add something
        """
        a = kwargs.get("dag_run").conf["a"]
        b = kwargs.get("dag_run").conf["b"]

        return a+b

    @task()
    def multiply(x: int, y: int=10) -> Dict:
        """
        Multiply two values
        """
        return x * y

    addition_result = add()
    multiply(addition_result)
