#date: 2022-06-24T16:48:46Z
#url: https://api.github.com/gists/d79baf086d1f60d32eeca96d489aa4f5
#owner: https://api.github.com/users/acirtep

spin_firefox_task = SSHOperator(
    task_id='spin_firefox',
    ssh_conn_id='ssh_localhost',
    command='bash --login /scripts/run_firefox_container.sh '
)
