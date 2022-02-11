#date: 2022-02-11T16:42:33Z
#url: https://api.github.com/gists/bca53721a404db35b333b0d75820aadb
#owner: https://api.github.com/users/RobertBemmann

from pagerduty.pagerduty_integration import PagerdutyConfig, pagerduty_integration

##############
# add DAG and tasks definition here
# dag = DAG(
# ...        
# )
##############

# this is your PagerDuty Airflow integration service_id
PAGERDUTY_SERVICE_ID_CONSTANT = "abc"

# this dict handles single task overwrites
task_overwrites = {
	"skipped_failure_task":{
		"page_on_failure":false
	},
	"failure_task":{
		"page_on_failure":true,
		"page_level":{
			"urgency":"low",
			"severity":"warning"
		}
	}
}

# pass default settings in config
pd_config = PagerdutyConfig(
  service_id=PAGERDUTY_SERVICE_ID_CONSTANT,
  page_on_dag_failure=True,
  page_on_task_failure=False,
  severity="high",
  urgency="critical",
  task_overwrites=task_overwrites
)

# add PagerDuty service to DAG and tasks
pagerduty_integration(dag, pd_config)