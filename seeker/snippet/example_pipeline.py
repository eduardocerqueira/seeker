#date: 2021-12-10T16:57:45Z
#url: https://api.github.com/gists/2245849b749ca27c6e8f751ceb73d320
#owner: https://api.github.com/users/duyttran

DRIVER_TRACKING_DATA_PIPELINE = Pipeline(
    name="driver_tracking_data_pipeline",
    owner="driver_tracking_team@keeptruckin.com",
    description="Pipeline generates driver metrics including distance, performance, and safety metrics",
    schedule="0 0 * * 1",
    email=["driver_tracking_team@keeptruckin.com"],
    tables=[WEEKLY_DRIVER_KM_DRIVEN, MOTION_EVENTS],
    opsgenie_alert=OpsgenieAlert(responders=DPL_TEAM, priority=PRIORITY_HIGH),
)