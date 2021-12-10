#date: 2021-12-10T16:56:39Z
#url: https://api.github.com/gists/d532a36baf283590f52d0d5af5b4875b
#owner: https://api.github.com/users/duyttran

WEEKLY_DRIVER_KM_DRIVEN = Table(
    "weekly_driver_km_driven",
    feature_area="driver_stats",
    storage="snowflake",
    schema=SCHEMA,
    deps=[MOTION_EVENTS],
    spark_fn=generate_table,
    table_comment=(
        """
        Contains the number of KMs driven by each driver
        """
    ),
    incremental_interval=timedelta(days=7),
)