#date: 2022-10-21T17:04:20Z
#url: https://api.github.com/gists/7c8f63071f00a65e3d01c74a7c471663
#owner: https://api.github.com/users/rossturk

import requests
from autopaginate_api_call import AutoPaginate
from astro import sql as aql
from astro.sql.table import Table
from airflow.models import DAG, Variable


CONN_ID = "dwh"
TOKEN = "**********"

@aql.dataframe
def get_orbit_members():
    results = AutoPaginate(
        session=requests.session(),
        url="https://app.orbit.love/api/v1/olmqz/members",
        pagination_type="page_number",
        data_path="data",
        paging_param_name="page",
        extra_headers={
            "accept": "application/json",
            "authorization": "**********"
        },
    )

    members = []
    for item in results:
        members.append(item)

    return pd.DataFrame(members)


@aql.dataframe
def get_orbit_orgs():
    results = AutoPaginate(
        session=requests.session(),
        url="https://app.orbit.love/api/v1/olmqz/organizations",
        pagination_type="page_number",
        data_path="data",
        paging_param_name="page",
        extra_headers={
            "accept": "application/json",
            "authorization": "**********"
        },
    )

    orgs = []
    for item in results:
        orgs.append(item)

    return pd.DataFrame(orgs)


with DAG(
    "orbit-members-orgs",
    schedule_interval="@daily",
    start_date=datetime(2022, 10, 20),
    catchup=False,
    default_args={
        "retries": 2,
    },
    tags=["orbit", "openlineage", "marquez"],
) as dag:

    drop_members = aql.drop_table(
        table=Table(
            name="ORBIT_MEMBERS",
            conn_id=CONN_ID,
        )
    )

    get_members = get_orbit_members(
        output_table=Table(
            name="ORBIT_MEMBERS",
            conn_id=CONN_ID,
        )
    )

    drop_orgs = aql.drop_table(
        table=Table(
            name="ORBIT_ORGS",
            conn_id=CONN_ID,
        )
    )

    orgs = get_orbit_orgs(
        output_table=Table(
            name="ORBIT_ORGS",
            conn_id=CONN_ID,
        )
    )

    aql.cleanup()
       )
    )

    aql.cleanup()
