#date: 2023-04-14T17:04:56Z
#url: https://api.github.com/gists/ace2bd219d77ed46b1c102eb4c11836d
#owner: https://api.github.com/users/ianmcook

import google.auth
import ibis
from ibis import _

credentials, billing_project = google.auth.default()

conn = ibis.bigquery.connect(billing_project, 'bigquery-public-data.samples')
t = conn.table('github_nested')

expr = (
    t.mutate(
         # get the hour in UTC during which a repo was created
         hour=_.created_at.to_timestamp('%Y/%m/%d %T %z').hour(),

         # compute the UTC offset to adjust in the next expression
         utc_offset=_.created_at.split(' ')[2].cast('int64') // 100
     )

     # group by the adjusted hour, count and sort by descending count
     .group_by(hour=_.hour + _.utc_offset)
     .count()
     .order_by(ibis.desc('count'))

     # sum up the number of repos that were created between midnight and 4 AM
     # local time
     .aggregate(
         total=_['count'].sum(),
         night_owl=_['count'].sum(where=_.hour.between(0, 4))
     )

     # compute the percentage of repos created between midnight and 4 AM
     .mutate(night_owl_perc=_.night_owl / _.total)
)

df = expr.execute()
