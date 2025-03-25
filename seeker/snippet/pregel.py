#date: 2025-03-25T17:10:37Z
#url: https://api.github.com/gists/61bb36265cf6eefdb9cf06b80be973b1
#owner: https://api.github.com/users/rjurney

from graphframes.lib import AggregateMessages as AM
from graphframes.examples import Graphs
from pyspark.sql.functions import sum as sqlsum


g = Graphs(spark).friends()  # Get example graph

# For each user, sum the ages of the adjacent users
msgToSrc = AM.dst["age"]
msgToDst = AM.src["age"]
agg = g.aggregateMessages(
    sqlsum(AM.msg).alias("summedAges"),
    sendToSrc=msgToSrc,
    sendToDst=msgToDst)
agg.show()