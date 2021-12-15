#date: 2021-12-15T17:14:43Z
#url: https://api.github.com/gists/734dcea63263fa5891b9bf22716951a4
#owner: https://api.github.com/users/tomasonjo

from neo4j import GraphDatabase
# Change the host and user/password combination to your neo4j
# Will not work with a localhost bolt url
host = 'bolt://44.200.249.124:7687'
user = 'neo4j'
password = 'battle-manpower-sand'
driver = GraphDatabase.driver(host,auth=(user, password))