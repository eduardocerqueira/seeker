#date: 2022-12-13T16:49:17Z
#url: https://api.github.com/gists/a00132a1d9818ec85fa61b13e9def8bd
#owner: https://api.github.com/users/sorphwer

from excel2neo4j.py import create_meta
node_schema = [
    {'name' : 'DataTransfer' , 'col': [5,6,7,8,11,12,13,14,15,19,20,21,22,23] },
    {'name' : 'OverseasRecipient' , 'col': [17]},
    {'name' : 'ChinaEntity' , 'col' : [16]},
    {'name' :'System' , 'col' : [9,10] , 'split': 9, 'separator' : [',','和']},
    {'name' :'Subteam','col' : [3] , 'split': 3,'separator' : [',','和']},
    {'name' :'Function','col': [2]},
    {'name' :'DataSubject','col': [4]},
    {'name' :'Region', 'col' :[18] , 'split': 18,'separator' : [',','和']}  
]
relationship_schema = [
    {'name':'TARGET_ENTITY','start':'DataTransfer','end':'OverseasRecipient','value':{'test':123} },
    {'name':'ORIGIN_ENTITY','start':'DataTransfer','end':'ChinaEntity'},
    {'name':'HAS_DATASUBJECT','start':'DataTransfer','end':'DataSubject'},
    {'name':'IN_FUNCTION','start':'Subteam','end':'Function'},
    {'name':'HAS_TEAM','start':'DataTransfer','end':'Subteam'},
    {'name':'LOCATE_IN','start':'OverseasRecipient','end':'Region'},
    {'name':'LOCATE_IN','start':'System','end':'Region'},
    {'name':'RUNNING_ON','start':'DataTransfer','end':'System'}
]
n,l = create_meta('input/test.xlsx',
            "Clinical Trial",
            1,
            8,
            node_schema,
            relationship_schema
           )