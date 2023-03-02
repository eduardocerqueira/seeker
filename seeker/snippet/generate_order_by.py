#date: 2023-03-02T17:01:43Z
#url: https://api.github.com/gists/cee4e805d5665521e70b97d856df6886
#owner: https://api.github.com/users/vibhatha

"""
Version 4
---------

Install Libraries
-----------------

On Mac M1
----------
pip install ibis ibis-substrait pyarrow duckdb-engine


Note that there was an issue with the latest installation of Ibis/Substrait. 
I worked on an older environment.

$ pip list

Package           Version
----------------- --------
atpublic          3.1.1
commonmark        0.9.1
greenlet          2.0.1
ibis              3.2.0
ibis-framework    4.0.0
ibis-substrait    2.19.0
multipledispatch  0.6.0
numpy             1.24.1
pandas            1.5.2
parsy             2.0
pip               22.2.1
protobuf          3.20.1
pyarrow           10.0.1
Pygments          2.14.0
python-dateutil   2.8.2
pytz              2022.7.1
rich              13.1.0
setuptools        63.2.0
six               1.16.0
SQLAlchemy        1.4.46
sqlglot           10.5.2
toolz             0.12.0
typing_extensions 4.4.0

------------------
Generate Substrait Plans and Corresponding SQL via Ibis
Requirements
------------
pip install ibis ibis-substrait pyarrow
Run Script
----------
python generate_orderby.py <path-to-save-directory>
Example:
    python generate_orderby.py /home/veloxuser/sandbox/queries
    
Output
------    
Within the specified folder *.sql files will contain the SQL queries and
*.json files will contain the Substrait plans. Corresponding SQL query and 
JSON plan has the same query id
"""

import os
import sys

import ibis
from ibis_substrait.compiler.core import SubstraitCompiler
from google.protobuf import json_format

def separator(char="="):
    return char * 80

def table():
    return ibis.table([("a", "int"), ("b", "date"), ("c", "int32")], "t",)

def write_sql(expr, fname_base):
    f = open(fname_base + ".sql", "w")
    ibis.show_sql(expr, file=f)
    f.close()

def write_json_plan(expr, fname_base):
    compiler = SubstraitCompiler()
    proto = compiler.compile(expr)
    json_plan = json_format.MessageToJson(proto)
    with open(fname_base+"_substrait.json", "w") as f:
        f.write(json_plan)

def write_sql_and_json(base_path, exprs):
    for idx, expr in enumerate(exprs):
        fname = os.path.join(base_path, "q"+str(idx))
        write_sql(expr, fname)
        write_json_plan(expr, fname)

# input table
t = table()

# orderby
expr7 = t.order_by("a")

if __name__ == "__main__":

    args = sys.argv

    if len(args) != 2:
        print("help>>>")
        print("\t python generate_orderby.py <path-to-save-directory>")
        
    elif os.path.exists(args[1]):
        write_sql_and_json(base_path=args[1], exprs=[expr7])
    else:
        print("Please enter a valid path to save files")