#date: 2024-09-10T16:56:58Z
#url: https://api.github.com/gists/b5462dcec2d47ef4e4bdcdf05d3e4303
#owner: https://api.github.com/users/justin2004

# first download https://dlcdn.apache.org/jena/binaries/apache-jena-5.1.0.zip
# then unzip it

% mkdir /tmp/db1
% ls ~/Downloads/labels.ttl
/Users/justin/Downloads/labels.ttl
% ~/Downloads/apache-jena-5.1.0/bin/tdb2.tdbloader --loader=parallel --loc /tmp/db1 ~/Downloads/labels.ttl
11:52:38 INFO  loader          :: Loader = LoaderParallel
11:52:38 INFO  loader          :: Start: /Users/justin/Downloads/labels.ttl
11:52:38 INFO  loader          :: Finished: /Users/justin/Downloads/labels.ttl: 5 tuples in 0.11s (Avg: 46)
11:52:38 INFO  loader          :: Finish - index OSP
11:52:38 INFO  loader          :: Finish - index POS
11:52:38 INFO  loader          :: Finish - index SPO
% cat /tmp/some.rq
select * where {
?s ?p ?o
} limit 2
% ~/Downloads/apache-jena-5.1.0/bin/tdb2.tdbquery --loc /tmp/db1 --query /tmp/some.rq
----------------------------------------------------------------------------------------------------------------------
| s                                           | p                                            | o                     |
======================================================================================================================
| <http://www.wikidata.org/entity/Q6553274>   | <http://www.w3.org/2000/01/rdf-schema#label> | "line number"@en      |
| <http://www.wikidata.org/entity/Q113515824> | <http://www.w3.org/2000/01/rdf-schema#label> | "contiguous lines"@en |
----------------------------------------------------------------------------------------------------------------------
% ~/Downloads/apache-jena-5.1.0/bin/tdb2.tdbquery --results=csv --loc /tmp/db1 --query /tmp/some.rq
s,p,o
http://www.wikidata.org/entity/Q6553274,http://www.w3.org/2000/01/rdf-schema#label,line number
http://www.wikidata.org/entity/Q113515824,http://www.w3.org/2000/01/rdf-schema#label,contiguous lines
