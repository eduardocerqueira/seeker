#date: 2023-11-03T16:48:26Z
#url: https://api.github.com/gists/b59a0218ab8c75000eb241c99343bd6b
#owner: https://api.github.com/users/ToroData

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setQuery("""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?abstract
    WHERE { 
        <http://dbpedia.org/resource/Computer_security> rdfs:comment ?abstract 
        FILTER (LANG(?abstract) = "en")
    }
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
for result in results["results"]["bindings"]:
    print(result["abstract"]["value"])
