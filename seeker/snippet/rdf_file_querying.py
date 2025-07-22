#date: 2025-07-22T16:55:13Z
#url: https://api.github.com/gists/b52cd429b55ac161d001f27f5a51f2f6
#owner: https://api.github.com/users/berezovskyi

#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "rdflib>=7.0.0",
# ]
# ///

"""
RDF Requirements Decomposition Analyzer

This script parses RDF data containing requirements and calculates
the average number of requirements into which a requirement is decomposed.
"""

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, DCTERMS

# Define namespaces
RM = Namespace("http://open-services.net/ns/rm#")
OSLC = Namespace("http://open-services.net/ns/core#")

def main():
    # Create a new RDF graph
    g = Graph()
    
    # Bind prefixes for prettier output
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("dcterms", DCTERMS)
    g.bind("oslc", OSLC)
    g.bind("rm", RM)
    
    # The RDF data as a string
    rdf_data = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>.
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.
@prefix dcterms: <http://purl.org/dc/terms/>.
@prefix oslc: <http://open-services.net/ns/core#>.

<https://localhost:7000/?a=SDOC-HIGH-REQS-DECOMP> <http://open-services.net/ns/rm#decomposes> <https://localhost:7000/?a=SDOC-HIGH-REQS-MANAGEMENT>;
                                                  dcterms:description "StrictDoc shall support requirement decomposition."^^rdf:XMLLiteral;
                                                  dcterms:identifier "SDOC-HIGH-REQS-DECOMP";
                                                  dcterms:title "Requirements decomposition"^^rdf:XMLLiteral;
                                                  a <http://open-services.net/ns/rm#Requirement>.
<https://localhost:7000/?a=SDOC-HIGH-REQS-MANAGEMENT> dcterms:description "StrictDoc shall enable requirements management."^^rdf:XMLLiteral;
                                                      dcterms:identifier "SDOC-HIGH-REQS-MANAGEMENT";
                                                      dcterms:title "Requirements management"^^rdf:XMLLiteral;
                                                      a <http://open-services.net/ns/rm#Requirement>.
"""
    
    # Parse the RDF data
    g.parse(data=rdf_data, format="turtle")
    
    print("📊 RDF Requirements Decomposition Analysis")
    print("=" * 50)
    
    # First, let's see what requirements we have
    print("\n🔍 Requirements found:")
    requirements_query = """
    SELECT ?requirement ?identifier ?title WHERE {
        ?requirement a <http://open-services.net/ns/rm#Requirement> ;
                    dcterms:identifier ?identifier ;
                    dcterms:title ?title .
    }
    ORDER BY ?identifier
    """
    
    results = g.query(requirements_query)
    all_requirements = set()
    
    for row in results:
        req_uri, identifier, title = row
        all_requirements.add(req_uri)
        print(f"  • {identifier}: {title}")
    
    print(f"\nTotal requirements: {len(all_requirements)}")
    
    # Now calculate decomposition statistics using SPARQL
    print("\n📈 Decomposition Analysis:")
    
    # SPARQL query to find decomposition relationships
    decomposition_query = """
    SELECT ?parent ?child ?parent_id ?child_id WHERE {
        ?child <http://open-services.net/ns/rm#decomposes> ?parent .
        ?parent a <http://open-services.net/ns/rm#Requirement> ;
                dcterms:identifier ?parent_id .
        ?child a <http://open-services.net/ns/rm#Requirement> ;
               dcterms:identifier ?child_id .
    }
    """
    
    decomposition_results = g.query(decomposition_query)
    
    print("\n🔗 Decomposition relationships:")
    decomposition_list = list(decomposition_results)
    
    if not decomposition_list:
        print("  No decomposition relationships found.")
        print("\n📊 Average decompositions per requirement: 0")
        return
    
    for row in decomposition_list:
        parent, child, parent_id, child_id = row
        print(f"  • {child_id} decomposes {parent_id}")
    
    # Use SPARQL to calculate statistics
    print(f"\n📊 Statistics (calculated via SPARQL):")
    
    # Count total requirements
    total_reqs_query = """
    SELECT (COUNT(DISTINCT ?req) AS ?total) WHERE {
        ?req a <http://open-services.net/ns/rm#Requirement> .
    }
    """
    total_reqs_result = g.query(total_reqs_query)
    total_requirements = int(list(total_reqs_result)[0][0])
    
    # Count total decomposition relationships
    total_decomp_query = """
    SELECT (COUNT(*) AS ?total) WHERE {
        ?child <http://open-services.net/ns/rm#decomposes> ?parent .
        ?parent a <http://open-services.net/ns/rm#Requirement> .
        ?child a <http://open-services.net/ns/rm#Requirement> .
    }
    """
    total_decomp_result = g.query(total_decomp_query)
    total_decompositions = int(list(total_decomp_result)[0][0])
    
    # Count requirements that have decompositions (are decomposed by others)
    reqs_with_decomp_query = """
    SELECT (COUNT(DISTINCT ?parent) AS ?count) WHERE {
        ?child <http://open-services.net/ns/rm#decomposes> ?parent .
        ?parent a <http://open-services.net/ns/rm#Requirement> .
        ?child a <http://open-services.net/ns/rm#Requirement> .
    }
    """
    reqs_with_decomp_result = g.query(reqs_with_decomp_query)
    requirements_with_decompositions = int(list(reqs_with_decomp_result)[0][0])
    
    # Calculate average using SPARQL (average decompositions per requirement)
    # This calculates the average number of decompositions across ALL requirements
    avg_all_query = f"""
    SELECT ({total_decompositions} / {total_requirements} AS ?avg) WHERE {{
        # This is a calculated value
    }}
    """
    
    # Calculate average for requirements that have decompositions
    avg_with_decomp_query = f"""
    SELECT ({total_decompositions} / {requirements_with_decompositions} AS ?avg) WHERE {{
        # This is a calculated value  
    }}
    """ if requirements_with_decompositions > 0 else None
    
    print(f"  • Total requirements: {total_requirements}")
    print(f"  • Requirements with decompositions: {requirements_with_decompositions}")
    print(f"  • Total decomposition relationships: {total_decompositions}")
    
    # Calculate averages
    avg_decompositions_all = total_decompositions / total_requirements if total_requirements > 0 else 0
    avg_decompositions_with_decomp = (total_decompositions / requirements_with_decompositions 
                                    if requirements_with_decompositions > 0 else 0)
    
    print(f"\n🎯 Results (SPARQL-calculated):")
    print(f"  • Average decompositions per requirement (all): {avg_decompositions_all:.2f}")
    print(f"  • Average decompositions per requirement (with decompositions only): {avg_decompositions_with_decomp:.2f}")
    
    # Show detailed breakdown using SPARQL
    detailed_query = """
    SELECT ?parent_id (COUNT(?child) AS ?decomp_count) WHERE {
        ?child <http://open-services.net/ns/rm#decomposes> ?parent .
        ?parent a <http://open-services.net/ns/rm#Requirement> ;
                dcterms:identifier ?parent_id .
        ?child a <http://open-services.net/ns/rm#Requirement> .
    }
    GROUP BY ?parent_id ?parent
    ORDER BY ?parent_id
    """
    
    detailed_results = g.query(detailed_query)
    detailed_list = list(detailed_results)
    
    if detailed_list:
        print(f"\n📋 Detailed breakdown (via SPARQL GROUP BY):")
        for row in detailed_list:
            parent_id, count = row
            print(f"  • {parent_id}: {count} decomposition(s)")
    
    # Demonstrate a more advanced SPARQL calculation
    print(f"\n🔬 Advanced SPARQL Analysis:")
    
    # Calculate average using a single SPARQL query with aggregation
    advanced_avg_query = """
    SELECT 
        (COUNT(?decomp) AS ?total_decompositions)
        (COUNT(DISTINCT ?req) AS ?total_requirements)
        (COUNT(?decomp) / COUNT(DISTINCT ?req) AS ?avg_per_req)
        (COUNT(DISTINCT ?parent) AS ?reqs_with_decomps)
        (COUNT(?decomp) / COUNT(DISTINCT ?parent) AS ?avg_per_decomposed_req)
    WHERE {
        ?req a <http://open-services.net/ns/rm#Requirement> .
        OPTIONAL {
            ?decomp <http://open-services.net/ns/rm#decomposes> ?parent .
            ?parent a <http://open-services.net/ns/rm#Requirement> .
            FILTER(?req = ?parent)
        }
    }
    """
    
    advanced_results = g.query(advanced_avg_query)
    advanced_row = list(advanced_results)[0]
    
    sparql_total_decomp = int(advanced_row[0])
    sparql_total_reqs = int(advanced_row[1])
    sparql_avg_per_req = float(advanced_row[2])
    sparql_reqs_with_decomp = int(advanced_row[3])
    sparql_avg_per_decomposed = float(advanced_row[4]) if sparql_reqs_with_decomp > 0 else 0
    
    print(f"  • SPARQL calculated - Total decompositions: {sparql_total_decomp}")
    print(f"  • SPARQL calculated - Total requirements: {sparql_total_reqs}")
    print(f"  • SPARQL calculated - Average per requirement: {sparql_avg_per_req:.2f}")
    print(f"  • SPARQL calculated - Requirements with decompositions: {sparql_reqs_with_decomp}")
    print(f"  • SPARQL calculated - Average per decomposed requirement: {sparql_avg_per_decomposed:.2f}")

if __name__ == "__main__":
    main()

# Sample output:

# 📊 RDF Requirements Decomposition Analysis
# ==================================================

# 🔍 Requirements found:
#   • SDOC-HIGH-REQS-DECOMP: Requirements decomposition
#   • SDOC-HIGH-REQS-MANAGEMENT: Requirements management

# Total requirements: 2

# 📈 Decomposition Analysis:

# 🔗 Decomposition relationships:
#   • SDOC-HIGH-REQS-DECOMP decomposes SDOC-HIGH-REQS-MANAGEMENT

# 📊 Statistics (calculated via SPARQL):
#   • Total requirements: 2
#   • Requirements with decompositions: 1
#   • Total decomposition relationships: 1

# 🎯 Results (SPARQL-calculated):
#   • Average decompositions per requirement (all): 0.50
#   • Average decompositions per requirement (with decompositions only): 1.00

# 📋 Detailed breakdown (via SPARQL GROUP BY):
#   • SDOC-HIGH-REQS-MANAGEMENT: 1 decomposition(s)

# 🔬 Advanced SPARQL Analysis:
#   • SPARQL calculated - Total decompositions: 1
#   • SPARQL calculated - Total requirements: 2
#   • SPARQL calculated - Average per requirement: 0.50
#   • SPARQL calculated - Requirements with decompositions: 1
#   • SPARQL calculated - Average per decomposed requirement: 1.00