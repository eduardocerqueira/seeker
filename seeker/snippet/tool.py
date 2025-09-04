#date: 2025-09-04T16:43:22Z
#url: https://api.github.com/gists/7e4570f083c11fc6e633293c9feb029f
#owner: https://api.github.com/users/rfrneo4j

1. Your Tool (with StructuredTool)


# app/llm/tools.py
import os
from typing import Any, Optional
from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase, RoutingControl
from pydantic import BaseModel, Field

def retrieve_location_context(
    functional_location_id: Optional[str] = None,
    functional_location_description: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Retrieve object references from Neo4j graph"""
    query = """
    MATCH (O:Object)-[R1:REFERENCES]-(T:Object) 
    RETURN O.name, T.name, (startNode(R1)).name as startNode, R1.verb, (endNode(R1)).name as endNode
    """

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth= "**********"
    )

    results = driver.execute_query(
        query,
        database_=os.getenv("NEO4J_DATABASE"),
        routing_=RoutingControl.READ,
        result_transformer_=lambda r: r.data(),
    )
    return results

class RetrieveLocationContextInput(BaseModel):
    functional_location_id: Optional[str] = Field(None, description="The functional location ID to retrieve context for.")
    functional_location_description: Optional[str] = Field(None, description="The functional location description to retrieve context for.")

retrieve_location_context_tool = StructuredTool.from_function(
    func=retrieve_location_context,
    args_schema=RetrieveLocationContextInput,
)


Option 1: New Graph Node

# In agentservice.py
def _location_context_node(self, state: MultiTurnState) -> MultiTurnState:
    """New graph node for object references"""
    # Call your tool
    results = retrieve_location_context_tool.invoke({"functional_location_id": "dummy"})
    
    # Update state with results
    state.location_results = results
    return state

def _build_graph(self):
    builder = StateGraph(state_schema=MultiTurnState)
    builder.add_node("text2cypher", self._text2cypher_node)
    builder.add_node("location_context", self._location_context_node)  # New node
    builder.add_node("evaluate", self._evaluate_cypher_node)
    builder.add_node("format", self._format_response_node)
    
    # Add routing logic to choose between nodes
    # ... routing based on question type


Option 2: Integrate into existing node


# In agentservice.py - modify existing _text2cypher_node
def _text2cypher_node(self, state: MultiTurnState) -> MultiTurnState:
    # Check if question needs object references
    if "references" in state.current_question.lower() or "object" in state.current_question.lower():
        # Call your tool instead of text2cypher
        results = retrieve_location_context_tool.invoke({"functional_location_id": "dummy"})
        state.results = results
    else:
        # Existing text2cypher logic
        # ... existing code ...
    
    return state # ... existing code ...
    
    return state