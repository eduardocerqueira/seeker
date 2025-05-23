#date: 2025-05-23T16:43:24Z
#url: https://api.github.com/gists/d578f2d0a23880819fb8398e70c4faee
#owner: https://api.github.com/users/natefox

import networkx as nx
import matplotlib.pyplot as plt

def create_dependency_graph(job_names, dependencies):
    """
    Creates a directed dependency graph from job names and their dependencies.

    Args:
        job_names (list): A list of job names (strings).  These are the nodes in the graph.
        dependencies (list): A list of tuples representing dependencies. Each tuple is 
                             (dependent_job, prerequisite_job), meaning 'dependent_job' 
                             requires 'prerequisite_job' to be completed first.

    Returns:
        networkx.DiGraph:  A directed graph object representing the job dependencies.
                           Returns None if there are circular dependencies (which would make
                           the graph impossible to resolve).
    """

    graph = nx.DiGraph()
    graph.add_nodes_from(job_names)

    for dependent, prerequisite in dependencies:
        if not graph.has_node(dependent):
            graph.add_node(dependent)  # Add if it wasn't already added from job_names
        if not graph.has_node(prerequisite):
            graph.add_node(prerequisite) # Same as above

        graph.add_edge(prerequisite, dependent)  # Edge goes *from* prerequisite *to* dependent

    try:
        nx.find_cycle(graph)  # Check for cycles.  Raises an exception if found.
        return graph
    except nx.NetworkXNoCycle:
        return graph # No cycle found - return the graph
    except nx.NetworkXCyclicGraph as e:
        print("Error: Circular dependency detected! The graph cannot be created.")
        print(e)  # Print details of the cycle if you want to debug it.
        return None


def visualize_graph(graph):
    """
    Visualizes a directed graph using matplotlib.

    Args:
        graph (networkx.DiGraph): The graph object to visualize.
    """
    if graph is None:
        print("No graph to visualize.")
        return

    pos = nx.spring_layout(graph, seed=42)  # Layout algorithm for node positioning. Seed ensures consistent layout.
    nx.draw(graph, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=10, font_weight="bold")
    plt.title("Job Dependency Graph")
    plt.show()


# Example Usage:

if __name__ == "__main__":
    job_names = ["A", "B", "C", "D", "E", "F"]  # List of all job names
    dependencies = [
        ("B", "A"),  # B depends on A
        ("C", "A"),  # C depends on A
        ("D", "B"),  # D depends on B
        ("E", "C"),  # E depends on C
        ("F", "D"),  # F depends on D
    ]

    graph = create_dependency_graph(job_names, dependencies)

    if graph:
        visualize_graph(graph)