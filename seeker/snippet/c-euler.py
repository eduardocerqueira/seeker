#date: 2021-12-17T17:03:17Z
#url: https://api.github.com/gists/eae78e32d885bf0ff87ab5ed96a53b5d
#owner: https://api.github.com/users/keruboDecra

# importing networkx as nx to draw random graphs
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph  # importing the library that generates random
# undirected graphs
import random  # importing the library for generating random numbers
import matplotlib.pyplot as plt

n = 10  # the number of vertices we want in the graph
p = random.random()  # The probability of the edges between the vertices. Using the random.random so that it can be

x = erdos_renyi_graph(n, p)  # storing the inbuilt function for generating graphs as the variable x

# Creating an empty list where we will append nodes that have even degrees
even_nodes = []
# Creating an empty list where will be appending nodes that have odd degrees
odd_nodes = []
# Looping through the vertex in the graph to find those vertices that are even so that we can add them in our empty
# even_nodes lists
for node in x.nodes:
    # print("Check",node)
    # print("Check nodes has a degree of", x.degree(node))
    # Isolating the nodes that have atleast one or more degree, and are even
    if p > 0 and x.degree(node) != 0 and x.degree(node) % 2 == 0:
        even_nodes.append(node)
        # print("Node ",node, "is even")
    # Isolating the nodes that have a degree of 0
    elif x.degree(node) == 0:
        print(" This graph is not connected")
        nx.draw_random(x, with_labels=True)
        plt.show()
        exit()
    # Adding the nodes that don't have an even number of degree
    else:
        odd_nodes.append(node)
# sorting the even_nodes list so that we can compare it later with the nodes in the graph
even_nodes.sort()
# Comparing the even_nodes and a list of vertices to see if they are equal. This will show that the graph is euler if
# they are equal.
if even_nodes == list(x.nodes):
    print("This is a euler circuit")

# Printing some statements to the user in case the even_nodes and the list of vertices are not equal
else:
    print("This graph is connected")
    print(f"This is not an Euler circuit because {odd_nodes} have odd degrees")
    print("Try running it again. you never know")

# estimating the probability of getting an euler circuit in an infinite sample space
# We will assume that the number of trials n is 10
print("\nCalculating the probability of getting an euler circuit ..")
no_of_trials = 10
print((no_of_trials - 1) / 2 ** no_of_trials)

# Using the nx module that we imported earlier to draw the random graph
nx.draw_random(x, with_labels=True)
plt.show()  # Plotting the graph to the user for better visualisation. Here we use the matploitlib.pyplot