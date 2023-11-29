#date: 2023-11-29T16:50:39Z
#url: https://api.github.com/gists/01f3b4a05749f9319056e894ef5b795b
#owner: https://api.github.com/users/tuck1s

from copy import deepcopy

# Expand a tree to the "leaves"
def leaves_of(item, dependencies):

    # return the leaf nodes
    def dfs2(item, level, visited):
        result = []
        if item in dependencies:
            visited.add(item) # mark that we've seen this branch
            print(f"{'.'*level}br: {item}")
            for i2 in dependencies[item]:
                if i2 in visited:
                    raise ValueError(f"Circular dependency detected, on item={item}, loops back to {i2}, visited branches={visited}")
                result += dfs2(i2, level+1, deepcopy(visited))
        else:
            print(f"{'.'*level}leaf:{item}")
            result.append(item) # a leaf node
        return result

    return dfs2(item, 0, set())

# Example usage:
items = ['A', 'B', 'C']

dependency_graph = {
    'A': ['B', 'C'],
    'B': ['C'],
    'C': ['D'], # Acceptable
#    'C': ['D', 'A'], # This intentionally shows detection a fatal loop!
}

for item in items:
    print("RESULT: ", item, leaves_of(item, dependency_graph))
