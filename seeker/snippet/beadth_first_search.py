#date: 2025-05-28T17:00:54Z
#url: https://api.github.com/gists/be55dfb09775d9dbcb67e0db22faa6f3
#owner: https://api.github.com/users/alsidneio

class Graph:
    def breadth_first_search(self, v):
        visited = []
        waiting_queue = [v]
        while len(waiting_queue) != 0: 
            curr_vrtx = waiting_queue.pop(0)
            visited.append(curr_vrtx)
            
            ## Unpacking set into an array and then sorting 
            neighbors = sorted([*self.graph[curr_vrtx]])
            print(f"current vertex: {curr_vrtx}")
            print(f"current neighbors: {neighbors}")
            
            for neighbor in neighbors: 
                if (neighbor not in visited) and (neighbor not in waiting_queue):
                    waiting_queue.append(neighbor)

            print(f"current waiting queue: {waiting_queue}")

        return visited
        

    # don't touch below this line

    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u in self.graph.keys():
            self.graph[u].add(v)
        else:
            self.graph[u] = set([v])
        if v in self.graph.keys():
            self.graph[v].add(u)
        else:
            self.graph[v] = set([u])

    def __repr__(self):
        result = ""
        for key in self.graph.keys():
            result += f"Vertex: '{key}'\n"
            for v in sorted(self.graph[key]):
                result += f"has an edge leading to --> {v} \n"
        return result