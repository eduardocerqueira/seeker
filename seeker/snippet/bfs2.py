#date: 2024-08-22T17:07:27Z
#url: https://api.github.com/gists/ad2e0810e167c020bacb71a25919c9ba
#owner: https://api.github.com/users/oliviagallucci

from collections import deque

class financialGraph:
    def __init__(self):
        self.graph = {}

    def add_instrument(self, instrument, dependencies):
        # add financial instrument and its dependencies to graph
        self.graph[instrument] = dependencies

    def bfs(self, start_instrument):
        visited = set()
        queue = deque([start_instrument])

        print("bfs traversal:")
        while queue:
            current_instrument = queue.popleft()

            if current_instrument not in visited:
                print(current_instrument)

                # mark current instrument as visited
                visited.add(current_instrument)

                # enqueue dependencies for exploration
                if current_instrument in self.graph:
                    queue.extend(self.graph[current_instrument])

# example usage:
if __name__ == "__main__":
    # create financial graph
    financial_graph = financialGraph()

    # add instruments and dependencies to graph
    financial_graph.add_instrument("stock A", ["stock B", "option C"])
    financial_graph.add_instrument("stock B", ["bond D"])
    financial_graph.add_instrument("option C", ["stock D"])
    financial_graph.add_instrument("bond D", [])

    # perform bfs traversal starting from specific instrument
    financial_graph.bfs("stock A")

# bfs traversal:
# stock A
# stock B
# option C
# bond D
# stock D