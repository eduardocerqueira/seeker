#date: 2022-02-22T16:49:59Z
#url: https://api.github.com/gists/9f91780ef13bb3a604adfe4a8b92c10b
#owner: https://api.github.com/users/HaidarChaito

class Node():
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node
class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node

def shortest_path(source, target):
    """
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.

    If no possible path, returns None.
    """
    frontier = QueueFrontier()
    start = Node(state=source, parent=None, action=None)
    frontier.add(start)
    explored = set()
    while True:
        if frontier.empty():
            return None
        node = frontier.remove()
        if node.state == target:
            solution = []
            while node.parent is not None:
                solution.append((node.action, node.state))
                node = node.parent
            solution.reverse()
            return solution
        explored.add(node.state)
        for mov, pid in neighbors_for_person(node.state):
            if not frontier.contains_state(pid) and pid not in explored:
                child = Node(state=pid, parent=node, action=mov)
                if child.state == target:
                    solution = []
                    while child.parent is not None:
                        solution.append((child.action, child.state))
                        child = child.parent
                    solution.reverse()
                    return solution
                frontier.add(child)