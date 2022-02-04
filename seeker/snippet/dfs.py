#date: 2022-02-04T17:05:44Z
#url: https://api.github.com/gists/f25faf1f954fdb5653804df42cfbc1e6
#owner: https://api.github.com/users/Playdead1709

graph = {
    '5':['3','7'],
    '3':['2','4'],
    '7':['8'],
    '2':[],
    '4':['8'],
    '8':[]
}

visited = set()

def dfs(visited,graph,node):
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited,graph,neighbour)

print("Following is the depth fist search")
dfs(visited,graph,'5')
