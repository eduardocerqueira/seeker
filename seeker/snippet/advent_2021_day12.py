#date: 2021-12-13T17:10:57Z
#url: https://api.github.com/gists/352f3cc2f841918d0023727394c81f86
#owner: https://api.github.com/users/sharno

graph =     [
  ("GC", "zi"),
  ("end", "zv"),
  ("lk", "ca"),
  ("lk", "zi"),
  ("GC", "ky"),
  ("zi", "ca"),
  ("end", "FU"),
  ("iv", "FU"),
  ("lk", "iv"),
  ("lk", "FU"),
  ("GC", "end"),
  ("ca", "zv"),
  ("lk", "GC"),
  ("GC", "zv"),
  ("start", "iv"),
  ("zv", "QQ"),
  ("ca", "GC"),
  ("ca", "FU"),
  ("iv", "ca"),
  ("start", "lk"),
  ("zv", "FU"),
  ("start", "zi")]

bigraph = [(b,a) for (a,b) in graph] + graph

bigraphDict = {}
for (a,b) in bigraph:
  bigraphDict[a] = bigraphDict.get(a, []) + [b]

def expand(path):
  return [path + [node] for node in bigraphDict[path[-1]]]

def validate(path):
  return path[-1].isupper() or path[-1] not in path[:-1]

paths = [["start"]]
finishedPaths = []
while len(paths) > 0:
  expandedPaths = [p for path in paths for p in expand(path)]
  newPaths = [p for p in expandedPaths if validate(p)]
  finishedPaths += [p for p in newPaths if p[-1] == "end"]
  paths = [p for p in newPaths if p[-1] != "end"]
print(len(finishedPaths))