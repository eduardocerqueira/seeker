//date: 2021-10-07T16:48:17Z
//url: https://api.github.com/gists/e39d1bffde3f53b3264f712b274393e4
//owner: https://api.github.com/users/mustafa-qamaruddin

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Stack;

class Traverse {
  int[][] graph;
  boolean[] visited;
  List<List<Integer>> retPaths;
  int noop = -1;

  public Traverse(int[][] graph) {
    this.graph = graph;
    retPaths = new ArrayList<>();
    reset();
  }

  void reset() {
    visited = new boolean[graph.length];
    for (int i = 0; i < visited.length; i++) {
      visited[i] = false;
    }
  }

  List<Integer> dfs(int node) {
    List<Integer> ret = new ArrayList<>();
    Stack<Integer> stack = new Stack<>();
    Deque<Integer> q = new ArrayDeque<>();
    stack.push(node);
    while (!stack.empty()) {
      int currNode = stack.pop();
      visited[currNode] = true;
      q.addFirst(currNode);
      for (int i = 0; i < graph[currNode].length; i++) {
        if (!visited[graph[currNode][i]]) {
          stack.push(graph[currNode][i]);
        }
      }
    }
    return new ArrayList<>(q);
  }

  List<List<Integer>> findDependencies() {
    for (int i = 0; i < graph.length; i++) {
      reset();
      var path = dfs(i);
      retPaths.add(path);
    }
    return retPaths;
  }
}

public class TaskDep {
  public static void main(String[] args) {
    int[][] adjLs = {
        {1, 3}, // 0
        {2},    // 1
        {},     // 2
        {},     // 3
        {2, 3}, // 4
    };
    var g = new Traverse(adjLs);
    List<List<Integer>> deps = g.findDependencies();
    for (int i = 0; i < deps.size(); i++) {
      System.out.println(i + ": " + deps.get(i));
    }
  }
}

