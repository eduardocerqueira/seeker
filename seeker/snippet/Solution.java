//date: 2021-09-27T17:07:33Z
//url: https://api.github.com/gists/e585ef13e8d0ac0d8ecaf582812a862e
//owner: https://api.github.com/users/ysyS2ysy

package programmers_49189_FarthestNode;

import java.util.LinkedList;
import java.util.Queue;
/**
 * 일    시: 2021-09-27
 * 작 성 자: 유 소 연
 * 결    과: 메모리초과
 * 이    유: 인접행렬 사용
 * */
class Solution2 {
	static int[][] matrix;
	static Queue<int[]> q = new LinkedList<>();
	static int[] visited;
	static int max = Integer.MIN_VALUE;
	
    public int solution(int n, int[][] edge) {
	int answer = 0;
	matrix = new int[n+1][n+1];
	visited = new int[n+1];
	for (int i = 0; i < edge.length; i++) {
			int a = edge[i][0];
			int b = edge[i][1];
			matrix[a][b] = 1;
			matrix[b][a] = 1;
		}

	// 1로부터의 최단거리를 찾는다 (최단거리 갱신)
	q.add(new int[] {1,0});
	visited[1] = -1;
	bfs(edge);

	// 최단거리가 가장 긴 노드가 몇개인지 찾는다.
	for (int i = 0; i < visited.length; i++) {
			if(visited[i] == max) {
				answer++;
			}
		}

	return answer;
    }

	private void bfs(int[][] edge) {
		while(q.size() > 0) {
			int[] cur = q.poll(); // cur[0]: node, cur[1]: depth
			int node = cur[0];
			for (int i = 0; i < matrix[node].length; i++) {
				if(matrix[node][i] == 1 && visited[i] == 0) {
					visited[i] = cur[1]+1;
					if(max < visited[i]) {
						max = visited[i];
					}
					q.add(new int[] {i, cur[1]+1});
				}
			}
		}
	}
} // end of class