//date: 2021-10-01T01:47:08Z
//url: https://api.github.com/gists/d955e3160067d20758b3fb04849b2956
//owner: https://api.github.com/users/ysyS2ysy

package 정올;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.PriorityQueue;
import java.util.StringTokenizer;

/**
 * 일    시: 2021-10-01
 * 작 성 자: 유 소 연
 * */
public class Main_정올_1113_119구급대_유소연_219ms {
	static int M, N, Y, X;
	static int[][] map;
	static boolean[][] visited;
	static int answer;
	
	static class Point implements Comparable<Point> {
		int y;
		int x;
		int dir;
		int rotationCnt;
		public Point(int y, int x, int dir, int rotationCnt) {
			this.y = y;
			this.x = x;
			this.dir = dir;
			this.rotationCnt = rotationCnt;
		}
		@Override
		public int compareTo(Point o) {
			// 회전수 오름차순, 만약 회전 수 같으면 y,x오름차순
			if(this.rotationCnt != o.rotationCnt) {
				return Integer.compare(this.rotationCnt, o.rotationCnt);
			}else {
				if(this.y != o.y) {
					return Integer.compare(this.y, o.y);
				}else {
					return Integer.compare(this.x, o.x);
				}
			}
		}
	}
	
	public static void main(String[] args) throws IOException {
		// 문제를 거꾸로 접근해볼까... 도착점 -> 시작점까지의 최소 코너도는횟수(방향전환횟수)
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringTokenizer st = new StringTokenizer(br.readLine(), " ");
		// M: 세로크기, N: 가로크기
		M = Integer.parseInt(st.nextToken());
		N = Integer.parseInt(st.nextToken());
		map = new int[M][N];
		visited = new boolean[M][N];
		// 도착지점 좌표 y,x
		st = new StringTokenizer(br.readLine(), " ");
		Y = Integer.parseInt(st.nextToken());
		X = Integer.parseInt(st.nextToken());
		// map (1:도로, 2:장애물)
		for (int i = 0; i < M; i++) {
			st = new StringTokenizer(br.readLine(), " ");
			for (int j = 0; j < N; j++) {
				map[i][j] = Integer.parseInt(st.nextToken());
			}
		}
		// end of input
		
		PriorityQueue<Point> q = new PriorityQueue<>();
		q.add(new Point(Y,X,0,0)); // y,x,현재 방향,방향전환횟수
		bfs(q);
		System.out.println(answer);
		
	}

	private static void bfs(PriorityQueue<Point> q) {
		int[] dy = {-1,1,0,0};
		int[] dx = {0,0,-1,1};
		
		while(q.size() > 0) {
			Point cur = q.poll();
//			System.out.println(cur.y+", "+cur.x+": "+cur.rotationCnt+": "+cur.dir);
			visited[cur.y][cur.x] = true;
			
			int dir = cur.dir; // 현재 방향
			int rotationCnt = cur.rotationCnt; // 방향회전 수
			
			if(cur.y == 0 && cur.x == 0) {
				answer = cur.rotationCnt;
				return;
			}

			for (int d = 0; d < 4; d++) {
				int ny = cur.y + dy[d];
				int nx = cur.x + dx[d];
				if(inRange(ny,nx) && map[ny][nx] == 1 && !visited[ny][nx]) {
					if(dir != d && (cur.y!=Y || cur.x!=X)) {
						q.add(new Point(ny,nx,d,rotationCnt+1));
					}else {
						q.add(new Point(ny,nx,d,rotationCnt));
					}
				}
			} // end of for
		} // end of while
	} // end of bfs

	private static boolean inRange(int ny, int nx) {
		return ny >= 0 && nx >= 0 && ny < M && nx < N;
	}
}
