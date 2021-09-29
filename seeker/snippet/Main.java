//date: 2021-09-29T17:10:04Z
//url: https://api.github.com/gists/1f3c03e57f239d5db00dfa7b3877d47d
//owner: https://api.github.com/users/ysyS2ysy

package backjoon;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.Queue;
import java.util.StringTokenizer;

/**
 * 일    시: 2021-09-15 ~ 2021-09-30
 * 작 성 자: 유 소 연
 * https://www.acmicpc.net/problem/1600
 * BFS
 * */
public class Main_백준_1600_말이되고픈원숭이_유소연_568ms {
	static int K, W, H;
	static int[][] map;
	static boolean[][][] visited;
	static Queue<int[]> q = new LinkedList<>();
	
	public static void main(String[] args) throws NumberFormatException, IOException {
		// bfs
		// visited[][][K+1] 로 관리
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		
		// K: 말처럼 뛸 수 있는 횟수
		K = Integer.parseInt(br.readLine());
		// W: 가로길이, H: 세로길이
		StringTokenizer st = new StringTokenizer(br.readLine(), " ");
		W = Integer.parseInt(st.nextToken());
		H = Integer.parseInt(st.nextToken());
		map = new int[H][W];
		visited = new boolean[H][W][K+1];
		// map
		for (int i = 0; i < H; i++) {
			st = new StringTokenizer(br.readLine(), " ");
			for (int j = 0; j < W; j++) {
				map[i][j] = Integer.parseInt(st.nextToken());
			}
		}
		// end of input
		
		q.add(new int[] {0,0,0,K}); // y,x,cnt,K
		int answer = bfs();
		System.out.println(answer);
	} // end of main

	private static int bfs() {
		int[] hdy = {-2,-1,-2,-1,1,2,1,2}; // 말의 이동방향
		int[] hdx = {-1,-2,1,2,-2,-1,2,1};
		int[] dy = {-1,1,0,0}; // 원숭이 이동방향
		int[] dx = {0,0,-1,1};
		
		while(q.size() > 0) {
			int[] cur= q.poll();
			int cnt = cur[2];
			int k = cur[3];
			
			// 도착했다면 break;
			if(cur[0] == H-1 && cur[1] == W-1) {
				return cnt;
			}
			
			for (int d = 0; d < 8; d++) {
				// k를 언제 쓸지가....
				// k를 쓰고 안쓰고 두가지 경우로 나눠야겠군...
				
				// k를 쓰지 않는 경우
				if(d < 4) {
					int ny = cur[0] + dy[d];
					int nx = cur[1] + dx[d];
					if(inRange(ny,nx) && map[ny][nx] == 0 && !visited[ny][nx][k]) {
						q.add(new int[] {ny,nx,cnt+1,k});
						visited[ny][nx][k] = true;
					}
				}
				// k를 쓰는 경우
				if(k > 0) {
					int hy = cur[0] + hdy[d];
					int hx = cur[1] + hdx[d];
					if(inRange(hy,hx) && map[hy][hx] == 0 && !visited[hy][hx][k-1]) {
						q.add(new int[] {hy,hx,cnt+1,k-1});
						visited[hy][hx][k-1] = true;
					}
				}
			} // end of detect
		} // end of while
		
		return -1;
	} // end of bfs

	private static boolean inRange(int ny, int nx) {
		return ny >= 0 && nx >= 0 && ny < H && nx < W;
	}
} // end of class
