//date: 2023-03-01T17:12:04Z
//url: https://api.github.com/gists/43dfbad0635076ee5b2832f1521dfd7a
//owner: https://api.github.com/users/gdtknight

package baekjoon.bfs;

import java.io.BufferedReader;
import java.util.LinkedList;
import java.util.Queue;
import java.util.StringTokenizer;

import common.Initialization;
import common.Problem;

public class _2206_ implements Problem {
  static int[][] dirs = new int[][] {
      { 1, 0 },
      { 0, 1 },
      { -1, 0 },
      { 0, -1 }
  };

  public void solution(String[] args) throws Exception {
    BufferedReader br = Initialization.getBufferedReaderFromClass(this);
    // BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

    StringTokenizer st = "**********"

    int N = "**********"
    int M = "**********"

    int initVal = (N + M + 1) * 2;

    int[][] maze = new int[N][M];

    for (int n = 0; n < N; n++) {
      String[] line = br.readLine().split("");
      for (int m = 0; m < M; m++) {
        int num = Integer.parseInt(line[m]);
        maze[n][m] = num == 1 ? num * -1 : (initVal);
      }
    }

    Queue<WallPos> queue = new LinkedList<>();
    boolean[][][] visited = new boolean[N][M][2];

    visited[0][0][0] = true;
    queue.offer(new WallPos(0, 0, 1, false));

    int minStep = Integer.MAX_VALUE;
    while (!queue.isEmpty()) {
      WallPos cur = queue.poll();
      if (cur.getX() == N - 1 && cur.getY() == M - 1) {
        minStep = minStep > cur.getStep() ? cur.getStep() : minStep;
      }

      for (int i = 0; i < dirs.length; i++) {
        int nextX = cur.getX() + dirs[i][0];
        int nextY = cur.getY() + dirs[i][1];

        int nextStep = cur.getStep() + 1;

        // 미로 벗어난 경우
        if (0 > nextX || 0 > nextY || N <= nextX || M <= nextY) {
          continue;
        }

        // 벽을 한 번 부순 상태에서 벽을 만나는 경우
        if (cur.isThrough() && maze[nextX][nextY] == -1) {
          continue;
        }
        // 벽을 한 번도 부수지 않은 상태에서 벽을 만나는 경우
        else if (!cur.isThrough() && maze[nextX][nextY] == -1) {
          visited[nextX][nextY][1] = true;
          queue.offer(new WallPos(nextX, nextY, nextStep, true));
        }

        // 벽을 한 번 부순 상태에서 아직 방문하지 않은 곳을 방문하는 경우
        else if (cur.isThrough() && !visited[nextX][nextY][1]) {
          visited[nextX][nextY][1] = true;
          queue.offer(new WallPos(nextX, nextY, nextStep, cur.isThrough()));
        }

        // 벽을 부수지 않은 상태에서 아직 방문하지 않은 곳을 방문하는 경우
        else if (!cur.isThrough() && visited[nextX][nextY][0]) {
          visited[nextX][nextY][0] = true;
          queue.offer(new WallPos(nextX, nextY, nextStep, cur.isThrough()));
        }
      }
    }

    System.out.println(minStep == 0 ? -1 : minStep);

    br.close();
  }

}

class WallPos {
  int x;
  int y;
  int step;
  boolean through;

  public WallPos(int x, int y, int step, boolean through) {
    this.x = x;
    this.y = y;
    this.step = step;
    this.through = through;
  }

  public int getX() {
    return x;
  }

  public int getY() {
    return y;
  }

  public int getStep() {
    return step;
  }

  public boolean isThrough() {
    return through;
  }
}ep;
  }

  public boolean isThrough() {
    return through;
  }
}