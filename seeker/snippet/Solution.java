//date: 2022-02-04T16:52:42Z
//url: https://api.github.com/gists/e5369acda6bdab26f9239743f105f2f6
//owner: https://api.github.com/users/ysyS2ysy

// 2022-02-05 1:25~1:44
class Solution {
    static boolean[] visited;
    static int max;
    static boolean DEBUG;
    
    public int solution(int k, int[][] dungeons) {
        
        // 슨열로 탐험순서의 모든 경우의 수 구한다.
        visited = new boolean[dungeons.length + 1];
        permutation(0,0,k,dungeons); // depth, path, k, dungeons
        
        return max;
    } // end of solution
    
    
    public void permutation(int depth, int path, int k, int[][] dungeons) {
        
        if(depth == dungeons.length) {
            // 해당 path로 돌아보기
            int numberOfAdventure = go(path, k, dungeons);
            if(max < numberOfAdventure) {
                max = numberOfAdventure;
            }
            return;
        }
        
        for(int i = 1; i <= dungeons.length; i++) {
            if(visited[i]) continue;
            visited[i] = true;
            permutation(depth + 1, path * 10 + i, k, dungeons);
            visited[i] = false;
        }
        
    } // end of permutation
    
    
    public int go(int _path, int k, int[][] dungeons) {
        String path = Integer.toString(_path);
        if(DEBUG) System.out.println("path: "+path);
        int numberOfAdventure = 0;
        
        for(int i = 0; i < path.length(); i++) {
            int no = path.charAt(i) - '0' - 1;
            int leastHealthy = dungeons[no][0];
            int wasteHealthy = dungeons[no][1];
            if(DEBUG) System.out.printf("%d번째 던전 방문 => 최소체력:%d, 소모체력:%d\n", no,dungeons[no][0],dungeons[no][1]);
            if(DEBUG) System.out.printf("현재 체력 %d\n", k);
            
            if(k < leastHealthy) {
                if(DEBUG) System.out.println("현재체력 낮아 탐험할 수 없음");
                continue;
            }
            
            k -= wasteHealthy;
            numberOfAdventure++;
        }
        if(DEBUG) System.out.println("numberOfAdventure "+numberOfAdventure);
        return numberOfAdventure;
    }
    
} // end of class