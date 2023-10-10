//date: 2023-10-10T16:58:57Z
//url: https://api.github.com/gists/15f4596af740233f956ca510f4f9d094
//owner: https://api.github.com/users/moha-kun

public List<String> bomberMan(int sec, List<String> grid) {
        // Write your code here
        int m = grid.size(),
            n = grid.get(0).length();
        int[][] grd = new int[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                grd[i][j] = grid.get(i).charAt(j) == 'O' ? 1 : -1;
            }
        for (int k = 1; k <= sec; k++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) {
                    grd[i][j]++;
                    if (grd[i][j] == 3) {
                        grd[i][j] = -1;
                        if (i - 1 >= 0 && grd[i - 1][j] < 3) grd[i - 1][j] = -1;
                        if (i + 1 < m && grd[i + 1][j] + 1 < 3) grd[i + 1][j] = -2;
                        if (j - 1 >= 0 && grd[i][j - 1] < 3) grd[i][j - 1] = -1;
                        if (j + 1 < n && grd[i][j + 1] + 1 < 3) grd[i][j + 1] = -2;
                    }
                }
        grid.clear();
        for (int i = 0; i < m; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < n; j++)
                sb.append(grd[i][j] == -1 ? '.' : 'O');
            grid.add(sb.toString());
        }
        return grid;
    }