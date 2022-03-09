//date: 2022-03-09T17:07:04Z
//url: https://api.github.com/gists/a0f1806fad64dff988deea00b0a1c4b4
//owner: https://api.github.com/users/TehleelMir

class Solution {
    public int removeCoveredIntervals(int[][] i) {
        Arrays.sort(i, new customComparator());
        int result = 0, right = 0;
        for(int[] x : i)
            if(x[1] > right) {
                result++;
                right = x[1];
            }
        return result;
    }
}

class customComparator implements Comparator<int[]> {
    @Override
    public int compare(int[] val1, int[] val2) {
        if(val1[0] > val2[0]) return 1; 
        if(val1[0] < val2[1]) return -1;
        
        if(val1[1] < val2[1]) return 1;
        if(val1[1] > val2[1]) return -1;
        
        return 0;
    }
}