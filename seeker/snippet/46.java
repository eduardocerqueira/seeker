//date: 2022-03-09T17:06:05Z
//url: https://api.github.com/gists/0890fd3113f0fecb47564623fbee17c5
//owner: https://api.github.com/users/TehleelMir

class Solution {
    public int removeCoveredIntervals(int[][] intervals) {
        int size = intervals.length;
        
        for(int i=0; i<intervals.length; i++) {
            int[] arr = intervals[i];
            int one = arr[0];
            int two = arr[1];
            
            for(int j=0; j<intervals.length; j++) {
                if(i == j) continue;
                int[] brr = intervals[j];
                int oneb = brr[0];
                int twob = brr[1];
                
                if(oneb <= one && twob >= two) {
                    size--;
                    break;
                }
            }
        }
        return size;
    }
}