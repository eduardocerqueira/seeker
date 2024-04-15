//date: 2024-04-15T16:39:47Z
//url: https://api.github.com/gists/048ce93e486516fa29b0a7d9e070972a
//owner: https://api.github.com/users/samruds1

import java.util.*;

public class Solution {
  
  public static int[][] mergeIntervals(int[][] intervals) {
    
    LinkedList<int[]> result = new LinkedList<>();
    
    if (intervals.length == 0) {
      return new int[][]{}; 
    }
    
    result.add(intervals[0]);
    
    for (int i=1; i <intervals.length; i++) {
      int[] interval = intervals[i];
      int[] lastAddedInterval = result.getLast();
      int currStart = interval[0];
      int currEnd = interval[1];
      int prevEnd = lastAddedInterval[1];
      
      if (currStart <= prevEnd) { 
        lastAddedInterval[1] = Math.max(currEnd, prevEnd);
      } else {
        result.add(new int[]{currStart, currEnd}); 
      }
      
    }
    return result.toArray(new int[][]{});
  }
}