//date: 2024-06-12T17:05:09Z
//url: https://api.github.com/gists/5619e09e3da17158d442546fac5157a8
//owner: https://api.github.com/users/neilghosh

/*
 * Sort the given array by their frequency and then by their own value
 */

import java.io.*;
import java.util.*;

class SortByValueThenKey {
  public static void main(String[] args) {
    Map<Integer, Integer> ageFreq = new HashMap<>();
    int[] ages = {7, 1, 2, 3, 7, 1, 54, 5, 65, 7};
    for(int age : ages) {
      Integer currentValue = ageFreq.putIfAbsent(Integer.valueOf(age), 1);
      if(currentValue != null) {
        ageFreq.put(Integer.valueOf(age), ++currentValue);
      }
    }
    List<Map.Entry<Integer, Integer>> entries = new ArrayList<>(ageFreq.entrySet());
    entries.sort(new MapValueComparator(ageFreq));
    for(Map.Entry<Integer, Integer> entry : entries) {
      System.out.println(entry.getKey());
    }
  }

  public static class MapValueComparator implements Comparator<Map.Entry<Integer, Integer>>{
    Map<Integer, Integer> mapToCompare;
    public MapValueComparator(Map<Integer, Integer> mapToCompare){
      this.mapToCompare = mapToCompare;
    }
    public int compare(Map.Entry<Integer, Integer> first, Map.Entry<Integer, Integer> second ) {
      if(first.getValue()<second.getValue()) return 1;
      else if(second.getValue() < first.getValue()) return -1;
      else {
        if(first.getKey() < second.getKey()) return 1;
        else return -1;
      }
    }
  }
}
