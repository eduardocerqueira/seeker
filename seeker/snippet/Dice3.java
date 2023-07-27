//date: 2023-07-27T16:41:11Z
//url: https://api.github.com/gists/3d8a73f17712f74e21b9397e3669b450
//owner: https://api.github.com/users/dalinaum

import java.util.Collections;
import java.util.HashMap;

class Solution {
    public int solution(int a, int b, int c, int d) {
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(a, map.getOrDefault(a, 0) + 1);
        map.put(b, map.getOrDefault(b, 0) + 1);
        map.put(c, map.getOrDefault(c, 0) + 1);
        map.put(d, map.getOrDefault(d, 0) + 1);
        
        int mapSize = map.size();
        
        if (mapSize == 1) {
            return 1111 * a;
        } else if (mapSize == 2) {
            int[] values = map.values()
                .stream()
                .mapToInt(Integer::intValue)
                .toArray();
            int[] keys = map.keySet()
                .stream()
                .mapToInt(Integer::intValue)
                .toArray();
            
            int firstValue = values[0];
            if (firstValue == 2) {
                int p = keys[0];
                int q = keys[1];
                return (p + q) * Math.abs(p - q);
            } 
            int p;
            int q;
            if (firstValue == 3) {
                p = keys[0];
                q = keys[1];
            } else {
                q = keys[0];
                p = keys[1];
            }
            return (int) Math.pow(10 * p + q, 2);
        } else if (mapSize == 3) {
            return map.entrySet()
                .stream()
                .filter(pair -> pair.getValue() != 2)
                .mapToInt(pair -> pair.getKey())
                .reduce(1, (total, key) -> total * key);
        } else {
            return Collections.min(map.keySet());
        }
    }
}