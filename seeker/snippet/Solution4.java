//date: 2025-12-29T17:08:57Z
//url: https://api.github.com/gists/7715ef5dc0cd0b9264dc525ac9ba6683
//owner: https://api.github.com/users/qornanali

package org.example.exercisev2.day15;

import java.util.HashMap;
import java.util.Map;

public class Solution4 {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> characterLastSeen = new HashMap<>();
        int leftWindowIdx = 0;
        int bestLength = 0;

        for (int rightWindowIdx = 0; rightWindowIdx < s.length(); rightWindowIdx++) {
            Character character = s.charAt(rightWindowIdx);

            if (characterLastSeen.containsKey(character)) {
                leftWindowIdx = Math.max(leftWindowIdx, characterLastSeen.get(character) + 1);
            }

            characterLastSeen.put(character, rightWindowIdx);
            bestLength = Math.max(bestLength, rightWindowIdx - leftWindowIdx + 1);
        }

        return bestLength;
    }
}
