//date: 2025-07-18T17:03:10Z
//url: https://api.github.com/gists/270a23bb0d50219cbf29c4b724d8e20e
//owner: https://api.github.com/users/Mr-Techganesh

// LeetCode: Fruit Into Baskets
// Sliding Window | Time: O(n), Space: O(1)

public class FruitIntoBaskets {
    public int totalFruit(int[] fruits) {
        Map<Integer, Integer> count = new HashMap<>();
        int left = 0, max = 0;

        for (int right = 0; right < fruits.length; right++) {
            count.put(fruits[right], count.getOrDefault(fruits[right], 0) + 1);

            while (count.size() > 2) {
                count.put(fruits[left], count.get(fruits[left]) - 1);
                if (count.get(fruits[left]) == 0) {
                    count.remove(fruits[left]);
                }
                left++;
            }

            max = Math.max(max, right - left + 1);
        }

        return max;
    }
}
