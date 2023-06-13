//date: 2023-06-13T17:00:48Z
//url: https://api.github.com/gists/4285cb225b4c3b089e74188fcc6f13f7
//owner: https://api.github.com/users/gtrght

import java.util.ArrayDeque;
import java.util.Queue;
import java.util.TreeSet;

public class Solution {
    static class Node implements Comparable<Node> {
        private final int time;
        private final int value;

        Node(int time, int value) {
            this.time = time;
            this.value = value;
        }

        /**
         * This is a different implementation compared to one we had
         * in the hacker-rank - added by-time disambiguation
         */
        @Override
        public int compareTo(Node node) {
            int compare = -Integer.compare(value, node.value);

            if (compare == 0) return Integer.compare(time, node.time); //all we need is to disambiguate conflict

            return compare;
        }
    }

    private final int windowSize;
    private final TreeSet<Node> set = new TreeSet<>();
    private final Queue<Node> queue = new ArrayDeque<>();

    public Solution(int windowSize) {
        this.windowSize = windowSize;
    }

    public void setValue(int time, int value) {
        Node node = new Node(time, value);
        queue.add(node);
        set.add(node);
    }

    public int maxValue(int time) {
        int minTime = time - windowSize;

        while (!queue.isEmpty() && queue.peek().time <= minTime) {
            set.remove(queue.poll());
        }

        if (set.isEmpty()) {
            return Integer.MIN_VALUE;
        } else {
            return set.first().value;
        }
    }

    public static void main(String[] args) {
        Solution solution = new Solution(3);
        solution.setValue(0, 6);
        solution.setValue(1, 4);
        solution.setValue(2, 6);
        solution.setValue(3, 5);
        System.out.println(solution.maxValue(4));
        System.out.println(solution.maxValue(5));
        System.out.println(solution.maxValue(6));
        System.out.println(solution.maxValue(7));
    }
}
