//date: 2023-01-25T17:05:09Z
//url: https://api.github.com/gists/b69d8d364d7672bc96c1d7c55113d370
//owner: https://api.github.com/users/glaciyan

package cc.glaciyan;

import java.util.LinkedList;
import java.util.List;
import java.util.TreeSet;

public class Main {
    private record Pair<T>(T first, T second) {
    }

    public static void main(String[] args) {
        int[] nums = new int[100_000];
        for (int i = 0; i < nums.length; i++) {
            nums[i] = (int) (Math.random() * 3000);
        }

        var firstPairs = fun1971(nums);
        System.out.println(firstPairs.stream().distinct().count());

        var secondPairs = f(nums);
        System.out.println(secondPairs.stream().distinct().count());
    }


    public static List<Pair<Integer>> fun1971(int[] x) {
        List<Pair<Integer>> sums = new LinkedList<>();

        for (int i = 1; i < x.length - 1; i++)
            for (int j = i + 1; j < x.length; j++)
                if (x[i] + x[j] == 1971)
                    sums.add(new Pair<>(x[i], x[j]));

        return sums;
    }

    public static List<Pair<Integer>> f(int[] x) {
        List<Pair<Integer>> sums = new LinkedList<>();

        // Alle elemente in TreeSet
        TreeSet<Integer> s = new TreeSet<>();
        for (int a : x) {
            s.add(a);
        }

        for (int a : x) {
            // wir suchen nach der anderen Zahl
            if (s.contains(1971 - a)) {
                sums.add(new Pair<>(a, 1971-a));
//                System.out.println(a + ", " + (1971 - a));
            }
        }

        return sums;
    }
}
