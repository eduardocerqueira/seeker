//date: 2022-09-09T17:09:49Z
//url: https://api.github.com/gists/69c119d34ccd5709eaba0d963db9b6a3
//owner: https://api.github.com/users/dal-2

/*
최홍준
미니 과제 6번
 */

import java.util.Random;

public class mini_project_6 {
    public static void main(String[] args) {
        Random rnd = new Random();
        String[][] list = {
                {"1","이재명","0"},
                {"2","윤석열","0"},
                {"3","심상정","0"},
                {"4","안철수","0"}
        };

        for (int i = 1; i <= 10000; i++) {
            int vote = rnd.nextInt(40) + 1;
            if (vote >= 1 && vote <= 10) list[0][2] = String.valueOf(Integer.valueOf(list[0][2]) + 1);
            else if (vote >= 11 && vote <= 20) list[1][2] = String.valueOf(Integer.valueOf(list[1][2]) + 1);
            else if (vote >= 21 && vote <= 30) list[2][2] = String.valueOf(Integer.valueOf(list[2][2]) + 1);
            else if (vote >= 31 && vote <= 40) list[3][2] = String.valueOf(Integer.valueOf(list[3][2]) + 1);

            System.out.println(String.format("\n" + "[투표진행율]: %.2f%%, %d명 투표 => %s", (double) i / 100f, i,
                    list[(int) Math.ceil((double) vote / 10) - 1][1]));
            for (int k = 0; k < 4; k++) {
                System.out.println(String.format("[기호:%s] %s: %02.2f%%, (투표수:%s)", list[k][0], list[k][1],
                        Double.valueOf(list[k][2]) / 100, list[k][2]));
            }
        }

        String[] max = new String[list.length];
        for (String[] compare : list) {
            if (max[2] == null || Integer.valueOf(max[2]) <= Integer.valueOf(compare[2])) {
                for (int i = 0; i < compare.length; i++) {
                    max[i] = compare[i];
                }
            }
        }

        String result = "";
        System.out.print("[투표결과] ");
        for (String[] compare : list) {
            if (Integer.valueOf(max[0]) != Integer.valueOf(compare[0])
                    && Integer.valueOf(max[2]) == Integer.valueOf(compare[2])) {
                result += String.format("\"%s\" ", compare[1]);
            }
        }
        if (result.length() != 0) {
            System.out.print(String.format("\"%s\" %s투표수가 동률이므로 무효", max[1], result));
        } else {
            System.out.println("당선인: " + max[1]);
        }

    }
}
