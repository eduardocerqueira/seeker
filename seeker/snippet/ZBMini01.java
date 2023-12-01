//date: 2023-12-01T16:48:49Z
//url: https://api.github.com/gists/cc8c573d26b9de7f0cbdcf79c033c115
//owner: https://api.github.com/users/Lyckabc

/*
제로베이스 백엔드 20기 신동호
미니과제01 - 구구단 출력
 */

public class ZBMini01 {
    public static void main(String[] args) {
        int ans = 0;
        System.out.println("[구구단 출력]");
        for (int i = 1; i < 10; i++) {
            for (int j = 1; j < 10; j++) {
                ans = i * j;
                System.out.print(String.format("%02d x %02d = %02d    ", j, i, ans));
            }
            System.out.println();
        }
    }
}