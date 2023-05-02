//date: 2023-05-02T17:08:22Z
//url: https://api.github.com/gists/0242d00f6b9866d5873b4c4568cae633
//owner: https://api.github.com/users/ChanggiMin

// 민창기
public class Practice1 {
    public static void main(String[] args) {
        System.out.println("[구구단 출력]");
        for (int i = 1; i < 10; i++) {
            for (int j = 1; j < 10; j++) {
                String table = String.format("%02d X %02d = %02d\t", j, i, i * j);
                System.out.print(table);;
            }
            System.out.println();
        }
    }
}