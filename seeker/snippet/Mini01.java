//date: 2022-05-18T16:51:45Z
//url: https://api.github.com/gists/80e8714cc57d145e954607d23c0783ab
//owner: https://api.github.com/users/fairyerica

public class Mini01 {
    public static int solution(int n) {
        return 0;
    }
    public static void main(String[] args){
        System.out.println("[구구단 출력]");
        for (int i = 1; i < 10; i++) {
            for (int j = 1; j < 10; j++) {
                System.out.print(String.format("%02d X %02d = %02d\t",i ,j ,i*j));
            }
            System.out.println();
        }
    }
}
