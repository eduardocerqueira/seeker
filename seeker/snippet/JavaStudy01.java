//date: 2024-02-01T16:53:34Z
//url: https://api.github.com/gists/efef1b936a1064b1ca217624dac59b17
//owner: https://api.github.com/users/suuuubin

// 제출자 류수빈

public class Main {
    public static void main(String[] args) {
        System.out.println("[구구단 출력]");
        for(int i=1; i<10; i++){
            for(int j=1; j<10; j++){
                String str = String.format("%02d x %02d = %02d",j,i,j*i);
                System.out.print(str+"\t");
            }
            System.out.println();
        }
    }
}