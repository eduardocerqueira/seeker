//date: 2023-11-01T16:56:07Z
//url: https://api.github.com/gists/93c984d8752aa008a3ae6257b0904530
//owner: https://api.github.com/users/ramyo564


public class Main {
    public static void main(String[] args) {

        System.out.println("[구구단 출력]");
        int start = 1;

        while (start < 10) {
            for (int i = 1; i < 10; i++) {
                int result = i*start;
                if (result < 10){
                    String multiplicationTable = String.format("0%s x 0%s = 0%s", i, start, result);
                    System.out.print(multiplicationTable + "    ");
                }else {
                    String multiplicationTable = String.format("0%s x 0%s = %s", i, start, result);
                    System.out.print(multiplicationTable + "    ");
                }
            }
            System.out.println();
            start += 1;
        }



    }
}