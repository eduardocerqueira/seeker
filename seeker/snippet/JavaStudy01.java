//date: 2023-11-10T17:05:16Z
//url: https://api.github.com/gists/29b65cad4bd03338afa57ae73fa01187
//owner: https://api.github.com/users/seungwonyang1995

/*
  양승원
*/
public class Gugudan {
        public static void main(String[] args) {
            String text = "[구구단 출력]";
            System.out.println(text);
            for (int i = 1; i < 10; i++) {
                for (int j = 1; j < 10; j++) {
                    System.out.printf("0%d x 0%d = %02d", j, i, (j * i));
                    System.out.print("\t");
                }
                System.out.println();
            }
        }
    }
