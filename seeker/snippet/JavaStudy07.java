//date: 2024-01-12T17:00:04Z
//url: https://api.github.com/gists/741041bc08d22c7dacb04c33154c0bd2
//owner: https://api.github.com/users/hwangchahae

import java.util.*;
//      제로베이스 21기 황차해 미니과제 7

public class Main {
    public static void main(String[] args) {


        System.out.println("[로또 당첨 프로그램]\n");
        Scanner scanner = new Scanner(System.in);
        int intValue = 64;
        int arraySize = 5;


        while (true) {
            try {
                System.out.print("로또 개수를 입력해 주세요.(숫자 1 ~ 10):");
                int ea = scanner.nextInt();

                if (ea > 0 && ea < 11) {
                    for (int i = 1; i < ea + 1; i++) {
                        Integer[] lottoNum = new Integer[6];
                        for (int j = 0; j < lottoNum.length; j++) {
                            Random random = new Random();
                            lottoNum[j] = random.nextInt(45) + 1;
                            if (random.nextInt(45) + 1 == lottoNum[j]) {

                            }

                        }


                        intValue++;

                        char charValue = (char) intValue;
                        Arrays.sort(lottoNum);
                        System.out.println();
                        System.out.print(charValue + "  ");

                        for (int k = 0; k < lottoNum.length; k++) {

                            System.out.printf("%02d", lottoNum[k]);
                            if (k < lottoNum.length - 1) {
                                System.out.print(", ");
                            }
                        }
                    }
                    System.out.println();
                    Integer[] luckLottoNum = new Integer[6];
                    for (int a = 0; a < luckLottoNum.length; a++) {
                        Random random = new Random();
                        luckLottoNum[a] = random.nextInt(45) + 1;
                    }

                    Arrays.sort(luckLottoNum);
                    System.out.println();
                    System.out.println("[로또 발표]\n");
                    System.out.print("   ");
                    for (int b = 0; b < luckLottoNum.length; b++) {
                        System.out.printf("%02d", luckLottoNum[b]);
                        if (b < luckLottoNum.length - 1) {
                            System.out.printf(", ");
                        }
                    }
                    break;
                } else {
                    System.out.println("숫자 1 ~ 10 까지 값으로 다시 입력하세요");

                }
            } catch (InputMismatchException e) {
                System.out.println("error = " + e);
                System.out.println("error = 숫자만 입력하세요.");
                scanner.nextLine();
            }
        }
    }
}

//미니과제가 맞는걸까?
//정녕 이걸 다 푸는게 기본인건가.... 미니일 정도로?
//자꾸 에러만 만들어낸다 ㅠㅠ

