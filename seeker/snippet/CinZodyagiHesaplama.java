//date: 2022-07-14T16:53:49Z
//url: https://api.github.com/gists/7c766b94893f3679d21a7df8b2856282
//owner: https://api.github.com/users/aryesilmen

package Giris;

import java.util.Scanner;

public class CinZodyagiHesaplama {
    public static void main(String[] args) {

        int birthdate, kalan;
        String zodiac = "";
        boolean isError=false;
        Scanner input = new Scanner(System.in);
        System.out.print("Doğum Yılınızı Giriniz: ");
        birthdate = input.nextInt();
        kalan = birthdate % 12;
        if (birthdate >= 1900 && birthdate < 2023) {
            switch (kalan) {
                case 0:
                    zodiac = "Maymun";
                    break;
                case 1:
                    zodiac = "Horoz";
                    break;
                case 2:
                    zodiac = "Köpek";
                    break;
                case 3:
                    zodiac = "Domuz";
                    break;
                case 4:
                    zodiac = "Fare";
                    break;
                case 5:
                    zodiac = "Öküz";
                    break;
                case 6:
                    zodiac = "Kaplan";
                    break;
                case 7:
                    zodiac = "Tavşan";
                    break;
                case 8:
                    zodiac = "Ejderha";
                    break;
                case 9:
                    zodiac = "Yılan";
                    break;
                case 10:
                    zodiac = "At";
                    break;
                case 11:
                    zodiac = "Koyun";
                    break;
                default:
                    isError=true;
                    break;
            }
        }else {
            isError=true;
        }
        if (isError){
            System.out.println("Hatalı Veri Girdiniz ! Tekrar deneyiniz.");
        }else {
            System.out.println("Çin Zodyağı Burcunuz: " + zodiac);
        }
    }
}
