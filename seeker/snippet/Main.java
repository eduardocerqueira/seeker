//date: 2022-10-12T17:12:48Z
//url: https://api.github.com/gists/022283cb6da57821423bf6d420e310f4
//owner: https://api.github.com/users/ebrar0

import java.util.Scanner;
public class Main {
public static void main(String []args){
    Scanner input = new Scanner(System.in);
    int yil;
    System.out.println("Doğum Yılını Gir:");
    yil=input.nextInt();
    int hes=yil%12;
    switch (hes){
        case 0:
            System.out.print("Çin Zodyağı Burcunuz : Maymun");
            break;
        case 1:
            System.out.print("Çin Zodyağı Burcunuz : Horoz");
            break;
        case 2:
            System.out.print("Çin Zodyağı Burcunuz : Köpek");
            break;
        case 3:
            System.out.print("Çin Zodyağı Burcunuz : Domuz");
            break;
        case 4:
            System.out.print("Çin Zodyağı Burcunuz : Fare");
            break;
        case 5:
            System.out.print("Çin Zodyağı Burcunuz : Öküz");
            break;
        case 6:
            System.out.print("Çin Zodyağı Burcunuz : Kaplan");
            break;
        case 7:
            System.out.print("Çin Zodyağı Burcunuz : Tavşan");
            break;
        case 8:
            System.out.print("Çin Zodyağı Burcunuz : Ejderha");
            break;
        case 9:
            System.out.print("Çin Zodyağı Burcunuz : Yılan");
            break;
        case 10:
            System.out.print("Çin Zodyağı Burcunuz : At");
            break;
        case 11:
            System.out.print("Çin Zodyağı Burcunuz : Koyun");
            break;
    }
}
}
