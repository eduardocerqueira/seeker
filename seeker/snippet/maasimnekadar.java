//date: 2022-11-25T16:57:57Z
//url: https://api.github.com/gists/39ce72fc08727963f90b1a6ed9062a17
//owner: https://api.github.com/users/tweezz-haha

import java.util.Scanner;


public class maasimnekadar {
    public static void main(String[] args) {

        /*teşekkürleeer erenincisağlık*/

    double asgariUcret , yeniAsgariUcret , maasArtis , yeniMaas , eskiMaas;

    Scanner imp = new Scanner(System.in);

        System.out.print("Asgari ücreti giriniz : ");
        asgariUcret = imp.nextInt();

        System.out.print("Yeni asgari ücreti giriniz : ");
        yeniAsgariUcret = imp.nextInt();


        maasArtis = yeniAsgariUcret / asgariUcret ;
        System.out.println("Asgari ücret artış oranı : " + maasArtis);

        System.out.print("Eski maaşınızı giriniz : " );
        eskiMaas = imp.nextInt();

        yeniMaas = maasArtis * eskiMaas ;
        System.out.println("Yeni dönem maaşınız : " + yeniMaas);

    }
}
