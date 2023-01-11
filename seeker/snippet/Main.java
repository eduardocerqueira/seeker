//date: 2023-01-11T17:00:57Z
//url: https://api.github.com/gists/2005e5c014e7fb860ed58555c6abbabc
//owner: https://api.github.com/users/danieljaatma

import java.util.Scanner;

class For Loop {
    public static void main(String[] args) {

        /* Loo programm, kus kasutajal on võimalus sisestada kaks täisnumbrit.
        Nende numbrite vahele jääb vahemik. Numbrid ise kaasa arvatud.
        Programm peaks printima vahemiku, kuid iga viie  ja kolmega jagunevate
        arvude asemel kirjutama 'FizzBuzz', vaid kolmega jagunevatele 'Fizz' ja
        vaid viiega jagunevatele 'Buzz'.
        Kõik numbrid tuleks kirjutada eraldi reale.*/


        Scanner scanner = new Scanner(System.in);
        int beginningOfInterval = scanner.nextInt();
        int endOfInterval = scanner.nextInt();
        /*Võtame kaks täisarvu skänneriga vastu. Nimetame need võimalikult
        loogiliselt tegevuse järgi*/

        for (int i = beginningOfInterval;i <=endOfInterval; i++) {
            if (i % 3 == 0 && i % 5 == 0) {
                System.out.println("FizzBuzz");
            } else if (i % 3 == 0) {
                System.out.println("Fizz");
            } else if (i % 5 == 0) {
                System.out.println("Buzz");
            } else {
                System.out.println(i);
            }
        }
        /* Kasutame for loopi.
        1. Sätestame esimese algnumbri
        2. Sätestame, et kui loop kehtib kuni lõppnumber on algnumbrist
        väiksem või võrdne. Võrdne, sest lõppnumber peab olema vahemikus esindatud.
        3. Liigutame algnumbrit iga loopiga ühe võrra kõrgemale.
        4. Esimesena katame ära kolme ja viiega jagamise ühiskattuvused,
        sest programm töötab kronoloogiliselt. Juhul, kui mõlemad tingimused
        on täidetud, prindi välja "FizzBuzz"
        5. Nendest arvudest ülejäänud viie ja kolmega jagamine ei ole ole otseselt
        oluline. Kui kolmega jagamise jääk võrdub 0 prindi "Fizz"
        6. Või kui viiega jagamise jääk võrdub 0 prindi "Buzz"
        7. Kõik ülejäänud juhud prindi algnumber, mis iga kordusega üles liigub.*/

            }
        }