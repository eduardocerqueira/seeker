//date: 2023-04-12T16:41:32Z
//url: https://api.github.com/gists/0edd086b691bf6bb0a629d86da700472
//owner: https://api.github.com/users/GG1RRka

import java.sql.SQLOutput;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.println("Добро пожаловать в покер!");
        System.out.println("Покер (англ. poker) — карточная игра, цель которой собрать выигрышную комбинацию или вынудить всех соперников прекратить участвовать в игре. Игра идёт с полностью или частично закрытыми картами. Конкретные правила могут варьироваться в зависимости от разновидности покера. Обобщающими элементами всех разновидностей покера являются комбинации и наличие торговли в процессе игры. Ввиду того, что игрок не знает карты своих противников, покер является игрой с неполной информацией, как и многие другие карточные игры, в отличие от, например, шахмат, в которых оба игрока видят положение всех фигур на доске. ");
        System.out.println("Вы хотите начать игру? Напишите yes или no:");
        String s = input.next();
        s = s.toLowerCase();
        if (s.equals("no")) {
            System.out.println("До свидания!");
            return;
        }
        do {
            System.out.println("Началась новая игра!");
            Game game = new Game();
            System.out.println(game.play());
            System.out.println("Хотите ли вы продолжить игру? Напишите yes или no:");
            s = input.next();
            s = s.toLowerCase();
            if (s.equals("no")) {
                System.out.println("До свидания!");
                return;
            }
        } while (true);
    }
}