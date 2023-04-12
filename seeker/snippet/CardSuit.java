//date: 2023-04-12T16:41:32Z
//url: https://api.github.com/gists/0edd086b691bf6bb0a629d86da700472
//owner: https://api.github.com/users/GG1RRka

import java.util.concurrent.ThreadLocalRandom;
public class CardSuit {
    private char suit;
    public CardSuit(char suit) {
        this.suit = suit;
    }
    private char[] suitArray = {'♣', '♦', '♥', '♠'};
    public CardSuit() {
        this.suit = generateSuit();
    }
    public char generateSuit() {
        return suitArray[ThreadLocalRandom.current().nextInt(0, 4)];
    }
    public char get() {
        return suit;
    }
}