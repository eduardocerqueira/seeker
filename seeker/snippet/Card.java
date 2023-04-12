//date: 2023-04-12T16:41:32Z
//url: https://api.github.com/gists/0edd086b691bf6bb0a629d86da700472
//owner: https://api.github.com/users/GG1RRka

public class Card {
    public CardValue value;
    public CardSuit suit;
    public Card(CardValue value, CardSuit suit) {
        this.value = value;
        this.suit = suit;
    }
    public Card() {
        value = new CardValue();
        suit = new CardSuit();
    }

    public String print() {
        String s = value.get() + suit.get();
        return s;
    }
}