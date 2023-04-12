//date: 2023-04-12T16:41:32Z
//url: https://api.github.com/gists/0edd086b691bf6bb0a629d86da700472
//owner: https://api.github.com/users/GG1RRka

import java.util.ArrayList;
import java.util.List;
public class Game {
    List<Card> usedCards = new ArrayList<Card>();
    List<Card> player1 = new ArrayList<Card>(), player2 = new ArrayList<Card>(), common = new ArrayList<Card>();
    public Card newCard() {
        Card card = new Card();
        while (true) {
            boolean ok = true;
            for (Card curCard : usedCards) {
                if (card.print().equals(curCard.print())) {
                    ok = false;
                    break;
                }
            }
            if (ok) break;
            card = new Card();
        }
        usedCards.add(card);
        return card;
    }

    public Game() {
        init();
    }
    private void init() {
        //Random
        player1.add(newCard());
        player1.add(newCard());
        player2.add(newCard());
        player2.add(newCard());
        common.add(newCard());
        common.add(newCard());
        common.add(newCard());
        common.add(newCard());
        common.add(newCard());

        //Straight flush example
        /*player1.add(new Card(new CardValue("7"), new CardSuit('♣')));
        player1.add(new Card(new CardValue("6"), new CardSuit('♥')));
        player2.add(new Card(new CardValue("5"), new CardSuit('♠')));
        player2.add(new Card(new CardValue("6"), new CardSuit('♣')));
        common.add(new Card(new CardValue("K"), new CardSuit('♣')));
        common.add(new Card(new CardValue("8"), new CardSuit('♣')));
        common.add(new Card(new CardValue("9"), new CardSuit('♣')));
        common.add(new Card(new CardValue("10"), new CardSuit('♣')));
        common.add(new Card(new CardValue("J"), new CardSuit('♣')));*/

        //Kare example
        /*player1.add(new Card(new CardValue("7"), new CardSuit('♣')));
        player1.add(new Card(new CardValue("4"), new CardSuit('♥')));
        player2.add(new Card(new CardValue("2"), new CardSuit('♣')));
        player2.add(new Card(new CardValue("6"), new CardSuit('♠')));
        common.add(new Card(new CardValue("6"), new CardSuit('♦')));
        common.add(new Card(new CardValue("6"), new CardSuit('♣')));
        common.add(new Card(new CardValue("6"), new CardSuit('♥')));
        common.add(new Card(new CardValue("J"), new CardSuit('♣')));
        common.add(new Card(new CardValue("Q"), new CardSuit('♣')));*/

        //Full house example
        /*player1.add(new Card(new CardValue("7"), new CardSuit('♣')));
        player1.add(new Card(new CardValue("7"), new CardSuit('♥')));
        player2.add(new Card(new CardValue("2"), new CardSuit('♣')));
        player2.add(new Card(new CardValue("10"), new CardSuit('♠')));
        common.add(new Card(new CardValue("7"), new CardSuit('♠')));
        common.add(new Card(new CardValue("6"), new CardSuit('♣')));
        common.add(new Card(new CardValue("6"), new CardSuit('♥')));
        common.add(new Card(new CardValue("K"), new CardSuit('♣')));
        common.add(new Card(new CardValue("A"), new CardSuit('♣')));*/

        System.out.print("Player 1: ");
        for (Card card : player1) {
            System.out.print(card.print() + " ");
        }
        System.out.println();
        System.out.print("Player 2: ");
        for (Card card : player2) {
            System.out.print(card.print() + " ");
        }
        System.out.println();
        System.out.print("Common: ");
        for (Card card : common) {
            System.out.print(card.print() + " ");
        }
        System.out.println();
    }

    public String play() {
        Combination c = new Combination(player1, player2, common);
        String result;

//        System.out.println("proverka1");
        result = c.checkStraightFlush();
        if (!result.equals("Draw.")) return result;

//        System.out.println("proverka2");
        result = c.checkKare();
        if (!result.equals("Draw.")) return result;

//        System.out.println("proverka3");
        result = c.checkFullHouse();
        if (!result.equals("Draw.")) return result;

//        System.out.println("proverka4");
        result = c.checkFlush();
        if (!result.equals("Draw.")) return result;

//        System.out.println("proverka5");
        result = c.checkStraight();
        if (!result.equals("Draw.")) return result;

//        System.out.println("proverka6");
        result = c.checkTriplet();
        if (!result.equals("Draw.")) return result;

//        System.out.println("proverka7");
        result = c.checkTwoPair();
        if (!result.equals("Draw.")) return result;

//        System.out.println("proverka8");
        result = c.checkPair();
        if (!result.equals("Draw.")) return result;

//        System.out.println("proverka9");
        result = c.checkHighCard();
        if (!result.equals("Draw.")) return result;

        return "Draw (no combinations).";
    }
}