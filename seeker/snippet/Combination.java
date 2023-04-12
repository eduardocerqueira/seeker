//date: 2023-04-12T16:41:32Z
//url: https://api.github.com/gists/0edd086b691bf6bb0a629d86da700472
//owner: https://api.github.com/users/GG1RRka

import java.util.ArrayList;
import java.util.List;

public class Combination {

    private String[] valueArray = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"};
    private char[] suitArray = {'♣', '♦', '♥', '♠'};
    List<Card> player1 = new ArrayList<Card>(), player2 = new ArrayList<Card>(), common = new ArrayList<Card>();

    Combination(List<Card> player1, List<Card> player2, List<Card> common) {
        this.player1 = player1;
        this.player2 = player2;
        this.common = common;
    }

    private int whoCanGetCombination(List<Card> lst) {
        int cntPlayer1 = 0, cntPlayer2 = 0, cntCommon = 0;
        for (Card card : lst) {
            for (Card cardCommon : common) {
                if (card.print().equals(cardCommon.print())) {
                    cntCommon++;
                    break;
                }
            }
            for (Card cardPlayer1 : player1) {
                if (card.print().equals(cardPlayer1.print())) {
                    cntPlayer1++;
                    break;
                }
            }
            for (Card cardPlayer2 : player2) {
                if (card.print().equals(cardPlayer2.print())) {
                    cntPlayer2++;
                    break;
                }
            }
        }
        if (cntPlayer1 + cntCommon >= 5 && cntPlayer2 + cntCommon >= 5) return 3;
        else if (cntPlayer1 + cntCommon >= 5) return 1;
        else if (cntPlayer2 + cntCommon >= 5) return 2;
        return 0;
    }

    public String result(String combination, boolean player1win, boolean player2win) {
        if (player1win && player2win) return "Draw (both make " + combination + ").";
        else if (player1win) return "Player 1 won (makes " + combination + ").";
        else if (player2win) return "Player 2 won (makes " + combination + ").";
        return "Draw.";
    }

    public String checkStraightFlush() {
        boolean player1win = false, player2win = false;
        for (char suit12345 : suitArray) {
            for (int i = 0; i + 4 <= 13; i++) {
                List<Card> lst = new ArrayList<Card>();
                lst.add(new Card(new CardValue(valueArray[i]), new CardSuit(suit12345)));
                lst.add(new Card(new CardValue(valueArray[i + 1]), new CardSuit(suit12345)));
                lst.add(new Card(new CardValue(valueArray[i + 2]), new CardSuit(suit12345)));
                lst.add(new Card(new CardValue(valueArray[i + 3]), new CardSuit(suit12345)));
                lst.add(new Card(new CardValue(valueArray[i + 4]), new CardSuit(suit12345)));
                if (whoCanGetCombination(lst) == 1 || whoCanGetCombination(lst) == 3) player1win = true;
                if (whoCanGetCombination(lst) == 2 || whoCanGetCombination(lst) == 3) player2win = true;
            }
        }
        return result("straight flush", player1win, player2win);
    }

    public String checkKare() {
        boolean player1win = false, player2win = false;
        for (String value1234 : valueArray) {
            for (char suit5 : suitArray) {
                for (String value5 : valueArray) {
                    if (value1234.equals(value5)) continue;
                    List<Card> lst = new ArrayList<Card>();
                    lst.add(new Card(new CardValue(value1234), new CardSuit(suitArray[0])));
                    lst.add(new Card(new CardValue(value1234), new CardSuit(suitArray[1])));
                    lst.add(new Card(new CardValue(value1234), new CardSuit(suitArray[2])));
                    lst.add(new Card(new CardValue(value1234), new CardSuit(suitArray[3])));
                    lst.add(new Card(new CardValue(value5), new CardSuit(suit5)));
                    if (whoCanGetCombination(lst) == 1 || whoCanGetCombination(lst) == 3) player1win = true;
                    if (whoCanGetCombination(lst) == 2 || whoCanGetCombination(lst) == 3) player2win = true;
//                    for (Card card : lst) {
//                        System.out.print(card.print() + " ");
//                    }
//                    System.out.println();
                }
            }
        }
        return result("kare", player1win, player2win);
    }

    public String checkFullHouse() {
        boolean player1win = false, player2win = false;
        for (char suit1 : suitArray) {
            for (char suit2 : suitArray) {
                if (suit1 == suit2) continue;
                for (char suit3 : suitArray) {
                    if (suit1 == suit3 || suit2 == suit3) continue;
                    for (String value123 : valueArray) {
                        for (String value45 : valueArray) {
                            if (value123.equals(value45)) continue;
                            for (char suit4 : suitArray) {
                                for (char suit5 : suitArray) {
                                    if (suit4 == suit5) continue;
                                    List<Card> lst = new ArrayList<Card>();
                                    lst.add(new Card(new CardValue(value123), new CardSuit(suit1)));
                                    lst.add(new Card(new CardValue(value123), new CardSuit(suit2)));
                                    lst.add(new Card(new CardValue(value123), new CardSuit(suit3)));
                                    lst.add(new Card(new CardValue(value45), new CardSuit(suit4)));
                                    lst.add(new Card(new CardValue(value45), new CardSuit(suit5)));
                                    if (whoCanGetCombination(lst) == 1 || whoCanGetCombination(lst) == 3) player1win = true;
                                    if (whoCanGetCombination(lst) == 2 || whoCanGetCombination(lst) == 3) player2win = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        return result("full house", player1win, player2win);
    }

    public String checkFlush() {
        boolean player1win = false, player2win = false;
        for (char suit12345 : suitArray) {
            for (String value1 : valueArray) {
                for (String value2 : valueArray) {
                    if (value1.equals(value2)) continue;
                    for (String value3 : valueArray) {
                        if (value1.equals(value3) || value2.equals(value3)) continue;
                        for (String value4 : valueArray) {
                            if (value1.equals(value4) || value2.equals(value4) || value3.equals(value4)) continue;
                            for (String value5 : valueArray) {
                                if (value1.equals(value5) || value2.equals(value5) || value3.equals(value5) || value4.equals(value5)) continue;
                                List<Card> lst = new ArrayList<Card>();
                                lst.add(new Card(new CardValue(value1), new CardSuit(suit12345)));
                                lst.add(new Card(new CardValue(value2), new CardSuit(suit12345)));
                                lst.add(new Card(new CardValue(value3), new CardSuit(suit12345)));
                                lst.add(new Card(new CardValue(value4), new CardSuit(suit12345)));
                                lst.add(new Card(new CardValue(value5), new CardSuit(suit12345)));
                                if (whoCanGetCombination(lst) == 1 || whoCanGetCombination(lst) == 3) player1win = true;
                                if (whoCanGetCombination(lst) == 2 || whoCanGetCombination(lst) == 3) player2win = true;
                            }
                        }
                    }
                }
            }
        }
        return result("flush", player1win, player2win);
    }

    public String checkStraight() {
        boolean player1win = false, player2win = false;
        for (char suit1 : suitArray) {
            for (char suit2 : suitArray) {
                for (char suit3 : suitArray) {
                    for (char suit4 : suitArray) {
                        for (char suit5 : suitArray) {
                            for (int i = 0; i + 4 <= 13; i++) {
                                List<Card> lst = new ArrayList<Card>();
                                lst.add(new Card(new CardValue(valueArray[i]), new CardSuit(suit1)));
                                lst.add(new Card(new CardValue(valueArray[i + 1]), new CardSuit(suit2)));
                                lst.add(new Card(new CardValue(valueArray[i + 2]), new CardSuit(suit3)));
                                lst.add(new Card(new CardValue(valueArray[i + 3]), new CardSuit(suit4)));
                                lst.add(new Card(new CardValue(valueArray[i + 4]), new CardSuit(suit5)));
                                if (whoCanGetCombination(lst) == 1 || whoCanGetCombination(lst) == 3) player1win = true;
                                if (whoCanGetCombination(lst) == 2 || whoCanGetCombination(lst) == 3) player2win = true;
                            }
                        }
                    }
                }
            }
        }
        return result("straight", player1win, player2win);
    }

    public String checkTriplet() {
        boolean player1win = false, player2win = false;
        for (String value123 : valueArray) {
            for (String value4 : valueArray) {
                if (value123.equals(value4)) continue;
                for (String value5 : valueArray) {
                    if (value123.equals(value5) || value4.equals(value5)) continue;
                    for (char suit1 : suitArray) {
                        for (char suit2 : suitArray) {
                            for (char suit3 : suitArray) {
                                if (suit1 == suit2 || suit1 == suit3 || suit2 == suit3) continue;
                                for (char suit4 : suitArray) {
                                    for (char suit5 : suitArray) {
                                        List<Card> lst = new ArrayList<Card>();
                                        lst.add(new Card(new CardValue(value123), new CardSuit(suit1)));
                                        lst.add(new Card(new CardValue(value123), new CardSuit(suit2)));
                                        lst.add(new Card(new CardValue(value123), new CardSuit(suit3)));
                                        lst.add(new Card(new CardValue(value4), new CardSuit(suit4)));
                                        lst.add(new Card(new CardValue(value5), new CardSuit(suit5)));
                                        if (whoCanGetCombination(lst) == 1 || whoCanGetCombination(lst) == 3) player1win = true;
                                        if (whoCanGetCombination(lst) == 2 || whoCanGetCombination(lst) == 3) player2win = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return result("triplet", player1win, player2win);
    }

    public String checkTwoPair() {
        boolean player1win = false, player2win = false;
        for (String value12 : valueArray) {
            for (String value34 : valueArray) {
                if (value12.equals(value34)) continue;
                for (String value5 : valueArray) {
                    if (value12.equals(value5) || value34.equals(value5)) continue;
                    for (char suit1 : suitArray) {
                        for (char suit2 : suitArray) {
                            if (suit1 == suit2) continue;
                            for (char suit3 : suitArray) {
                                for (char suit4 : suitArray) {
                                    if (suit3 == suit4) continue;
                                    for (char suit5 : suitArray) {
                                        List<Card> lst = new ArrayList<Card>();
                                        lst.add(new Card(new CardValue(value12), new CardSuit(suit1)));
                                        lst.add(new Card(new CardValue(value12), new CardSuit(suit2)));
                                        lst.add(new Card(new CardValue(value34), new CardSuit(suit3)));
                                        lst.add(new Card(new CardValue(value34), new CardSuit(suit4)));
                                        lst.add(new Card(new CardValue(value5), new CardSuit(suit5)));
                                        if (whoCanGetCombination(lst) == 1 || whoCanGetCombination(lst) == 3) player1win = true;
                                        if (whoCanGetCombination(lst) == 2 || whoCanGetCombination(lst) == 3) player2win = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return result("two pair", player1win, player2win);
    }

    public String checkPair() {
        boolean player1win = false, player2win = false;
        for (String value12 : valueArray) {
            for (String value3 : valueArray) {
                if (value12.equals(value3)) continue;
                for (String value4 : valueArray) {
                    if (value12.equals(value4) || value3.equals(value4)) continue;
                    for (String value5 : valueArray) {
                        if (value3.equals(value5) || value4.equals(value5)) continue;
                        for (char suit1 : suitArray) {
                            for (char suit2 : suitArray) {
                                if (suit1 == suit2) continue;
                                for (char suit3 : suitArray) {
                                    for (char suit4 : suitArray) {
                                        for (char suit5 : suitArray) {
                                            List<Card> lst = new ArrayList<Card>();
                                            lst.add(new Card(new CardValue(value12), new CardSuit(suit1)));
                                            lst.add(new Card(new CardValue(value12), new CardSuit(suit2)));
                                            lst.add(new Card(new CardValue(value3), new CardSuit(suit3)));
                                            lst.add(new Card(new CardValue(value4), new CardSuit(suit4)));
                                            lst.add(new Card(new CardValue(value5), new CardSuit(suit5)));
                                            if (whoCanGetCombination(lst) == 1 || whoCanGetCombination(lst) == 3) player1win = true;
                                            if (whoCanGetCombination(lst) == 2 || whoCanGetCombination(lst) == 3) player2win = true;
//                                            for (Card card : lst) {
//                                                System.out.print(card.print() + " ");
//                                            }
//                                            System.out.println();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return result("pair", player1win, player2win);
    }

    public int compare(Card card1, Card card2) {
        if (card1.value.position() > card2.value.position()) return 1;
        if (card1.value.position() < card2.value.position()) return 2;
        return 0;
    }

    public String checkHighCard() {
        boolean player1win = false, player2win = false;
        Card highCard1 = new Card(new CardValue("2"), new CardSuit('♣'));
        Card highCard2 = new Card(new CardValue("2"), new CardSuit('♣'));
        if (compare(player1.get(0), player1.get(1)) == 1) highCard1 = player1.get(0);
        else highCard1 = player1.get(1);
        if (compare(player2.get(0), player2.get(1)) == 1) highCard2 = player2.get(0);
        else highCard2 = player2.get(1);
        if (compare(highCard1, highCard2) == 1) player1win = true;
        else if (compare(highCard1, highCard2) == 2) player2win = true;
        return result("high card", player1win, player2win);
    }
}