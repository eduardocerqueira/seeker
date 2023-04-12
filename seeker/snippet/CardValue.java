//date: 2023-04-12T16:41:32Z
//url: https://api.github.com/gists/0edd086b691bf6bb0a629d86da700472
//owner: https://api.github.com/users/GG1RRka

import java.util.concurrent.ThreadLocalRandom;
public class CardValue {
    private String value;
    public CardValue(String value) {
        this.value = value;
    }
    private String[] valueArray = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"};
    public CardValue() {
        this.value = generateValue();
    }
    public String generateValue() {
        return valueArray[ThreadLocalRandom.current().nextInt(0, 13)];
    }
    public String get() {
        return value;
    }
    public int position() {
        for (int i = 0; i <= 12; i++) {
            if (valueArray[i].equals(value)) return i;
        }
        return -1;
    }
}