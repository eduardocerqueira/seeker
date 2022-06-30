//date: 2022-06-30T17:16:17Z
//url: https://api.github.com/gists/cb531b0ce794d1cfc773206eaa23011a
//owner: https://api.github.com/users/CodeLover254

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.Base64;

/**
 * Vending key generator class for the 160-bit vending key
 */
public class VendingKey160Generator {
    private static void generate160BitKey() throws IOException {
        //generate 20 random bytes. 160 bits=20 bytes
        SecureRandom secureRandom = new SecureRandom();
        byte[] vendingKey = new byte[20];
        secureRandom.nextBytes(vendingKey);

        //write the generated key to a file. Will use this later
        BufferedWriter writer = new BufferedWriter(new FileWriter("160BitVendingKey.key"));
        writer.write(Base64.getEncoder().encodeToString(vendingKey));
        writer.close();
    }

    /**
     * main method
     * @param args
     */
    public static void main(String[] args) {
        try {
            generate160BitKey();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
