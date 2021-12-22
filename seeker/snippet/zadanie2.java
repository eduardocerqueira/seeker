//date: 2021-12-22T17:20:35Z
//url: https://api.github.com/gists/0f5ad65f6f5d9646eb4ad2f85f151feb
//owner: https://api.github.com/users/mortdekai

package Crypto;

import java.security.NoSuchAlgorithmException;
import java.util.Scanner;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;

public class wolfi {
    private SecretKey secretKey;

    public wolfi() {
        try {
            KeyGenerator keyGenerator = KeyGenerator.getInstance("AES");
            keyGenerator.init(256);
            this.secretKey = keyGenerator.generateKey();
        } catch (NoSuchAlgorithmException var2) {
            var2.printStackTrace();
        }

    }

    public byte[] makeAes(byte[] rawMessage, int cipherMode) {
        try {
            Cipher cipher = Cipher.getInstance("AES");
            cipher.init(cipherMode, this.secretKey);
            byte[] output = cipher.doFinal(rawMessage);
            return output;
        } catch (Exception var5) {
            var5.printStackTrace();
            return null;
        }
    }
}

package Main;

        import Crypto.wolfi;
        import java.util.Scanner;

class Main {
    Main() {
    }

    public static void main(String[] args) {
        wolfi aes256 = new wolfi();
        Scanner e = new Scanner(System.in);
        String mes = e.nextLine();

        for(int i = 0; i < 1; ++i) {
            byte[] shifr = aes256.makeAes(mes.getBytes(), 1);
            System.out.println(new String(shifr));
            byte[] src = aes256.makeAes(shifr, 2);
            System.out.println(new String(src));
        }

    }
}