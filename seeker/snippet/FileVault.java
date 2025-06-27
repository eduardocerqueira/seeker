//date: 2025-06-27T16:42:14Z
//url: https://api.github.com/gists/edf0bd51c7c5c8de25af163827d0d24b
//owner: https://api.github.com/users/MelkiZedekICT

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.io.*;
import java.util.Scanner;

public class FileVault {
    private static final String AES = "AES";

    public static void encrypt(File inputFile, File outputFile, SecretKeySpec key) throws Exception {
        Cipher cipher = Cipher.getInstance(AES);
        cipher.init(Cipher.ENCRYPT_MODE, key);
        byte[] inputBytes = new FileInputStream(inputFile).readAllBytes();
        byte[] outputBytes = cipher.doFinal(inputBytes);
        new FileOutputStream(outputFile).write(outputBytes);
    }

    public static void decrypt(File inputFile, File outputFile, SecretKeySpec key) throws Exception {
        Cipher cipher = Cipher.getInstance(AES);
        cipher.init(Cipher.DECRYPT_MODE, key);
        byte[] inputBytes = new FileInputStream(inputFile).readAllBytes();
        byte[] outputBytes = cipher.doFinal(inputBytes);
        new FileOutputStream(outputFile).write(outputBytes);
    }

    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(System.in);
        System.out.println("Secure File Vault\n1. Encrypt\n2. Decrypt\nEnter choice:");
        int choice = sc.nextInt(); sc.nextLine();

        System.out.print("Enter input file path: ");
        String inputPath = sc.nextLine();
        System.out.print("Enter output file path: ");
        String outputPath = sc.nextLine();
        System.out.print("Enter 16-char secret key: "**********"
        String keyStr = sc.nextLine();

        SecretKeySpec key = "**********"
        File inputFile = new File(inputPath);
        File outputFile = new File(outputPath);

        if (choice == 1) {
            encrypt(inputFile, outputFile, key);
            System.out.println("✅ File encrypted.");
        } else {
            decrypt(inputFile, outputFile, key);
            System.out.println("✅ File decrypted.");
        }
    }
}
);
        }
    }
}
