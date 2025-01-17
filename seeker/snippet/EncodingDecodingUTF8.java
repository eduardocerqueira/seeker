//date: 2025-01-17T16:58:20Z
//url: https://api.github.com/gists/266d73f81bfda2f3688674806c1d4028
//owner: https://api.github.com/users/nkchauhan003

package org.cb.encoding.textencoding;

import java.nio.charset.StandardCharsets;

public class EncodingDecodingUTF8 {
    public static void main(String[] args) {
        String text = "Hello, World!";
        System.out.println("Original text: " + text + ", length: " + text.length());

        // Encoding string to UTF-8 bytes
        byte[] utf8Bytes = text.getBytes(StandardCharsets.UTF_8);
        System.out.println("Encoded to UTF-8: ");
        printBytes(utf8Bytes);
        System.out.println();
        System.out.println("Length: " + utf8Bytes.length);

        // Decoding UTF-8 bytes back to string
        String decodedText = new String(utf8Bytes, StandardCharsets.UTF_8);
        System.out.println("Decoded from UTF-8: " + decodedText);
    }

    private static void printBytes(byte[] utf8Bytes) {
        for (byte b : utf8Bytes) {
            System.out.print(b + " ");
        }
    }
}