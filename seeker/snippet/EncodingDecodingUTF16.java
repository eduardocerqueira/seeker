//date: 2025-01-17T16:57:43Z
//url: https://api.github.com/gists/0e793bb94fbde07b20643ad9904ef8d9
//owner: https://api.github.com/users/nkchauhan003

package org.cb.encoding.textencoding;

import java.nio.charset.StandardCharsets;

public class EncodingDecodingUTF16 {
    public static void main(String[] args) {
        String text = "Hello, World!";
        System.out.println("Original text: " + text + ", length: " + text.length());

        // Encoding string to UTF-16 bytes
        byte[] utf16Bytes = text.getBytes(StandardCharsets.UTF_16);
        System.out.println("Encoded to UTF-16: ");
        printBytes(utf16Bytes);
        System.out.println();
        System.out.println("Length: " + utf16Bytes.length);

        // Decoding UTF-16 bytes back to string
        String decodedText = new String(utf16Bytes, StandardCharsets.UTF_16);
        System.out.println("Decoded from UTF-16: " + decodedText);
    }

    private static void printBytes(byte[] utf16Bytes) {
        for (byte b : utf16Bytes) {
            System.out.print(b + " ");
        }
    }
}