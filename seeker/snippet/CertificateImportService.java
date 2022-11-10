//date: 2022-11-10T17:05:09Z
//url: https://api.github.com/gists/c9265bc56b9ac09ca34785f0e270e3fc
//owner: https://api.github.com/users/tkalamdc

package com.digicert.util.service;

import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

@Component
public class CertificateImportService {

    private static final String PREFIX = "certificate";

    public String encodeToBase64(MultipartFile multipartFile, String suffix) throws IOException {
        String output;
        File file = toFile(multipartFile, suffix);
        try (InputStream fis = new FileInputStream(file)){
            byte[] data = new byte[(int) file.length()];
            fis.read(data);
            output = Base64.getEncoder().encodeToString(data);
            file.delete();
        }catch (IOException e){
            throw new CertificateImportException(e.getMessage());
        }
        return output;
    }


    public void decodeToFile(String encodedCertificate) throws IOException {
        byte[] decoded = Base64.getDecoder().decode(encodedCertificate.getBytes(StandardCharsets.US_ASCII));
        Path destination = Paths.get(".", PREFIX + ".p12");
        Files.write(destination, decoded);
    }

    private File toFile(MultipartFile multipartFile, String suffix) throws IOException {
        File tmp = File.createTempFile(PREFIX, suffix);
        try(OutputStream os = new FileOutputStream(tmp)) {
            os.write(multipartFile.getBytes());
        }
        return tmp;
    }
}