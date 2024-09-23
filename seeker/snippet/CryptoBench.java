//date: 2024-09-23T17:03:12Z
//url: https://api.github.com/gists/e80e4eda48b839d41d9cd52d0ba551ee
//owner: https://api.github.com/users/ehrmann

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import javax.crypto.Cipher;
import javax.crypto.CipherInputStream;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

public class CryptoBench {

  private static List<File> createFiles(File dir) throws IOException {
    List<File> result = new ArrayList<>();
    for (int i = 1024; i < 8 * 1024 * 1024; i *= 2) {
      File file = new File(dir, i + ".data");
      try (OutputStream out = new FileOutputStream(file)) {
        byte[] data = new byte[i];
        ThreadLocalRandom.current().nextBytes(data);
        out.write(data);
      }
      result.add(file);
    }
    return result;
  }

  private static Cipher getCipher() throws Exception {
    byte[] key = new byte[16];
    ThreadLocalRandom.current().nextBytes(key);
    Cipher c = Cipher.getInstance("AES/CBC/NoPadding");
    c.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, "AES"), new IvParameterSpec(new byte[16]));
    return c;
  }

  private static long decryptOld(File file) throws Exception {
    File dest = Files.createTempFile("tempfiles", ".tmp").toFile();
    try {
      long start = System.nanoTime();
      try (BufferedInputStream inputStream = new BufferedInputStream(new FileInputStream(file));
          InputStream decryptStream = new CipherInputStream(inputStream, getCipher());
          FileOutputStream fileOutputStream = new FileOutputStream(dest);
          BufferedOutputStream outputStream = new BufferedOutputStream(fileOutputStream)) {
        decryptStream.transferTo(outputStream);
      }
      return System.nanoTime() - start;
    } finally {
      dest.delete();
    }
  }

  private static long decryptNew(File file) throws Exception {
    var dest = Files.createTempFile("tempfiles", ".tmp").toFile();
    try {
      long start = System.nanoTime();
      try (var inChannel = Files.newByteChannel(file.toPath(), StandardOpenOption.READ);
           var outChannel = Files.newByteChannel(file.toPath(), StandardOpenOption.WRITE, StandardOpenOption.CREATE)) {
        ByteBuffer inBuffer = ByteBuffer.allocate(16384);
        ByteBuffer outBuffer = ByteBuffer.allocate(16384);
        var cipher = getCipher();
        while (inChannel.read(inBuffer) > 0) {
          inBuffer.flip();
          cipher.update(inBuffer, outBuffer);
          outBuffer.flip();
          outChannel.write(outBuffer);
          inBuffer.clear();
          outBuffer.clear();
        }
      }
      return System.nanoTime() - start;
    } finally {
      dest.delete();
    }
  }

  public static void main(String[] args) throws Exception {
    var dir = Files.createTempDirectory("tempdir").toFile();
    try {
      List<File> testFiles = createFiles(dir);
      var stats = new TreeMap<Long, Map<Boolean, long[]>>();

      for (var attempt = 0; attempt < 40000; ++attempt) {
        if (attempt % 1000 == 0) {
          System.out.printf("%d%n", attempt);
        }
        var old = ThreadLocalRandom.current().nextBoolean();
        var file = testFiles.get(ThreadLocalRandom.current().nextInt(testFiles.size()));
        var time = old ? decryptOld(file) : decryptNew(file);
        stats
            .computeIfAbsent(file.length(), k -> new HashMap<>())
            .merge(old, new long[] {time, 1}, (a, b) -> new long[] {a[0] + b[0], a[1] + b[1]});
      }

      System.out.printf("%5s,%10s,%10s%n", "size", "old", "new");
      for (var entry : stats.entrySet()) {
        var old = entry.getValue().getOrDefault(true, new long[] {-1, 1});
        var n = entry.getValue().getOrDefault(false, new long[] {-1, 1});
        System.out.printf("%4dK,%10.0f,%10.0f%n", entry.getKey() / 1024, old[0] / old[1] / 1000.0, n[0] / n[1] / 1000.0);
      }
    } finally {
      try (var dirStream = Files.walk(dir.toPath())) {
        dirStream.map(Path::toFile).sorted(Comparator.reverseOrder()).forEach(File::delete);
      }
    }
  }
}
