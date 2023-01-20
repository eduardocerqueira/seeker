//date: 2023-01-20T16:53:25Z
//url: https://api.github.com/gists/f3d2060d0bd13cd0ce2add70e6060ea0
//owner: https://api.github.com/users/Glavo

package org.glavo.jmh;

import com.google.common.jimfs.Jimfs;
import org.openjdk.jmh.annotations.*;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 10, time = 3)
@Measurement(iterations = 5, time = 5)
@Fork(value = 1, jvmArgsAppend = {"-XX:+UseG1GC", "-Xms8g", "-Xmx8g", "--add-opens=java.base/jdk.internal.access=ALL-UNNAMED"})
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
public class NoRepl {
    private static final Charset GBK = Charset.forName("GBK");

    @Param({"0", "1024", "8192", "1048576", "33554432", "268435456"})
    private int length;

    private FileSystem fs;
    private Path asciiFile;
    private Path utf8File;
    private Path gbkFile;


    @Setup
    public void setup() throws IOException {
        fs = Jimfs.newFileSystem();
        asciiFile = fs.getPath("ascii.txt");
        utf8File = fs.getPath("utf8.txt");
        gbkFile = fs.getPath("gbk.txt");

        Random random = new Random(0);
        try (OutputStream os = new BufferedOutputStream(Files.newOutputStream(asciiFile))) {
            for (int i = 0; i < length; i += 8) {
                os.write(random.nextInt(128));
            }
        }

        try (BufferedWriter utf8 = Files.newBufferedWriter(utf8File, StandardCharsets.UTF_8);
             BufferedWriter gbk = Files.newBufferedWriter(gbkFile, GBK)
        ) {
            for (int i = 0; i < length; i++) {
                char ch = (i % 1024) == 1023 ? (char) (random.nextInt(0x9fa5 - 0x4e00) + 0x4e00) : (char) random.nextInt(128);
                utf8.write(ch);
                gbk.write(ch);
            }
        }
    }

    @TearDown
    public void cleanup() throws IOException {
        fs.close();
    }

    @Benchmark
    public String testReadAscii() throws IOException {
        return Files.readString(asciiFile);
    }

    @Benchmark
    public String testReadUTF8() throws IOException {
        return Files.readString(utf8File);
    }

    @Benchmark
    public String testReadAsciiAsGBK() throws IOException {
        return Files.readString(asciiFile, GBK);
    }

    @Benchmark
    public String testReadGBK() throws IOException {
        return Files.readString(gbkFile, GBK);
    }
}
