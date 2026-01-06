//date: 2026-01-06T17:16:28Z
//url: https://api.github.com/gists/9a9bed0a8a272d9b8826d5c49f1708a4
//owner: https://api.github.com/users/kilink

package net.kilink.benchmark;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.random.RandomGenerator;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

@State(Scope.Benchmark)
public class GzipInputStreamBenchmark {

    final int INPUT_SIZE = 1024;
    final byte[] data;

    {
        var baos = new ByteArrayOutputStream();
        try (var gzos = new GZIPOutputStream(baos)) {
            var input = new byte[INPUT_SIZE];
            RandomGenerator.getDefault().nextBytes(input);
            gzos.write(input);
        } catch (IOException exc) {
            throw new UncheckedIOException(exc);
        }
        data = baos.toByteArray();
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    public byte[] testRead() throws IOException {
        try (InputStream is = new GZIPInputStream(new ByteArrayInputStream(data))) {
            return is.readAllBytes();
        }
    }
}
