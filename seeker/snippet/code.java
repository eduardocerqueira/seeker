//date: 2025-01-30T16:39:43Z
//url: https://api.github.com/gists/08d1b7614d1511f376907700befbb10a
//owner: https://api.github.com/users/GordPavel

package test;

import static java.util.concurrent.TimeUnit.NANOSECONDS;
import static org.openjdk.jmh.annotations.Mode.AverageTime;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

@BenchmarkMode(AverageTime)
@Warmup(iterations = 2, time = 10)
@Measurement(iterations = 2, time = 30)
@Threads(6)
@Fork(value = 2)
@OutputTimeUnit(NANOSECONDS)
public class TestStringBuilder {

    private static final int ITERATIONS_COUNT = 1000;

    private static void testStringConcatenation(Blackhole blackhole) {
        String result = "";
        for (int i = 0; i < ITERATIONS_COUNT; i++) {
            result += i;
        }
        blackhole.consume(result);
    }

    private static void testStringBuilderConcatenation(Blackhole blackhole) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < ITERATIONS_COUNT; i++) {
            result.append(i);
        }
        blackhole.consume(result.toString());
    }

    @Benchmark
    @Fork(
        jvm = ".../Library/Java/JavaVirtualMachines/openjdk-21/Contents/Home/bin/java"
    )
    public void testStringConcatenationOpenJdk21(Blackhole blackhole) {
        testStringConcatenation(blackhole);
    }

    @Benchmark
    @Fork(
        jvm = ".../Library/Java/JavaVirtualMachines/openjdk-21/Contents/Home/bin/java"
    )
    public void testStringBuilderConcatenationOpenJdk21(Blackhole blackhole) {
        testStringBuilderConcatenation(blackhole);
    }

    @Benchmark
    @Fork(
        jvm = "/Library/Java/JavaVirtualMachines/graalvm-21.jdk/Contents/Home/bin/java"
    )
    public void testStringConcatenationGraalVm21(Blackhole blackhole) {
        testStringConcatenation(blackhole);
    }

    @Benchmark
    @Fork(
        jvm = "/Library/Java/JavaVirtualMachines/graalvm-21.jdk/Contents/Home/bin/java"
    )
    public void testStringBuilderConcatenationGraalVm21(Blackhole blackhole) {
        testStringBuilderConcatenation(blackhole);
    }
}
