//date: 2022-07-15T17:21:09Z
//url: https://api.github.com/gists/8183ac4fb77ab63db68c27fd7723c8b1
//owner: https://api.github.com/users/tongcloudbeds

package com.cloudbeds.smartpolicy.service;
import io.vavr.Function5;
import io.vavr.Lazy;
import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.control.Try;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Arrays;
import java.util.Collections;
import java.util.function.BiFunction;
import java.util.function.Function;
import static io.vavr.API.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

public class VavrTest {
    final Logger logger = LoggerFactory.getLogger(VavrTest.class);

    @Test
    void giveInValidNumber_whenDivide_thenFail() {
        var result = Try.of(() -> 1 / 0);
        result.onFailure((ex) -> {
            logger.error("EventType=DivFailed Msg={}", ex.getMessage(), ex);
        });
        assertThat(result.isFailure()).isTrue();
    }

    @Test
    void giveValidNumber_whenDivide_thenSuccess() {
        var result = Try.of(() -> 4 / 2);
        result.onSuccess((r) -> {
            logger.info("EventType=DivSuccess Result={}", r);
            assertThat(r).isEqualTo(2);
        });
        assertThat(result.isSuccess()).isTrue();
    }

    // java8 only one or two parameter...
    @Test
    public void givenJava8BiFunction_whenWorks_thenCorrect() {
        Function<Integer, Integer> square = (num) -> num * num;
        int result = square.apply(2);

        assertEquals(4, result);

        BiFunction<Integer, Integer, Integer> sum = Integer::sum;
        int result2 = sum.apply(5, 7);

        assertEquals(12, result2);
    }

    // vavr enhancement
    @Test
    public void whenCreatesFunction_thenCorrect5() {
        Function5<String, String, String, String, String, String> concat =
                (a, b, c, d, e) -> a + b + c + d + e;
        String finalString = concat.apply(
                "Hello ", "world", "! ", "Learn ", "Vavr");

        assertEquals("Hello world! Learn Vavr", finalString);
    }


    @Test
    public void whenImmutableCollectionThrows_thenCorrect() {
        java.util.List<String> wordList = Arrays.asList("abracadabra");
        java.util.List<String> list = Collections.unmodifiableList(wordList);
        list.add("boom");
    }

    @Test
    public void givenFunction_whenEvaluatesWithLazy_thenCorrect() {
        Lazy<Double> lazy = Lazy.of(Math::random);
        assertFalse(lazy.isEvaluated());

        double val1 = lazy.get();
        assertTrue(lazy.isEvaluated());

        double val2 = lazy.get();
        assertEquals(val1, val2, 0.1);
    }

    @Test
    public void whenMatchworks_thenCorrect() {
        int input = 2;
        String output = Match(input).of(
                Case($(1), "one"),
                Case($(2), "two"),
                Case($(3), "three"),
                Case($(), "?"));

        assertEquals("two", output);
    }

    @Test
    public void java17_whenMatchworks_thenCorrect() {
        int input = 2;
        String output = switch (input) {
            case 1 -> "one";
            case 2 -> "two";
            case 3 -> "three";
            default -> "?";
        };
        assertEquals("two", output);
    }

    @Test
    public void whenCreatesTuple_thenCorrect1() {
        Tuple2<String, Integer> java8 = Tuple.of("Java", 8);
        String element1 = java8._1;
        int element2 = java8._2();

        assertEquals("Java", element1);
        assertEquals(8, element2);
    }
}
