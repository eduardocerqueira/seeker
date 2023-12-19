//date: 2023-12-19T16:59:05Z
//url: https://api.github.com/gists/b389dedfe3a0a36c8b5be52aae191e47
//owner: https://api.github.com/users/thimmwork

import ai.timefold.solver.core.api.function.TriFunction;
import ai.timefold.solver.core.api.score.stream.bi.BiConstraintCollector;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Collector that collects a min and a max value of comparables per key.
 * @param <KEY> key to group the values
 * @param <COMP> comparable of which to determine the min and the max
 */
public class MinMaxCollector<KEY, COMP extends Comparable<COMP>> {
    private final Map<KEY, Pair<COMP, COMP>> minMaxPerKey = new HashMap<>();

    public void collect(KEY key, COMP other) {
        var minMax = minMaxPerKey.get(key);
        var replace = false;
        if (minMax == null) {
            minMax = ImmutablePair.of(other, other);
            replace = true;
        } else if (minMax.getLeft().compareTo(other) > 0) {
            minMax = ImmutablePair.of(other, minMax.getRight());
            replace = true;
        }
        if (minMax.getRight().compareTo(other) < 0) {
            minMax = ImmutablePair.of(minMax.getLeft(), other);
            replace = true;
        }
        if (replace) {
            minMaxPerKey.put(key, minMax);
        }
    }

    public Map<KEY, Pair<COMP, COMP>> getMinMaxPerKey() {
        return minMaxPerKey;
    }

    /**
     * @return a Timefold BiConstraintCollector that uses the first fact as a key and collects min and max values of the second fact.
     * The second fact must implement Comparable
     * @param <KEY> key to group the values
     * @param <COMP> comparable of which to determine the min and the max
     */
    public static <KEY, COMP extends Comparable<COMP>> BiConstraintCollector<KEY, COMP, MinMaxCollector<KEY, COMP>, Map<KEY, Pair<COMP, COMP>>> collector() {
        return new BiConstraintCollector<>() {
            @Override
            public Supplier<MinMaxCollector<KEY, COMP>> supplier() {
                return MinMaxCollector::new;
            }

            @Override
            public TriFunction<MinMaxCollector<KEY, COMP>, KEY, COMP, Runnable> accumulator() {
                return (minMaxCollector, key, comp) -> () -> minMaxCollector.collect(key, comp);
            }

            @Override
            public Function<MinMaxCollector<KEY, COMP>, Map<KEY, Pair<COMP, COMP>>> finisher() {
                return MinMaxCollector::getMinMaxPerKey;
            }
        };
    }
}
