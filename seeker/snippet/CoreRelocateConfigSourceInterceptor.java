//date: 2023-12-21T16:57:48Z
//url: https://api.github.com/gists/04f1088c044b8db4b85fd42fa4c49179
//owner: https://api.github.com/users/dmlloyd

package io.quarkus.runtime.configuration;

import java.io.Serial;
import java.util.Iterator;
import java.util.Map;
import java.util.stream.Collectors;

import io.smallrye.config.ConfigSourceInterceptorContext;
import io.smallrye.config.ConfigValue;
import io.smallrye.config.RelocateConfigSourceInterceptor;

/**
 * Central location for all relocated core configuration entries.
 */
public final class CoreRelocateConfigSourceInterceptor extends RelocateConfigSourceInterceptor {
    @Serial
    private static final long serialVersionUID = -3096467506527701713L;

    private static final Map<String, String> MAPPINGS = Map.of(
            // relocated in #37445
            "quarkus.package.decompiler.enabled", "quarkus.package.vineflower.enabled",
            "quarkus.package.decompiler.jar-directory", "quarkus.package.vineflower.jar-directory");

    private static final Map<String, String> REV_MAPPINGS = MAPPINGS.entrySet().stream()
            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    public CoreRelocateConfigSourceInterceptor() {
        super(MAPPINGS);
    }

    public Iterator<String> iterateNames(final ConfigSourceInterceptorContext context) {
        Iterator<String> itr = super.iterateNames(context);
        return new Iterator<String>() {
            public boolean hasNext() {
                return itr.hasNext();
            }

            public String next() {
                String next = itr.next();
                return REV_MAPPINGS.getOrDefault(next, next);
            }
        };
    }

    public Iterator<ConfigValue> iterateValues(final ConfigSourceInterceptorContext context) {
        Iterator<ConfigValue> itr = super.iterateValues(context);
        return new Iterator<ConfigValue>() {
            public boolean hasNext() {
                return itr.hasNext();
            }

            public ConfigValue next() {
                ConfigValue next = itr.next();
                String name = next.getName();
                String mapped = REV_MAPPINGS.getOrDefault(name, name);
                return name.equals(mapped) ? next : next.withName(name);
            }
        };
    }
}
