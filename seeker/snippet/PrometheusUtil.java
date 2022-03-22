//date: 2022-03-22T16:55:10Z
//url: https://api.github.com/gists/a06c498272db9caff93793d9fdf02fe8
//owner: https://api.github.com/users/jadbaz

import io.prometheus.client.Collector;
import io.prometheus.client.CollectorRegistry;
import io.prometheus.client.Counter;
import io.prometheus.client.Gauge;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

public class PrometheusUtil {
    public static List<String> prometheusCollectorToText(CollectorRegistry registry) {
        final Enumeration<Collector.MetricFamilySamples> metricFamilySamplesEnumeration = registry.metricFamilySamples();

        final List<String> allSamplesTextLines = new ArrayList<>();
        while (metricFamilySamplesEnumeration.hasMoreElements()) {
            final List<String> samplesTextLines = metricFamilySamplesEnumeration.nextElement().samples.stream().map(sample -> {
                final StringBuilder sb = new StringBuilder();
                sb.append(sample.name);

                if (!sample.labelValues.isEmpty()) {
                    final Iterator<String> valueIterator = sample.labelValues.iterator();
                    final String labels = sample.labelNames.stream().map(labelName -> String.format("%s=\"%s\"", labelName, valueIterator.next())).collect(Collectors.joining(","));
                    sb.append("{").append(labels).append("}");
                }
                sb.append(" ");
                sb.append(sample.value);

                return sb.toString();
            }).collect(Collectors.toList());

            allSamplesTextLines.addAll(samplesTextLines);
        }
        return allSamplesTextLines;
    }

    public static List<String> prometheusCollectorToText() {
        return prometheusCollectorToText(CollectorRegistry.defaultRegistry);
    }
    public static String prometheusMetricsToText() {
        return String.join("\n", prometheusCollectorToText());
    }

    public static void main(String[] args) {
        final Counter requestsTotal = Counter.build()
                .name("requests_total").help("Total requests.")
                .labelNames("method", "status").register();

        final Gauge currentTemperature = Gauge.build()
                .name("temperature_celcius").help("The current temperature in Celcius")
                .register();

        requestsTotal.labels("POST", "200").inc();
        requestsTotal.labels("POST", "200").inc();
        requestsTotal.labels("POST", "200").inc();
        requestsTotal.labels("POST", "401").inc();
        requestsTotal.labels("POST", "404").inc();
        requestsTotal.labels("GET", "200").inc();
        currentTemperature.set(25);

        System.out.println(prometheusMetricsToText());
    }
}
