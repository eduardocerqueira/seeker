//date: 2021-09-27T17:00:50Z
//url: https://api.github.com/gists/156a9e6453f5d3f8242689cfe6a6c07a
//owner: https://api.github.com/users/lmolkova

package org.example;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import io.opentelemetry.exporter.jaeger.JaegerGrpcSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.trace.ReadableSpan;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;
import org.reactivestreams.Subscriber;
import org.reactivestreams.Subscription;
import reactor.core.CoreSubscriber;
import reactor.core.Scannable;
import reactor.core.publisher.Hooks;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Operators;
import reactor.util.context.ContextView;
import reactor.util.retry.Retry;

import java.time.Duration;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiFunction;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;


public class Tests {

    // tracing instrumentation
    private <T> Mono<T> traceMono(Mono<T> publisher, String spanName) {
        // this hack forces 'publisher' to run in the onNext callback of `TracingSubscriber`
        // created for this publisher and with current() span that refer to span created here
        // without it it runs in the parent scope
        return Mono.just("1")
                .flatMap(i -> publisher)
                .doOnEach((signal) -> {
                    if (signal.isOnError()) {
                        recordException(signal.getContextView(), signal.getThrowable());
                        endSpan(StatusCode.ERROR, signal.getContextView());
                    } else if (signal.isOnComplete()){
                        endSpan(StatusCode.UNSET, signal.getContextView());
                    }
                })
                .contextWrite(ctx -> startSpan(ctx, spanName));
    }

    private reactor.util.context.Context startSpan(reactor.util.context.Context subscriberContext, String name) {
        io.opentelemetry.context.Context parent =
                subscriberContext.getOrDefault("otel-context-key", io.opentelemetry.context.Context.current());

        Span span = tracer.spanBuilder(name).setParent(parent).startSpan();

        System.out.printf("Starting span '%s', Parent span id %s, started id - %s\n",
                name,
                Span.fromContext(parent).getSpanContext().getSpanId(),
                span.getSpanContext().getSpanId());

        return subscriberContext.put("otel-context-key",  parent.with(span));
    }

    private void endSpan(StatusCode status, ContextView subscriberContext) {
        io.opentelemetry.context.Context current =
                subscriberContext.getOrDefault("otel-context-key", io.opentelemetry.context.Context.current());
        Span span = Span.fromContext(current);
        span.setStatus(status);
        span.end();
        System.out.printf("Ended span '%s', id %s\n", ((ReadableSpan)span).getName(), span.getSpanContext().getSpanId());
    }

    private void recordException(ContextView subscriberContext, Throwable t) {
        io.opentelemetry.context.Context current =
                subscriberContext.getOrDefault("otel-context-key", io.opentelemetry.context.Context.current());
        Span span = Span.fromContext(current);
        span.recordException(t);
    }
    // end of instrumentation
    
    private static Tracer tracer;
    private static TestExporter exporter;
    static SpanExporter jaeger = JaegerGrpcSpanExporter.builder()
            .setEndpoint("http://localhost:14250")
            .build();
    private static final Mono<String> dummy = Mono.just("1");

    @AfterAll
    public static void after() throws InterruptedException {
        Thread.sleep(1000);
        jaeger.shutdown();
    }
    @BeforeEach
    public void setup(TestInfo info) {
        TracingOperator2.registerOnEachOperator();

        exporter = new TestExporter();
        SdkTracerProvider otelProvider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .addSpanProcessor(SimpleSpanProcessor.create(jaeger)).build();

        tracer = OpenTelemetrySdk.builder().setTracerProvider(otelProvider).build().getTracer(info.getDisplayName());
    }


    Mono<Span> outer(Mono<Span> inner) {
        return inner.transform(n -> traceMono(n, "outer"));
    }

    private <T> T runTest(Mono<T> source) {
        Span parent = tracer.spanBuilder("parent").startSpan();

        T res = null;
        try (Scope s = parent.makeCurrent()) {
            res = source.block();
        }

        parent.end();

        return res;
    }

    @Test
    public void testJust() {
        Mono<Span> source =
                // oops, since it's just runs synchronously all the way, just is executed before
                // parent span even starts
                Mono
                    .just(Span.current()).transform(i -> traceMono(i, "innerJust"))
                    .transform(o -> traceMono(o, "outer"));

        Span innerCurrent = runTest(source);

        // !!!
        assertFalse(innerCurrent.getSpanContext().isValid());
        assertIsParent(exporter.exportedSpans.get("outer"), exporter.exportedSpans.get("innerJust"));
        assertIsParent(exporter.exportedSpans.get("parent"), exporter.exportedSpans.get("outer"));
    }

    @Test
    public void testDefer() {
        Mono<Span> source =
                // now it's deferred and happens in 'innerDefer' context
                Mono.defer(() -> Mono.just(Span.current())).transform(i -> traceMono(i, "innerDefer"))
                    .transform(o -> traceMono(o, "outer"));

        Span innerCurrent = runTest(source);

        assertSpanEquals(innerCurrent, exporter.exportedSpans.get("innerDefer"));
        assertIsParent(exporter.exportedSpans.get("outer"), exporter.exportedSpans.get("innerDefer"));
        assertIsParent(exporter.exportedSpans.get("parent"), exporter.exportedSpans.get("outer"));
    }

    @Test
    public void testCallable() {

        Mono<Span> source =
                // now it's callable and happens in 'innerCallable' context
                Mono.fromCallable(() -> Span.current()).transform(i -> traceMono(i, "innerCallable"))
                .transform(o -> traceMono(o, "outer"));

        Span innerCurrent = runTest(source);

        assertSpanEquals(innerCurrent, exporter.exportedSpans.get("innerCallable"));
        assertIsParent(exporter.exportedSpans.get("outer"), exporter.exportedSpans.get("innerCallable"));
        assertIsParent(exporter.exportedSpans.get("parent"), exporter.exportedSpans.get("outer"));
    }


    @Test
    public void testInnerNested() {

        Mono<Span> source = outer(Mono.defer( () -> {
            Span nested = tracer.spanBuilder("innerNested").startSpan();

            // DON'T do!
            // don't make spans current, use mono.transform(m -> traceMono()) instead
            try (Scope s = nested.makeCurrent()) {
                return Mono.delay(Duration.ofMillis(1)).map(l -> Span.current()).transform(d -> traceMono(d, "innerDelay"))
                        .contextWrite(ctx -> {
                            if (!Span.current().getSpanContext().getSpanId().equals(nested.getSpanContext().getSpanId())) {
                                // when it executes, current is gone already
                                System.out.println("'innerNested' is no longer current");
                            }
                            // or pass context
                            return ctx.put("otel-context-key", io.opentelemetry.context.Context.current().with(nested));
                        });
            } finally {
                nested.end();
            }
        }));

        Span innerCurrent = runTest(source);

        assertSpanEquals(innerCurrent, exporter.exportedSpans.get("innerDelay"));
        assertIsParent(exporter.exportedSpans.get("innerNested"), exporter.exportedSpans.get("innerDelay"));
        assertIsParent(exporter.exportedSpans.get("outer"), exporter.exportedSpans.get("innerNested"));
        assertIsParent(exporter.exportedSpans.get("parent"), exporter.exportedSpans.get("outer"));
    }

    @Test
    public void testDelay() {

        Mono<Span> source = outer(Mono
                .delay(Duration.ofMillis(1))
                .flatMap(l -> Mono.just(Span.current()))
                .transform(d -> traceMono(d, "innerDelay"))
        );

        Span innerCurrent = runTest(source);

        assertSpanEquals(innerCurrent, exporter.exportedSpans.get("innerDelay"));
        assertIsParent(exporter.exportedSpans.get("outer"), exporter.exportedSpans.get("innerDelay"));
        assertIsParent(exporter.exportedSpans.get("parent"), exporter.exportedSpans.get("outer"));
    }

    @Test
    public void testReuseMono() {

        Mono<Span> source = outer(Mono
                .defer(() -> Mono.just(Span.current()))
                .transform(j -> traceMono(j, "innerJust")));

        for (int i = 0; i < 5; i ++) {
            exporter.exportedSpans.clear();

            Span parent = tracer.spanBuilder("parent").startSpan();

            Span innerCurrent = runTest(source);

            assertSpanEquals(innerCurrent, exporter.exportedSpans.get("innerJust"));
            assertIsParent(exporter.exportedSpans.get("outer"), exporter.exportedSpans.get("innerJust"));
            assertIsParent(exporter.exportedSpans.get("parent"), exporter.exportedSpans.get("outer"));
        }
    }

    @Test
    public void testRetries() {

        AtomicReference<Integer> attempt = new AtomicReference<>(0);

        Mono<Span> source = outer(Mono.defer(() -> {
                            Span.current().updateName("inner" + attempt.get());
                            return (attempt.getAndUpdate(i -> i + 1) % 3 == 0)
                                    ? Mono.error(new TestException())
                                    : Mono.just(Span.current());
                        })
                        .transform(t -> traceMono(t, "inner"))
                        .retryWhen(Retry.fixedDelay(3, Duration.ofMillis(1)).filter(t -> t instanceof TestException))
                        .transform(t -> traceMono(t, "innerWithRetries")));

        Span innerCurrent = runTest(source);

        assertSpanEquals(innerCurrent, exporter.exportedSpans.get("inner1"));
        assertIsParent(exporter.exportedSpans.get("innerWithRetries"), exporter.exportedSpans.get("inner0"));
        assertIsParent(exporter.exportedSpans.get("innerWithRetries"), exporter.exportedSpans.get("inner1"));
        assertIsParent(exporter.exportedSpans.get("outer"), exporter.exportedSpans.get("innerWithRetries"));

        assertEquals(StatusCode.ERROR, exporter.exportedSpans.get("inner0").getStatus().getStatusCode());
        assertEquals(StatusCode.UNSET, exporter.exportedSpans.get("inner1").getStatus().getStatusCode());
        assertEquals(StatusCode.UNSET, exporter.exportedSpans.get("innerWithRetries").getStatus().getStatusCode());
        assertEquals(StatusCode.UNSET, exporter.exportedSpans.get("outer").getStatus().getStatusCode());
        assertEquals(StatusCode.UNSET, exporter.exportedSpans.get("parent").getStatus().getStatusCode());

    }

    @Test
    public void testZip() {

        Mono<Span> inner = Mono
                .delay(Duration.ofMillis(1)).transform(t -> traceMono(t, "delay"))
                .zipWith(Mono.just("1").transform(t -> traceMono(t, "just")))
                // NOTE: !! current is 'zip' span if Mono.defer or Mono.fromCallable is called.
                // If Mono.just(Span.current()) was used,  current would be the 'outer' span
                .flatMap(i -> Mono.fromCallable(() -> Span.current()).transform(t -> traceMono(t, "zip")));

        Mono<Span> source = outer(inner);

        Span parent = tracer.spanBuilder("parent").startSpan();

        Span innerCurrent = null;
        try (Scope s = parent.makeCurrent()) {
            try {
                innerCurrent = source.block();
            } catch (Exception ex) {
                ex.printStackTrace();
            }

            System.out.println("Inner: current span id " + innerCurrent);
        }

        parent.end();

        assertSpanEquals(innerCurrent, exporter.exportedSpans.get("zip"));
        assertIsParent(exporter.exportedSpans.get("outer"), exporter.exportedSpans.get("delay"));
        assertIsParent(exporter.exportedSpans.get("outer"), exporter.exportedSpans.get("just"));
        assertIsParent(exporter.exportedSpans.get("outer"), exporter.exportedSpans.get("zip"));
    }

    // TODO
    // map vs flatmap
    // then
    // concurrent

    private void assertSpanEquals(Span span, SpanData spanData) {
        assertSpanEquals(((ReadableSpan) span).toSpanData(), spanData);
    }

    private void assertSpanEquals(SpanData expected, SpanData actual) {
        assertEquals(expected.getName(), actual.getName());
        assertEquals(expected.getSpanId(), actual.getSpanId());
        assertEquals(expected.getTraceId(), actual.getTraceId());
        assertEquals(expected.getParentSpanId(), actual.getParentSpanId());
    }

    private void assertIsParent(SpanData parent, SpanData child) {
        assertEquals(parent.getTraceId(), child.getTraceId());
        assertEquals(parent.getSpanId(), child.getParentSpanId());
    }

    private class TestExporter implements SpanExporter {

        public Map<String, SpanData> exportedSpans = new ConcurrentHashMap<>();
        @Override
        public CompletableResultCode export(Collection<SpanData> collection) {
            for (SpanData sp : collection) {
                exportedSpans.put(sp.getName(), sp);
            }

            return CompletableResultCode.ofSuccess();
        }

        @Override
        public CompletableResultCode flush() {
            return CompletableResultCode.ofSuccess();
        }

        @Override
        public CompletableResultCode shutdown() {
            return CompletableResultCode.ofSuccess();
        }
    }

    class TestException extends Exception {
    }

}

class TracingOperator2 {

    public static void registerOnEachOperator() {
        Hooks.onEachOperator(TracingSubscriber.class.getName(), Operators.lift(new TracingOperator2.Lifter<>()));
    }

    public static class Lifter<T>
            implements BiFunction<Scannable, CoreSubscriber<? super T>, CoreSubscriber<? super T>> {

        @Override
        public CoreSubscriber<? super T> apply(Scannable publisher, CoreSubscriber<? super T> sub) {
            return new TracingOperator2.TracingSubscriber<>(sub, sub.currentContext());
        }
    }

    static class TracingSubscriber<T> implements CoreSubscriber<T> {
        private io.opentelemetry.context.Context traceContext;
        private final Subscriber<? super T> subscriber;
        private final reactor.util.context.Context context;

        public TracingSubscriber(Subscriber<? super T> subscriber, reactor.util.context.Context ctx) {
            this(subscriber, ctx, io.opentelemetry.context.Context.current());
        }

        public TracingSubscriber(
                Subscriber<? super T> subscriber,
                reactor.util.context.Context ctx,
                io.opentelemetry.context.Context contextToPropagate) {
            this.subscriber = subscriber;
            this.context = ctx;
            this.traceContext = context.getOrDefault("otel-context-key", contextToPropagate);
        }

        @Override
        public void onSubscribe(Subscription subscription) {

            withActiveSpan(() -> subscriber.onSubscribe(subscription), "onSubscribe");
        }

        @Override
        public void onNext(T o) {
            withActiveSpan(() -> subscriber.onNext(o), "onNext");
        }

        @Override
        public void onError(Throwable throwable) {
            withActiveSpan(() -> subscriber.onError(throwable), "onError");
        }

        @Override
        public void onComplete() {
            withActiveSpan(subscriber::onComplete, "onComplete");
        }

        @Override
        public reactor.util.context.Context currentContext() {
            return context;
        }

        private void withActiveSpan(Runnable runnable, String callback) {
            try (Scope ignored = traceContext.makeCurrent()) {
                /*System.out.printf("<-- Making span current '%s', current id - %s\n",
                        callback, Span.current().getSpanContext().getSpanId());*/
                runnable.run();
                //System.out.printf("  done %s-->\n", Span.current().getSpanContext().getSpanId());

            }
        }
    }
}

