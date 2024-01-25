//date: 2024-01-25T16:41:34Z
//url: https://api.github.com/gists/038f997ad501000c8bf46a5a6b159716
//owner: https://api.github.com/users/Nagelfar

package at.salzburgag.iot.dataplatform.smartmeter.utils.tracing;

import io.micronaut.aop.InterceptedMethod;
import io.micronaut.aop.InterceptorBean;
import io.micronaut.aop.MethodInterceptor;
import io.micronaut.aop.MethodInvocationContext;
import io.micronaut.core.annotation.Nullable;
import io.micronaut.core.convert.ConversionService;
import io.micronaut.core.propagation.PropagatedContext;
import io.micronaut.tracing.opentelemetry.OpenTelemetryPropagationContext;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;
import io.opentelemetry.instrumentation.api.instrumenter.Instrumenter;
import io.opentelemetry.instrumentation.api.instrumenter.util.ClassAndMethod;
import jakarta.inject.Named;
import jakarta.inject.Singleton;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;

import java.util.Collection;
import java.util.Optional;
import java.util.concurrent.CompletionStage;
import java.util.function.Consumer;

import static net.logstash.logback.argument.StructuredArguments.keyValue;

/**
 * Creates a new Trace-Span for methods of the type and tags them as SQL-Spans.
 * In addition to tagging, the result(s) are captured as an event.
 */
@Slf4j
@Singleton
@InterceptorBean(SqlSpan.class)
public class SqlSpanInterceptor implements MethodInterceptor<Object, Object> {
    private final Instrumenter<ClassAndMethod, Object> instrumenter;
    private final ConversionService conversionService;

    public SqlSpanInterceptor(
            @Named("micronautCodeTelemetryInstrumenter") Instrumenter<ClassAndMethod, Object> instrumenter,
            ConversionService conversionService) {
        this.instrumenter = instrumenter;
        this.conversionService = conversionService;
    }

    private static Object recordResult(Span span, Object result) {
        switch (result) {
            case Collection<?> collection -> {
                span.addEvent(
                        "Received several rows",
                        Attributes.of(AttributeKey.longKey("rowCount"), (long) collection.size())
                );
                log.info("Received result with {} rows", keyValue("rowCount", collection.size()));
            }
            case Optional<?> optional when optional.isEmpty() -> {
                span.addEvent("Received an empty optional result");
                log.info("Received an empty optional result");
            }
            case Optional<?> optional -> recordResult(span, optional.get());
            case null -> {
                span.addEvent("Received null/void result");
                log.info("Received null/void result");
            }
            case Object other -> {
                span.addEvent("Received single result");
                log.info("Received single result");
            }
            // TODO handle Stream, Page/Slice, Future, Publisher,...
        }
        return result;
    }

    @Override
    public @Nullable Object intercept(MethodInvocationContext<Object, Object> context) {
        // most of the code was copied from NewSpanOpenTelemetryTraceInterceptor
        var classAndMethod = ClassAndMethod.create(context.getDeclaringType(), context.getMethodName());

        InterceptedMethod interceptedMethod = InterceptedMethod.of(context, conversionService);

        Context currentContext = Context.current();
        if (!instrumenter.shouldStart(currentContext, classAndMethod)) {
            return context.proceed();
        }

        final Context newContext = instrumenter.start(currentContext, classAndMethod);

        log.info(
                "Starting SQL {} {}",
                keyValue("class", classAndMethod.declaringClass()),
                keyValue("method", classAndMethod.methodName())
        );
        Consumer<Throwable> exitWithError = throwable ->
        {
            log.info("Existing with error", throwable);
            instrumenter.end(
                    newContext,
                    classAndMethod,
                    null,
                    throwable
            );
        };
        Runnable exitWithoutResult = () -> {
            log.info("Successfully executed query without result");
            instrumenter.end(newContext, classAndMethod, null, null);
        };
        try (PropagatedContext.Scope ignore = PropagatedContext.getOrEmpty()
                .plus(new OpenTelemetryPropagationContext(newContext))
                .propagate()) {

            Span span = Span.current();
            span.setAttribute("isSQL", "true");

            Consumer<Object> exitWithResult = result -> {
                log.info("Successfully executed query with result");
                instrumenter.end(
                        newContext,
                        classAndMethod,
                        recordResult(span, result),
                        null
                );
            };
            switch (interceptedMethod.resultType()) {
                case PUBLISHER -> {
                    return interceptedMethod.handleResult(
                            Flux.from(interceptedMethod.interceptResultAsPublisher())
                                    .doOnNext(exitWithResult)
                                    .doOnComplete(exitWithoutResult)
                                    .doOnError(exitWithError)
                    );
                }
                case COMPLETION_STAGE -> {
                    CompletionStage<?> completionStage = interceptedMethod.interceptResultAsCompletionStage();
                    if (completionStage != null) {
                        completionStage = completionStage.whenComplete((o, throwable) -> {
                            if (throwable != null) {
                                exitWithError.accept(throwable);
                            } else {
                                exitWithResult.accept(o);
                            }
                        });
                    }
                    return interceptedMethod.handleResult(completionStage);
                }
                case SYNCHRONOUS -> {
                    Object response = context.proceed();
                    exitWithResult.accept(response);
                    return response;
                }
                default -> {
                    return interceptedMethod.unsupported();
                }
            }
        } catch (Exception e) {
            exitWithError.accept(e);
            return interceptedMethod.handleException(e);
        }
    }
}