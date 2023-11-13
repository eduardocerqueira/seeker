//date: 2023-11-13T16:49:50Z
//url: https://api.github.com/gists/7bc07f3708577a83dbf4b619a5867d5f
//owner: https://api.github.com/users/Nagelfar

import java.util.function.Consumer;
import java.util.function.Function;

sealed interface Result<T> {
    static <TOK> Result<TOK> ok(TOK value) {
        return new Ok<>(value);
    }

    static <TOK> Result<TOK> err(String error) {
        return new Err<>(error);
    }

    Result<T> flatMap(Function<T, Result<T>> mapper);

    <TM> Result<TM> map(Function<T, TM> mapper);

    Result<T> tee(Consumer<T> consumer);

    <TResult> TResult ifPresentOrElse(Function<T, TResult> okConsumer, Function<String, TResult> errorConsumer);

    record Ok<T>(T value) implements Result<T> {
        @Override
        public <TM> Result<TM> map(Function<T, TM> mapper) {
            return Result.ok(mapper.apply(value));
        }

        @Override
        public Result<T> tee(Consumer<T> consumer) {
            consumer.accept(value);
            return this;
        }

        @Override
        public <TResult> TResult ifPresentOrElse(Function<T, TResult> okConsumer, Function<String, TResult> errorConsumer) {
            return okConsumer.apply(value);
        }

        @Override
        public Result<T> flatMap(Function<T, Result<T>> mapper) {
            return mapper.apply(value);
        }
    }

    record Err<T>(String error) implements Result<T> {
        @Override
        public Result<T> flatMap(Function<T, Result<T>> mapper) {
            return this;
        }

        @Override
        public <TM> Result<TM> map(Function<T, TM> mapper) {
            return Result.err(error);
        }

        @Override
        public Result<T> tee(Consumer<T> consumer) {
            return this;
        }

        @Override
        public <TResult> TResult ifPresentOrElse(Function<T, TResult> okConsumer, Function<String, TResult> errorConsumer) {
            return errorConsumer.apply(error);
        }

    }
}



class Scratch {

    public static void main(String[] args) {
        var result = executeUsecase_a(new Request(1, "name", ""));
    }

    public static HttpResponse<?> executeUsecase_a(Request input) {
        return Validation.validateRequest(input)
                .map(SingleTrack::canonicalizeEmail)
                .tee(DeadEnd::updateDB)
                .flatMap(DeadEnd::sendEmail)
                .ifPresentOrElse(HttpResponse::ok, HttpResponse::badRequest);
    }

    public static HttpResponse<?> executeUsecase_b(Request input) {
        return Output.returnMessage(
                Validation.validateRequest(input)
                        .map(SingleTrack::canonicalizeEmail)
                        .tee(DeadEnd::updateDB)
                        .flatMap(DeadEnd::sendEmail)
        );
    }


    public static class Validation {
        public static Result<Request> validateInput(Request input) {
            if (input.name.isEmpty())
                return Result.err("Name must not be blank");
            else if (input.email.isEmpty())
                return Result.err("Email must not be blank");
            else
                return Result.ok(input);
        }

        public static Result<Request> nameNotBlank(Request input) {
            if (input.name.isEmpty())
                return Result.err("Name must not be blank");
            else
                return Result.ok(input);
        }

        public static Result<Request> name50(Request input) {
            if (input.name.isEmpty())
                return Result.err("Name must not be longer than 50 chars");
            else
                return Result.ok(input);
        }

        public static Result<Request> emailNotBlank(Request input) {
            if (input.email.isEmpty())
                return Result.err("Email must not be blank");
            else
                return Result.ok(input);
        }

        public static Result<Request> validateRequest(Request input) {
            return nameNotBlank(input)
                    .flatMap(Validation::name50)
                    .flatMap(Validation::emailNotBlank);
        }
    }

    public static class SingleTrack {
        public static Request canonicalizeEmail(Request input) {
            return new Request(
                    input.userId(),
                    input.name(),
                    input.email().trim().toLowerCase()
            );
        }
    }

    public static class DeadEnd {
        public static <T> void updateDB(T result) {
            // TODO persist into DB
        }

        public static Result<Request> sendEmail(Request request) {
            try {
                // TODO send email
                return Result.ok(request);
            } catch (Exception e) {
                return Result.err(e.toString());
            }
        }
    }

    public static class Output {
        public static <T> HttpResponse<?> returnMessage(Result<T> result) {
            return switch (result) {
                case Result.Ok(var body) -> HttpResponse.ok(body);
                case Result.Err(var error) -> HttpResponse.badRequest("An error occurred:" + error);
            };
        }
    }

    public record Request(
            int userId,
            String name,
            String email) {
    }
}

// imagine this is io.micronaut.http.HttpResponse
interface HttpResponse<T> {
    static <T> HttpResponse<T> ok(T body) {
        return new HttpResponse<T>() {
            @Override
            public int hashCode() {
                return ("ok" + body).hashCode();
            }
        };
    }

    static <T> HttpResponse<T> badRequest(T body) {
        return new HttpResponse<T>() {
            @Override
            public int hashCode() {
                return ("badRequest" + body).hashCode();
            }
        };
    }
}