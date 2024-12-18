//date: 2024-12-18T17:05:20Z
//url: https://api.github.com/gists/62129fe81f8db007463b670ba5d97a5e
//owner: https://api.github.com/users/suvincent

package com.example.demo;

import java.util.function.Function;
import java.util.function.Predicate;
import io.vavr.control.Either;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;

public class RuleSet<T, R> {
    private final Function<T, Either<List<Exception>, T>> validator;
    private final Class<R> clazz;

    public RuleSet(Function<T, Either<List<Exception>, T>> validator, Class<R> clazz) {
        this.validator = validator;
        this.clazz = clazz;
    }

    // Static factory method to convert single Exception validator
    public static <T, R> RuleSet<T, R> create(
        Function<T, Either<Exception, T>> validator, 
        Class<R> clazz
    ) {
        return new RuleSet<>(
            input -> validator.apply(input)
                .mapLeft(exception -> List.of(exception)),
            clazz
        );
    }

    public RuleSet<T, R> or(RuleSet<T, R> other) {
        return new RuleSet<>(
            input -> {
                Either<List<Exception>, R> firstResult = this.validate(input);
                if (firstResult.isRight()) {
                    return Either.right(input);
                }
                
                Either<List<Exception>, R> secondResult = other.validate(input);
                if (secondResult.isRight()) {
                    return Either.right(input);
                }
                
                // Combine exceptions if both validations fail
                List<Exception> combinedExceptions = new ArrayList<>();
                combinedExceptions.addAll(firstResult.getLeft());
                combinedExceptions.addAll(secondResult.getLeft());
                return Either.left(combinedExceptions); // Or handle multiple exceptions as needed
            },
            this.clazz
        );
    }

    public Either<List<Exception>, R> validate(T input) {
        Either<List<Exception>, T> validationResult = validator.apply(input);
        
        if (validationResult.isLeft()) {
            return Either.left((validationResult.getLeft()));
        }
        
        try {
            Constructor<R> constructor = findSuitableConstructor(input);
            R instance = constructor.newInstance(input);
            return Either.right(instance);
        } catch (Exception e) {
            return Either.left(List.of(new Exception("Could not instantiate class", e)));
        }
    }

    private Constructor<R> findSuitableConstructor(T input) throws NoSuchMethodException {
        try {
            // Try finding a constructor that takes the exact input type
            return clazz.getConstructor(input.getClass());
        } catch (NoSuchMethodException e) {
            // If not found, try finding a constructor that takes the input type's superclass
            return clazz.getConstructor(input.getClass().getSuperclass());
        }
    }
}
