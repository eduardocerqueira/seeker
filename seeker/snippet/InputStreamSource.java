//date: 2023-01-26T17:00:56Z
//url: https://api.github.com/gists/862c75eead3770ec1bf4e4d30983bc92
//owner: https://api.github.com/users/benravago

package lib.junit5.tools;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import org.junit.jupiter.params.provider.ArgumentsSource;

@Target({ ElementType.ANNOTATION_TYPE, ElementType.METHOD })
@Retention(RetentionPolicy.RUNTIME)

@ArgumentsSource(InputStreamArgumentsProvider.class)

public @interface InputStreamSource {
  String[] resources() default {};   // resource names 
  String[] files() default {};       // file paths
  String[] directories() default {}; // directory paths with optional glob in Path.fileName
}
