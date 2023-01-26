//date: 2023-01-26T17:00:56Z
//url: https://api.github.com/gists/862c75eead3770ec1bf4e4d30983bc92
//owner: https://api.github.com/users/benravago

package lib.junit5.tools;

import static java.util.stream.Collectors.toList;

import java.io.IOException;
import java.io.InputStream;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

import java.util.function.Function;
import java.util.function.Supplier;

import java.util.regex.Pattern;
import java.util.stream.Stream;

import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.params.support.AnnotationConsumer;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import static org.junit.jupiter.params.provider.Arguments.arguments;

import org.junit.platform.commons.JUnitException;
import org.junit.platform.commons.util.Preconditions;

class InputStreamArgumentsProvider implements AnnotationConsumer<InputStreamSource>, ArgumentsProvider {

  private List<Input> sources;

  record Input( Supplier<String> name, Function<ExtensionContext,InputStream> input ) {}
  
  @Override // for AnnotationConsumer
  public void accept(InputStreamSource annotation) {
    var resources = Arrays.stream(annotation.resources()).map(this::resource);
    var files = Arrays.stream(annotation.files()).map(Paths::get).map(this::file);
    var paths = Arrays.stream(annotation.directories()).flatMap(this::directory).map(this::file);
    sources = Stream.concat(resources, Stream.concat(files, paths)).collect(toList());
  }

  @Override // ArgumentsProvider
  public Stream<? extends Arguments> provideArguments(ExtensionContext context) {
    return Preconditions.notEmpty(this.sources, "sources must not be empty")
      .stream().map( source -> arguments( source.name.get(), source.input.apply(context)) );
  }
  
  Input resource(String path) {
    return new Input( () -> path, context -> open(path,context.getRequiredTestClass()) );
  }
  
  InputStream open(String path, Class<?> testClass) {
    Preconditions.notBlank(path, () -> "resource name must not be null or blank");
    var inputStream = testClass.getResourceAsStream(path);
    return Preconditions.notNull(inputStream, () -> "resource not found at "+path);
  }
  
  Input file(Path path) {
    return new Input( ()-> path.toString(), context -> open(path) );
  }
  
  InputStream open(Path path) {
    Preconditions.notBlank(path.toString(), () -> "file path must not be null or blank");
    try { return Files.newInputStream(path); }
    catch (IOException e) { throw new JUnitException("file at" + path + " could not be read", e); }
  }
  
  static final Pattern globChars = Pattern.compile("[*?{}]");
  
  Stream<Path> directory(String path) {
    var dir = Paths.get(path);
    var file = dir.getFileName().toString();
    try (
      var ds = globChars.matcher(file).find()
        ? Files.newDirectoryStream(dir.getParent(), file)
        : Files.newDirectoryStream(dir);
    ) {
      var list = new ArrayList<Path>();
      ds.forEach(list::add);
      return list.stream();
    }
    catch (IOException e) {
      throw new JUnitException("directory at" + path + " could not be read", e);
    }
  }

}
