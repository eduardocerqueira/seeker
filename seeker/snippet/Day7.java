//date: 2022-12-07T16:57:32Z
//url: https://api.github.com/gists/6d6a73831b4b99d55fc378882dd4bcc6
//owner: https://api.github.com/users/gamestoy

package day7;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;

public class Day7 {

  static class Terminal {

    private static final Long MAX_SPACE = 70000000L;

    File root;
    File curr;

    Terminal() {
      root = new File("/");
    }

    public void cd(String name) {
      switch (name) {
        case ".." -> curr = curr.getParent();
        case "/" -> curr = root;
        default -> curr = curr.getContent().stream().filter(c -> c.getName().equals(name))
            .filter(File::isDir).findFirst().orElseThrow();
      }
    }

    public void mkdir(String name) {
      curr.addContent(new File(name));
    }

    public void create(String name, int size) {
      curr.addContent(new File(name, size));
    }

    public int du(int size) {
      return curr.getSizeWithLimit(size);
    }

    public long rm(int spaceToAllocate) {
      return curr.findDirectoryToDelete(spaceToAllocate - (MAX_SPACE - curr.getSize()));
    }
  }


  static class File {

    private final String name;
    private final boolean isDir;
    private File parent;
    private final Set<File> content;
    private int size;

    public File(String name) {
      this.name = name;
      this.isDir = true;
      this.content = new HashSet<>();
      this.size = 0;
    }

    public File(String name, int size) {
      this.name = name;
      this.isDir = false;
      this.content = new HashSet<>();
      this.size = size;
    }

    public void addContent(File file) {
      file.setParent(this);
      content.add(file);
    }

    public String getName() {
      return name;
    }

    public boolean isDir() {
      return isDir;
    }

    public File getParent() {
      return parent;
    }

    public Set<File> getContent() {
      return content;
    }

    public int deleteDirectory(long space) {
      if (this.getSize() < space) {
        return Integer.MAX_VALUE;
      }
      var min = this.getContent().stream().filter(File::isDir)
          .map(f -> f.deleteDirectory(space))
          .filter(f -> space <= f).sorted().findFirst().orElse(this.getSize());
      return Math.min(min, this.getSize());
    }

    public int getSizeWithLimit(int limit) {
      return (this.getSize() <= limit ? this.getSize() : 0) + this.getContent().stream()
          .filter(File::isDir)
          .mapToInt(f -> f.getSizeWithLimit(limit))
          .sum();
    }

    public int getSize() {
      if (size == 0) {
        size = getContent().stream().map(File::getSize).mapToInt(Integer::intValue).sum();
      }
      return size;
    }

    public void setParent(File parent) {
      this.parent = parent;
    }
  }

  static class Parser {

    private final Path path;

    public Parser(Path path) {
      this.path = path;
    }

    public Terminal createTerminal() throws IOException {
      var terminal = new Terminal();
      Files.lines(path).forEach(l -> {
        switch (l.charAt(0)) {
          case '$' -> processOperation(terminal, l);
          default -> processFile(terminal, l);
        }
      });
      return terminal;
    }

    private void processOperation(Terminal terminal, String expression) {
      var tokens = "**********"
      if (tokens[1].equals("cd")) {
        terminal.cd(tokens[2]);
      }
    }

    private void processFile(Terminal terminal, String line) {
      var tokens = "**********"
      if (tokens[0].equals("dir")) {
        terminal.mkdir(tokens[1]);
      } else {
        terminal.create(tokens[1], Integer.parseInt(tokens[0]));
      }
    }
  }

  public static void main(String[] args) throws IOException {
    var parser = new Parser(Paths.get(args[0]));
    var terminal = parser.createTerminal();
    terminal.cd("/");
    var part1 = terminal.du(100000);
    var part2 = terminal.rm(30000000);
  }
}
00000);
  }
}
