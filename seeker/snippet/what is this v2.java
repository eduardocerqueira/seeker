//date: 2025-03-11T17:04:25Z
//url: https://api.github.com/gists/70defbd0d775ddd090a76c317e42f566
//owner: https://api.github.com/users/RainVaporeon

public class SubClass {

  public static void main(String... args) {
    class Main {
      static int add(int a, int b) { return a + b; }
      int i = 65535;
      
      public Main() {}
      public Main(int i) { this.i = i; }
      
      class Nested {
        private final int i;
        public Nested(int i) { this.i = i; }
        
        @Override
        public int hashCode() {
          return Nested.this.i + Main.this.i;
        }
      }
    }

    interface $interface__0<T> {
      T apply(T t, Object o);
    }

    var n = new Main().add(new $interface__0<Integer>() {
      @Override public Integer apply(Integer t, Object o) { return t + System.identityHashCode(o); }
    }.apply(15, null), 13);

    var w = Main.add(new SubClass().hashCode(), new Main(16).new Nested(32).hashCode());

    System.out.printf("n = %d, w = %d", n, w);
  }

  @Override
  public final int hashCode() { return 100; }
}