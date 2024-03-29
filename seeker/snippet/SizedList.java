//date: 2024-03-29T16:53:48Z
//url: https://api.github.com/gists/97988969f9f117466a4b5e744f73793f
//owner: https://api.github.com/users/homedirectory

/**
 * @param <T> the type of elements in the list
 */
public sealed interface SizedList<T, LENGTH extends Nat> {

    /**
     * The empty list.
     */
    final class Nil<T> implements SizedList<T, Zero> {
        private static final Nil<?> NIL = new Nil<>();

        private Nil() {}

        @Override
        public String toString() {
            return "()";
        }
    }

    @SuppressWarnings("unchecked")
    static <T> Nil<T> nil() {
        return (Nil<T>) Nil.NIL;
    }

    /**
     * Non-empty list.
     */
    record Cons<T, CDR_LEN extends Nat>
        (T car, SizedList<? extends T, CDR_LEN> cdr)
        implements SizedList<T, Succ<CDR_LEN>>
    {
        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder();

            sb.append('(');
            sb.append(car);

            SizedList<?, ?> tmp = this.cdr;
            boolean going = true;
            while (going) {
                switch (tmp) {
                    case Nil $ -> {
                        going = false;
                    }
                    case Cons(var car, var cdr) -> {
                        sb.append(' ');
                        sb.append(car);
                        tmp = cdr;
                    }
                }
            }

            sb.append(')');
            return sb.toString();
        }
    }

    static <T, N extends Nat> Cons<T, N> cons(T item, SizedList<? extends T, N> cons) {
        return new Cons<>(item, cons);
    }

    // --- Showtime ---

    /**
     * Returns a list of sums by adding elements of both lists pairwise.
     * This method's signature guarantees that both lists have the same length.
     */
    @SuppressWarnings("unchecked")
    static <N extends Nat> Cons<Integer, N> sums(Cons<Integer, N> cons1, Cons<Integer, N> cons2) {
        final var cdr = switch (cons1.cdr()) {
            case Nil $ -> $;
            case Cons cdr1 -> switch (cons2.cdr()) {
                case Cons cdr2 -> sums(cdr1, cdr2);
                case Nil $ -> $;
            };
        };
        return cons(cons1.car() + cons2.car(), cdr);
    }

    static Nil<Integer> sums(Nil<Integer> nil1, Nil<Integer> nil2) {
        return nil();
    }

    static void main(String[] args) throws Exception {
        Cons<Integer, Zero> list1 = cons(1, nil());
        System.out.println(list1);

        Cons<Integer, Zero> sums1 = sums(list1, list1);
        System.out.println(sums1);

        Nil<Integer> sums0 = sums(nil(), nil());
        System.out.println(sums0);

        Cons<Integer, Succ<Zero>> list2 = cons(1, cons(2, nil()));
        System.out.println(list2);

        // let's try summing up lists of different lengths
        // sums(list2, list1);
        //
        // compile-time ERROR: lengths differ
        // required: Cons<Integer,N>           Cons<Integer,N>
        // found:    Cons<Integer,Succ<Zero>>  Cons<Integer,Zero>

        // let's hack it by hiding the length!
        // sums((SizedList<Integer, ?>) list2, (SizedList<Integer, ?>) list1);
        //
        // compile-time ERROR: the wildcards capture different types, thus sums() can't be applied
        // required: Cons<Integer,N>           Cons<Integer,N>
        // found:    SizedList<Integer,CAP#1>  SizedList<Integer,CAP#2>
    }

}

/**
 * Natural number.
 */
sealed interface Nat {}

/**
 * Represents number zero.
 * Is never instantiated.
 */
final class Zero implements Nat {
    private Zero() {}
}

/**
 * Successor number type, i.e., n + 1.
 * Is never instantiated.
 */
final class Succ<N extends Nat> implements Nat {
    private Succ() {}
}
