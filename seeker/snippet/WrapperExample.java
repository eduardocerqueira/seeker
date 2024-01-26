//date: 2024-01-26T16:47:39Z
//url: https://api.github.com/gists/3fd28e1b2ac42c9f0804627f7923e76a
//owner: https://api.github.com/users/delta-dev-software

public class WrapperExample {
    public static void main(String[] args) {
        int primitiveInt = 42;

        // Converting primitive int to Integer (autoboxing)
        Integer wrappedInt = Integer.valueOf(primitiveInt);

        // Converting Integer to primitive int (unboxing)
        int unwrappedInt = wrappedInt.intValue();

        System.out.println("Wrapped Integer: " + wrappedInt);
        System.out.println("Unwrapped int: " + unwrappedInt);
    }
}