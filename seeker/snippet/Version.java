//date: 2022-12-30T16:53:45Z
//url: https://api.github.com/gists/95685480f91970bf6ea2c6ece97c3459
//owner: https://api.github.com/users/williancorrea

public class Version implements Comparable<Version> {
    @NonNull
    public final int[] numbers;

    public Version(@NonNull String version) {
        final String split[] = version.split("\\-")[0].split("\\.");
        numbers = new int[split.length];
        for (int i = 0; i < split.length; i++) {
            numbers[i] = Integer.valueOf(split[i]);
        }
    }

    @Override
    public int compareTo(@NonNull Version another) {
        final int maxLength = Math.max(numbers.length, another.numbers.length);
        for (int i = 0; i < maxLength; i++) {
            final int left = i < numbers.length ? numbers[i] : 0;
            final int right = i < another.numbers.length ? another.numbers[i] : 0;
            if (left != right) {
                return left < right ? -1 : 1;
            }
        }
        return 0;
    }
}