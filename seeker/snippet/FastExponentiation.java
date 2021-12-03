//date: 2021-12-03T17:14:53Z
//url: https://api.github.com/gists/8cc1aa8de38e7f4a11f2f445b33d7c29
//owner: https://api.github.com/users/centraladeprogramare

public long fastExp(long base, long n, long modulo) {
    // multiplication identity
    long result = 1L;
    // base^(2^0)
    long step = base;
    // while n still has bits
    while (n > 0) {
        // the least significant bit
        long lastBit = n % 2;
        // if the bit is 1, multiply the result with step = base^(2^i)
        if (lastBit == 1) {
            result = (result * step) % modulo;
        }
        // computes next step = base^(2^(i+1)) as base^(2^i) * base^(2^i)
        step = (step * step) % modulo;
        // removes least significant bit
        n /= 2L;
    }
    return result;
}