//date: 2025-09-17T17:04:14Z
//url: https://api.github.com/gists/daa4f678a879ce0f4db74ae317524f76
//owner: https://api.github.com/users/harmandeepcentric

/**
     * Checks if a given number is prime.
     * A prime number is a natural number greater than 1 that has no positive
     * divisors
     * other than 1 and itself.
     * 
     * @param number the number to check
     * @return true if the number is prime, false otherwise
     */
    public static boolean isPrime(int number) {
        // Numbers less than or equal to 1 are not prime
        if (number <= 1) {
            return false;
        }

        // 2 is the only even prime number
        if (number == 2) {
            return true;
        }

        // Even numbers greater than 2 are not prime
        if (number % 2 == 0) {
            return false;
        }

        // Check for odd divisors up to the square root of the number
        // Only need to check up to sqrt(number) because if number has a divisor
        // greater than sqrt(number), it must also have a corresponding divisor
        // less than sqrt(number)
        for (int i = 3; i * i <= number; i += 2) {
            if (number % i == 0) {
                return false;
            }
        }

        return true;
    }