//date: 2025-09-17T17:07:38Z
//url: https://api.github.com/gists/a4053e7602fef263ddc13d8155fe279f
//owner: https://api.github.com/users/harmandeepcentric

@Test
    void testIsPrime() {
        // Test prime numbers
        assertTrue(OrderService.isPrime(2), "2 should be prime");
        assertTrue(OrderService.isPrime(3), "3 should be prime");
        assertTrue(OrderService.isPrime(5), "5 should be prime");
        assertTrue(OrderService.isPrime(7), "7 should be prime");
        assertTrue(OrderService.isPrime(11), "11 should be prime");
        assertTrue(OrderService.isPrime(13), "13 should be prime");
        assertTrue(OrderService.isPrime(17), "17 should be prime");
        assertTrue(OrderService.isPrime(19), "19 should be prime");
        assertTrue(OrderService.isPrime(23), "23 should be prime");
        assertTrue(OrderService.isPrime(97), "97 should be prime");

        // Test non-prime numbers
        assertFalse(OrderService.isPrime(1), "1 should not be prime");
        assertFalse(OrderService.isPrime(0), "0 should not be prime");
        assertFalse(OrderService.isPrime(-1), "-1 should not be prime");
        assertFalse(OrderService.isPrime(4), "4 should not be prime");
        assertFalse(OrderService.isPrime(6), "6 should not be prime");
        assertFalse(OrderService.isPrime(8), "8 should not be prime");
        assertFalse(OrderService.isPrime(9), "9 should not be prime");
        assertFalse(OrderService.isPrime(10), "10 should not be prime");
        assertFalse(OrderService.isPrime(12), "12 should not be prime");
        assertFalse(OrderService.isPrime(15), "15 should not be prime");
        assertFalse(OrderService.isPrime(25), "25 should not be prime");
        assertFalse(OrderService.isPrime(100), "100 should not be prime");
    }