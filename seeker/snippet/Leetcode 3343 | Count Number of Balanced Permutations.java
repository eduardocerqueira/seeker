//date: 2025-05-09T16:53:23Z
//url: https://api.github.com/gists/22697ef991b5916d07623ad4cfad9525
//owner: https://api.github.com/users/Tan12d

class Solution {
    final int M = 1000000007;
    int n;
	int totalDigitSum;
	long totalPermutationPossible = 0;

	// Binary Exponentiation
	private int findPower(long a, long b) {
		if (b == 0) {
			return 1;
		}

		long half = findPower(a, b / 2);
		long result = (half * half) % M;

		if (b % 2 == 1) {
			result = (result * a) % M;
		}

		return (int) result;
	}

	private int solve(int digit, int evenIndexDigitCount, int currSum, int freq[], long fermatFact[],
			int t[][][]) {
		if (digit == 10) {
			if (currSum == totalDigitSum / 2 && evenIndexDigitCount == (n + 1) / 2) {
				return (int) totalPermutationPossible;
			}

			return 0;
		}

		if (t[digit][evenIndexDigitCount][currSum] != -1) {
			return t[digit][evenIndexDigitCount][currSum];
		}

		long ways = 0;

		for (int count = 0; count <= Math.min(freq[digit], (n + 1) / 2 - evenIndexDigitCount); count++) {
			int evenPosCount = count;
			int oddPosCount = freq[digit] - count;

			long div = (fermatFact[evenPosCount] * fermatFact[oddPosCount]) % M;

			long val = solve(digit + 1, evenIndexDigitCount + evenPosCount, currSum + digit * count, freq, fermatFact,
					t);

			ways = (ways + (val * div) % M) % M;
		}

		return t[digit][evenIndexDigitCount][currSum] = (int) ways;
	}

	public int countBalancedPermutations(String num) {
		n = num.length();
		totalDigitSum = 0;

		int[] freq = new int[10];
		for (int i = 0; i < n; i++) {
			totalDigitSum += num.charAt(i) - '0';
			freq[num.charAt(i) - '0']++;
		}

		if (totalDigitSum % 2 != 0) {
			return 0;
		}

		// Precomputing factorial
		long[] fact = new long[n + 1];
		fact[0] = 1;
		for (int i = 1; i <= n; i++) {
			fact[i] = (fact[i - 1] * i) % M;
		}

		// Precomputing Fermat factorial (inverse factorial)
		long[] fermatFact = new long[n + 1];
		for (int i = 0; i <= n; i++) {
			fermatFact[i] = findPower(fact[i], M - 2);
		}

		totalPermutationPossible = (fact[(n + 1) / 2] * fact[n / 2]) % M;

		int[][][] t = new int[10][(n + 1) / 2 + 1][totalDigitSum + 1];
		for (int[][] arr2D : t)
			for (int[] arr1D : arr2D)
				Arrays.fill(arr1D, -1);

		return solve(0, 0, 0, freq, fermatFact, t);
	}
}