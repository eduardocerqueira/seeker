//date: 2022-08-25T16:49:22Z
//url: https://api.github.com/gists/553d5c1a7268349bc95ffa6c7143ac07
//owner: https://api.github.com/users/PalisthaGit


import java.util.Arrays;

public class SortArrayBruteForce {

	public int[] sortedSquares(int[] input) {

		// compute square
		for (int i = 0; i < input.length; i++) {
			input[i] = input[i] * input[i];
		}
		// sort array
		Arrays.sort(input);
		return input;
	}

	public static void main(String[] args) {
		SortArrayBruteForce obj = new SortArrayBruteForce();

		int array[] = { -4, -1, 0, 3, 10 };

		// sorted square array
		int sortedSquares[] = obj.sortedSquares(array);
		System.out.println(Arrays.toString(sortedSquares));

	}
}
