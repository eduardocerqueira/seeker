//date: 2022-08-25T16:54:04Z
//url: https://api.github.com/gists/95cac885c3384e22a1da028ec3da19bb
//owner: https://api.github.com/users/PalisthaGit


import java.util.Arrays;

public class SortArrayOptimized {

	public int[] sortedSquares(int[] input) {

		// create an array of input size
		int output[] = new int[input.length];

		// assign the last index of output to index
		// to fill output from backwards
		int index = output.length - 1;

		// assign first index of input as left
		int left = 0;

		// assign last index of input as right
		int right = input.length - 1;

		while (left <= right) {

			// check if +ve value of right is greater than left
			if (Math.abs(input[right]) > Math.abs(input[left])) {
				// assign the square of right to the index of output
				output[index] = input[right] * input[right];

				right = right - 1;

				index = index - 1;

			}

			else {

				// assign the square of left to the index of output
				output[index] = input[left] * input[left];

				left = left + 1;
				index = index - 1;
			}
		}
		return output;
	}

	public static void main(String[] args) {
		SortArrayOptimized obj = new SortArrayOptimized();

		int array[] = { -4, -1, 0, 3, 10 };

		// sorted square array
		int sortedSquares[] = obj.sortedSquares(array);
		System.out.println(Arrays.toString(sortedSquares));

	}
}
