//date: 2024-07-12T16:48:43Z
//url: https://api.github.com/gists/30efee98745cf1a8dde390307da4d6a6
//owner: https://api.github.com/users/avii-7

// Max sum in sub-arrays ( 2 ) 

// Problem Link: https://www.geeksforgeeks.org/problems/max-sum-in-sub-arrays0824/0

// Brute Force Approch

// TC -> O(n * n)
// SC -> O(1)

// Thought Process
// 1. I will generate all the sub-arrays.
// 2. Whenever a new element is inserted into the range, check with smallest and second smallest number.
// 3. After adjusting new number, if sum is greater than maxSum then maxSum will be replaced with sum.

public static long pairWithMaxSum(long arr[], long N)
{
    long maxSum = 0;

    for (int i = 0; i < N; i++) {

        long s = Long.MAX_VALUE, ss = Long.MAX_VALUE;

        for (int j = i; j < N; j++) {

            if(arr[j] < s) {
                ss = s;
                s = arr[j];
            }
            else if (arr[j] < ss) {
                ss = arr[j];
            }

            long tSum = s + ss;

            if (tSum > maxSum) {
                maxSum = tSum;
            }
        }
    }

    return maxSum;
}

// Optimal Approch

// TC-> O(n)
// SC-> O(1)

// Observation
// 1. I found that smallest and second smallest number with in range i...j (where i < j) 
// with maximum sum is always contigous (next to one another).

// Thought Process
// 1. According to observation, we only need to find contigous elements whose sum is greater than others contigous elements.

public static long pairWithMaxSum(long arr[], long N)
{
    long maxSum = 0;

    for (int i = 0; i <= N - 2; i++) {
        long tSum = arr[i] + arr[i + 1];
        if(tSum > maxSum) {
            maxSum = tSum;
        }
    }

    return maxSum;
}