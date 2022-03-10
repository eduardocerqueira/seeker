//date: 2022-03-10T17:04:37Z
//url: https://api.github.com/gists/3e47ccd1e3ac268459094621f4182c7c
//owner: https://api.github.com/users/TehleelMir

    static int findKthFibonacciByDp(int n, int[] arr) {
        if(n < 2) return n;
        if(arr[n] != 0) return arr[n];

        int result = findKthFibonacciByDp(n-1, arr) + findKthFibonacciByDp(n-2, arr);
        arr[n] = result;
        return result;
    }