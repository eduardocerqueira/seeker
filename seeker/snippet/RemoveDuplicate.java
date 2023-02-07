//date: 2023-02-07T16:43:22Z
//url: https://api.github.com/gists/09c8a306dcd8861f1cb5bbf35cca0ccc
//owner: https://api.github.com/users/sadekujjaman

// PROBLEM 03
	public static void removeDuplicates(int[] arr){
		int n = arr.length;
		if(n < 2){
			return;
		}
		int i = 1;
		int j = 1;
		while(i < n){
			if(arr[i] != arr[i - 1]){
				arr[j++] = arr[i];
			}
			i++;
		}
	}