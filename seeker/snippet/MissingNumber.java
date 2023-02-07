//date: 2023-02-07T16:43:22Z
//url: https://api.github.com/gists/09c8a306dcd8861f1cb5bbf35cca0ccc
//owner: https://api.github.com/users/sadekujjaman

// PROBLEM 02
	public static int missingNumber(int arr[]){
		int n = arr.length;
		boolean[] visited = new boolean[n + 1];
		for(int i = 0; i < n; i++){
			visited[arr[i]] = true;
		}
		for(int i = 0; i <= n; i++){
			if(!visited[i]){
				return i;
			}
		}
		return 0;
	}