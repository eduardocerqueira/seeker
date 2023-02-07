//date: 2023-02-07T16:43:22Z
//url: https://api.github.com/gists/09c8a306dcd8861f1cb5bbf35cca0ccc
//owner: https://api.github.com/users/sadekujjaman

	// PROBLEM 01
	public static boolean isPalindrome(String str){
		if(str.length() <= 1){
			return true;
		}
		int n = str.length();
		
		for(int i = 0; i < n / 2; i++){
			if(str.charAt(i) != str.charAt(n - 1 - i)){
				return false;
			}
		}
		return true;
	}