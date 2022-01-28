//date: 2022-01-28T17:10:43Z
//url: https://api.github.com/gists/5b4fd3803a609ed59164dba9f50699f4
//owner: https://api.github.com/users/IAFahim

func reverse(arr []byte, start, end int) {
	for i, j := start, end-1; i < end/2; i, j = i+1, j-1 {
		arr[i], arr[j] = arr[j], arr[i]
	}
}