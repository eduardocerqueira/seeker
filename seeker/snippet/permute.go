//date: 2021-12-08T17:04:28Z
//url: https://api.github.com/gists/70c0d38fb998364381802f49bbc2273e
//owner: https://api.github.com/users/kgaughan

package permute

func fact(n int) int {
	result := 1
	for i := 1; i <= n; i++ {
		result *= i
	}
	return result
}

func permute(chars string) []string {
	result := make([]string, 0, fact(len(chars)))
	generate(&result, []rune(chars), 0, len(chars)-1)
	return result
}

func generate(result *[]string, acc []rune, left, right int) {
	if left == right {
		*result = append(*result, string(acc))
	} else {
		for i := left; i <= right; i++ {
			acc[left], acc[i] = acc[i], acc[left]
			generate(result, acc, left+1, right)
			acc[left], acc[i] = acc[i], acc[left]
		}
	}
}
