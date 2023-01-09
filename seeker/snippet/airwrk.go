//date: 2023-01-09T17:07:22Z
//url: https://api.github.com/gists/3acb34911c2d9eb6863be09eaef5f0c7
//owner: https://api.github.com/users/diptomondal007

// Problem 1
func isPalindrome(num int) bool {
	s := strconv.FormatInt(int64(num), 10)
	if s[0] == s[len(s)-1] {
		return true
	}
	return false
}

// Problem 2
func singleNumber(arr []int) int {
	m := make(map[int]int)
	for i := range arr {
		m[arr[i]]++
	}

	for k := range m {
		if m[k] == 1 {
			return k
		}
	}
	return 0
}

// Problem 3
func goodStringLen(words []string, chars string) int {
	m := make(map[uint8]int)
	for i := range chars {
		m[chars[i]]++
	}

	var length int
	for i := range words {
		valid := true
		tempMap := make(map[uint8]int)
		for j := range words[i] {
			val := m[words[i][j]]
			//log.Println(string(words[i][j]), val)
			//cat
			if tempMap[words[i][j]] > val || val < 1 {
				//log.Println(string(words[i][j]), tempMap[words[i][j]])
				valid = false
				break
			}
			tempMap[words[i][j]]++
		}
		if valid {
			//log.Println(words[i])
			length += len(words[i])
		}
	}

	return length
}
