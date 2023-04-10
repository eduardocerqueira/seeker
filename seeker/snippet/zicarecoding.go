//date: 2023-04-10T17:04:00Z
//url: https://api.github.com/gists/cda5a49855a5d586d0fc2fd0a63a2a5d
//owner: https://api.github.com/users/neverbeenthisweeb

package zicarecoding

// Question:
//
// Find a number sequence inside an array of numbers
// Int [] main = new int[] {20, 7, 8, 10, 2, 5, 6} // non repeating numbers
// Int [] seq= new int [] {1,4}
// sequenceExists(main, [7,8]) ⇒ true
// sequenceExists(main, [8, 7]) ⇒ false
// sequenceExists(main, [7, 10]) ⇒ false
func HasSequence(target, seq []int) bool {
	// TODO: Confirm to interviewer what if given seq is empty
	// I assume for empty seq we always return true
	if len(seq) == 0 {
		return true
	}

	for i := 0; i < len(target); i++ {
		if target[i] == seq[0] {
			// Exit early if remaining elements in target is not enough to match with seq
			if len(target)-i < len(seq) {
				return false
			}

			isMatched := true

			for j := 0; j < len(seq); j++ {
				if seq[j] != target[i+j] {
					isMatched = false
					break
				}
			}

			if isMatched {
				return true
			}
		}
	}

	return false
}
