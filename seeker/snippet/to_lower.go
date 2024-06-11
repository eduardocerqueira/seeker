//date: 2024-06-11T16:44:51Z
//url: https://api.github.com/gists/2dede5a7a0ae593a1a0da14d10d5ebfb
//owner: https://api.github.com/users/mdigger

// FirstRuneToLower converts first rune to lower case if necessary.
func FirstRuneToLower(str string) string {
	if str == "" {
		return str
	}

	r, size := utf8.DecodeRuneInString(str)

	if !unicode.IsUpper(r) {
		return str
	}

	buf := &stringBuilder{}
	buf.WriteRune(unicode.ToLower(r))
	buf.WriteString(str[size:])
	return buf.String()
}
