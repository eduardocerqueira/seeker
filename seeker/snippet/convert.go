//date: 2021-08-31T03:13:53Z
//url: https://api.github.com/gists/5420caa98498c46a84ce94cd9655287a
//owner: https://api.github.com/users/chadleeshaw

func ByteCounter(b int64) string {
	const unit = 1000
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB",
		float64(b)/float64(div), "kMGTPE"[exp])
}