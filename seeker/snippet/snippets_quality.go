//date: 2022-10-18T17:19:13Z
//url: https://api.github.com/gists/25de742b9e79b37d4530692b8456c141
//owner: https://api.github.com/users/flexwang2

func norm(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "\"")
	s = strings.ReplaceAll(s, ",", "")
	s = strings.ReplaceAll(s, "'", "")
	s = strings.ReplaceAll(s, ".", "")
	s = strings.ReplaceAll(s, "!", "")
	s = strings.ReplaceAll(s, "?", "")
	return s
}

func getQuality(snippet string, snippetIndex *pb.SnippetIndex) (int32, int32) {
	words := strings.Split(snippet, " ")
	cnt := int32(0)
	mp := map[string]int{}
	for _, w := range words {
		w = norm(w)
		if w == "" {
			continue
		}
		mp[w]++
		cnt++
	}
	match := int32(0)
	for _, sec := range snippetIndex.DocSections {
		ws := strings.Split(sec.Text, " ")
		for _, w := range ws {
			w = norm(w)
			if w == "" {
				continue
			}
			if mp[w] > 0 {
				match++
				mp[w]--
			}
		}
	}
	return match, cnt
}