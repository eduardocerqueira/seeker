//date: 2022-10-11T17:13:16Z
//url: https://api.github.com/gists/303b8dba54d49675c2734a820493ef4f
//owner: https://api.github.com/users/WatipasoChirambo

func GetThought(thoughts []Thought, thoughtId int) string {
	for index, _ := range thoughts {
		if index+1 == thoughtId {
			return thoughts[index].title
		}
	}
	return "thought not available"
}