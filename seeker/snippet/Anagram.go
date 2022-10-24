//date: 2022-10-24T17:32:15Z
//url: https://api.github.com/gists/4de4a002d28b65837972de68ffbc7da4
//owner: https://api.github.com/users/MCNGN

func groupAnagrams(word []string) [][]string {
	//Buat array 2d terlebih dahulu
	result := [][]string{}
	//Variabel untuk menampung map
	tmp := map[[26]int][]string{}
	for _, s := range word {
		//Variabel  array untuk jumlah huruf yg muncul
		chars := [26]int{}
		for _, c := range s {
			//Byte akan diubah menjadi string dengan mengurangi nilai decimal 'a' 97
			//Sehingga contoh string cook dalam array akan menjadi [0 0 1 0 0 0 0 0 0 0 1 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0]
			chars[c-97]++
		}
		tmp[chars] = append(tmp[chars], s)
		//Chars dengan key yg sama akan mempunyai value array string
	}

	//Perulangan untuk mengambil nilai value
	for _, v := range tmp {
		result = append(result, v)
	}

	return result
}