//date: 2021-12-29T17:06:40Z
//url: https://api.github.com/gists/da04c592a9dc3d0888d8856972a8fe57
//owner: https://api.github.com/users/nwillc

func readBitmaps(fs embed.FS, path string) (Font, error) {
	runes := make(map[rune]FontRune)
	files, err := fs.ReadDir(path)
	if err != nil {
		return nil, err
	}
	for _, file := range files {
		r, err := toFontRune(fs, path, file.Name())
		if err != nil {
			return nil, err
		}
		name, err := toCharName(file.Name())
		if err != nil {
			return nil, err
		}
		runes[name] = r
	}
	return runes, nil
}