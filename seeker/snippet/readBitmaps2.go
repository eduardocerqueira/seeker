//date: 2021-12-29T17:09:44Z
//url: https://api.github.com/gists/04656cacae7725f87a835d8d987bb04b
//owner: https://api.github.com/users/nwillc

func readBitmaps(embedFs embed.FS, path string) Font {
	files, err := embedFs.ReadDir(path)
	if err != nil {
		panic(err)
	}
	toFontRuneKV := func(f fs.DirEntry) (rune, FontRune) {
		return toCharName(f.Name()), toFontRune(embedFs, path, f.Name())
	}
	return genfuncs.Associate(files, toFontRuneKV)
}