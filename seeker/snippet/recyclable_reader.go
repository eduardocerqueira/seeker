//date: 2022-11-21T17:05:06Z
//url: https://api.github.com/gists/419e4db42393adb5eb21d51c246b11c8
//owner: https://api.github.com/users/hlubek

type recyclableReader struct {
	originalReader io.Reader
	teeReader      io.Reader
	buffer         *bytes.Buffer
}

func newRecyclableReader(file io.Reader) *recyclableReader {
	buffer := new(bytes.Buffer)
	teeReader := io.TeeReader(file, buffer)

	return &recyclableReader{
		originalReader: file,
		teeReader:      teeReader,
		buffer:         buffer,
	}
}

func (s *recyclableReader) Read(p []byte) (n int, err error) {
	return s.teeReader.Read(p)
}

// recycle the reader to be read again with buffered data and unread data of the original reader
func (s *recyclableReader) recycle() {
	prevBuffer := s.buffer
	prevReader := s.originalReader
	s.originalReader = io.MultiReader(prevBuffer, prevReader)
	s.buffer = new(bytes.Buffer)
	s.teeReader = io.TeeReader(s.originalReader, s.buffer)
}

// unwrap returns an unbuffered reader (cannot be recycled again)
func (s *recyclableReader) unwrap() io.Reader {
	return s.originalReader
}