//date: 2025-04-23T16:39:46Z
//url: https://api.github.com/gists/5bf4aa9f2d0f6f6ac7cf8acb689b5701
//owner: https://api.github.com/users/swtch1

// ObjectStreamReader "streams" s3 objects by using incremental byte range requests for each Read() call. This is
// meant for large files in order to optimize memory consumption rather than download speed. Note there is no
// requirement to specify an explicit "chunk" size for each byte range request, Read() will use the provided
// byte slice length.
//
// NOTE: this is not safe for concurrent readers - if that's something you need, use the s3 Downloader
// implementation from the sdk
type ObjectStreamReader struct {
	bucket string
	key    string

	offset int64
	length int64

	client *s3.Client
	ctx    context.Context
}

// NewObjectStreamReader returns an ObjectStream for the given bucket and key. Note that this performs an initial
// HEAD request to ensure the object exists and to get its total size
func NewObjectStreamReader(ctx context.Context, client *s3.Client, bucket, key string) (*ObjectStreamReader, error) {
	key = strings.TrimPrefix(key, "/")
	metadata, err := client.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: &bucket,
		Key:    &key,
	})
	if err != nil {
		return nil, err
	}

	return &ObjectStreamReader{
		ctx:    ctx,
		client: client,
		bucket: bucket,
		key:    key,
		length: junk.Val(metadata.ContentLength),
	}, nil
}

// Read reads up to the next len(p) bytes from the object in s3 based on what has already been read, which may
// be less than len(p)
func (o *ObjectStreamReader) Read(p []byte) (int, error) {
	// edge case: if the object exists, but has 0 bytes, we'll get an error that the range request is not
	// satisfiable, so just check for this on the first read
	if o.offset == 0 && o.length == 0 {
		return 0, io.EOF
	}

	// construct the byte range for this read. ranges are inclusive so the final byte will be offset+len(p)-1
	byteRange := fmt.Sprintf("bytes=%d-%d", o.offset, o.offset+int64(len(p)-1))

	in := &s3.GetObjectInput{
		Bucket: &o.bucket,
		Key:    &o.key,
		Range:  &byteRange,
	}

	resp, err := o.client.GetObject(o.ctx, in)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	// copy the resp body into the provided slice. note this uses ReadAtLeast instead of ReadFull because the
	// actual returned data might be less than what we asked for (i.e. if it's the end) and we'd get an
	// unexpected eof error
	contentLength := int(junk.Val(resp.ContentLength))
	n, err := io.ReadAtLeast(resp.Body, p, contentLength)

	// update the offset based on what was read
	o.offset += int64(n)

	// if there was an actual error from this (there shouldn't have been) return that
	if err != nil {
		return n, err
	}

	// have we read everything?
	if o.offset >= o.length {
		return n, io.EOF
	}

	return n, nil
}

func (o *ObjectStreamReader) Seek(offset int64, whence int) (int64, error) {
	var newOffset int64
	switch whence {
	case io.SeekCurrent:
		newOffset = o.offset + offset
	case io.SeekEnd:
		newOffset = o.length - offset
	case io.SeekStart:
		newOffset = offset
	default:
		return 0, fmt.Errorf("unknown whence type")
	}
	if newOffset < 0 {
		return 0, fmt.Errorf("seek to negative offset")
	}
	if newOffset >= o.length {
		return 0, io.EOF
	}

	o.offset = newOffset
	return o.offset, nil
}