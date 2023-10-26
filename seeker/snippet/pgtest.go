//date: 2023-10-26T16:42:46Z
//url: https://api.github.com/gists/3828b89e2c32239ac96b8be6a80aca1f
//owner: https://api.github.com/users/raharper

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"

	"golang.org/x/sys/unix"

	"github.com/apex/log"
	gzip "github.com/klauspost/pgzip"
)

// Compressor is an interface which users can use to implement different
// compression types.
type Compressor interface {
	// Compress sets up the streaming compressor for this compression type.
	Compress(io.Reader, int, int) (io.ReadCloser, error)

	// MediaTypeSuffix returns the suffix to be added to the layer to
	// indicate what compression type is used, e.g. "gzip", or "" for no
	// compression.
	MediaTypeSuffix() string
}

// GzipCompressor provides gzip compression.
var GzipCompressor Compressor = gzipCompressor{}

type gzipCompressor struct{}

func (gz gzipCompressor) Compress(reader io.Reader, writeBufSize, readBufSize int) (io.ReadCloser, error) {
	pipeReader, pipeWriter := io.Pipe()

	gzw := gzip.NewWriter(pipeWriter)
	if err := gzw.SetConcurrency(writeBufSize, 2*runtime.NumCPU()); err != nil {
		return nil, fmt.Errorf("Error setting concurrency level to %v blocks: %s", 2*runtime.NumCPU(), err)
	}
	go func() {
		if _, err := Copy(gzw, reader, readBufSize); err != nil {
			log.Warnf("gzip compress: could not compress layer: %v", err)
			// #nosec G104
			_ = pipeWriter.CloseWithError(fmt.Errorf("Error compressing layer: %s", err))
			return
		}
		if err := gzw.Close(); err != nil {
			log.Warnf("gzip compress: could not close gzip writer: %v", err)
			// #nosec G104
			_ = pipeWriter.CloseWithError(fmt.Errorf("Error close gzip writer: %s", err))
			return
		}
		if err := pipeWriter.Close(); err != nil {
			log.Warnf("gzip compress: could not close pipe: %v", err)
			// We don't CloseWithError because we cannot override the Close.
			return
		}
	}()

	return pipeReader, nil
}

func (gz gzipCompressor) MediaTypeSuffix() string {
	return "gzip"
}

// Copy has identical semantics to io.Copy except it will automatically resume
// the copy after it receives an EINTR error.
func Copy(dst io.Writer, src io.Reader, size int) (int64, error) {
	// Make a buffer so io.Copy doesn't make one for each iteration.
	var buf []byte
	// log.Infof("Copy called with size %d", size)
	// size := 32 * 1024
	if lr, ok := src.(*io.LimitedReader); ok && lr.N < int64(size) {
		if lr.N < 1 {
			size = 1
		} else {
			size = int(lr.N)
		}
	}
	// log.Infof("Copy using size %d", size)
	buf = make([]byte, size)

	var written int64
	for {
		n, err := io.CopyBuffer(dst, src, buf)
		written += n // n is always non-negative
		if errors.Is(err, unix.EINTR) {
			continue
		}
		return written, err
	}
}

func doCompress(data []byte, writeBufSize, readBufSize int) {
	r := bytes.NewReader(data)
	var dest bytes.Buffer

	// umoci defaults
	compDefault, err := GzipCompressor.Compress(r, writeBufSize, readBufSize)
	if err != nil {
		panic(err.Error())
	}

	size, err := Copy(&dest, compDefault, readBufSize)
	if err != nil {
		panic(err.Error())
	}

	compDefault.Close()

	hasher := sha256.New()
	hasher.Write(dest.Bytes())
	sha256Str := hex.EncodeToString(hasher.Sum(nil))
	log.Infof("writeSize: %d readSize: %d resultSize: %d sha256: %s", writeBufSize, readBufSize, size, sha256Str)
}

func main() {
	// create/load some known data, multi-megabyte
	// sha256sum it
	// compress with exec'ed gzip
	// compress with pgzip with defaults, using Copy 32K buff
	// compress with pgzip with defaults, using Copy 64k buff
	// compress with pgzip with 1M size, using Copy 32K buff
	// compress with pgzip with 1M size, using Copy 64K buff
	// sha256sum each compressed object
	// print results

	helloWorld := []byte("hello, world\n")
	yesImg, err := os.ReadFile("yes.img")
	if err != nil {
		log.Errorf("Missing yes.img file, generate with: yes | dd iflag=fullblock bs=1M count=2 of=yes.img")
		os.Exit(1)
	}

	log.Infof("Testing Hello, World buffer")
	doCompress(helloWorld, 256<<10, 32*1024) // umoci defaults
	doCompress(helloWorld, 256<<10, 64*1024) // umoci deafults + larger copy buffer
	doCompress(helloWorld, 256<<12, 32*1024) // umoci+stacker defaults
	doCompress(helloWorld, 256<<12, 64*1024) // umoci+stacker defaults
	fmt.Printf("\n")
	log.Infof("Testing Yes 2M image buffer")
	doCompress(yesImg, 256<<10, 32*1024) // umoci defaults
	doCompress(yesImg, 256<<10, 64*1024) // umoci deafults + larger copy buffer
	doCompress(yesImg, 256<<12, 32*1024) // umoci+stacker defaults
	doCompress(yesImg, 256<<12, 64*1024) // umoci+stacker defaults
}
