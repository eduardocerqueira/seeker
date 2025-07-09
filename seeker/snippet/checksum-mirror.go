//date: 2025-07-09T17:07:05Z
//url: https://api.github.com/gists/3cd979f33027f2f01a129d2a9caeeea3
//owner: https://api.github.com/users/pravdomil

package main

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

const attrName = "user.checksum.sha256"

// computeSHA256Binary returns the raw 32-byte SHA-256 digest of the file at path.
func computeSHA256Binary(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return nil, err
	}
	return h.Sum(nil), nil
}

// getXAttrBinary reads exactly 32 bytes from the named xattr (assuming SHA-256 raw digest).
// Returns ENOATTR if the attribute is not present.
func getXAttrBinary(path, attr string) ([]byte, error) {
	buf := make([]byte, 32)
	n, err := unix.Lgetxattr(path, attr, buf)
	if err != nil {
		return nil, err
	}
	return buf[:n], nil
}

// setXAttrBinary writes the raw bytes into the named xattr (using Lsetxattr so symlinks are not followed).
func setXAttrBinary(path, attr string, data []byte) error {
	return unix.Lsetxattr(path, attr, data, 0)
}

// verifyFile reads (or initializes) the SHA-256 checksum xattr on the given file.
func verifyFile(path string) (stored []byte, actual []byte, err error) {
	// Try to read existing checksum attribute
	stored, err = getXAttrBinary(path, attrName)
	if err != nil && !errors.Is(err, unix.ENOATTR) {
		// some error other than “no attribute”
		return nil, nil, err
	}
	missing := errors.Is(err, unix.ENOATTR)

	// Compute the actual checksum
	actual, err = computeSHA256Binary(path)
	if err != nil {
		return nil, nil, err
	}

	if missing {
		// No attr → set it to the computed value
		if err = setXAttrBinary(path, attrName, actual); err != nil {
			return nil, nil, err
		}
		// stored stays nil, actual is what we just set
		return nil, actual, nil
	}

	// Attr existed → return both for comparison
	return stored, actual, nil
}

func main() {
	// Parse primary and secondary directories from flags
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s <primary-dir> <secondary-dir>\n", filepath.Base(os.Args[0]))
	}
	flag.Parse()

	if flag.NArg() != 2 {
		flag.Usage()
		os.Exit(1)
	}

	primary := flag.Arg(0)
	secondary := flag.Arg(1)

	var failures []string

	err := filepath.Walk(primary, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.Mode().IsRegular() {
			return nil
		}

		// Compute relative path
		rel, err := filepath.Rel(primary, path)
		if err != nil {
			return err
		}

		// Corresponding file in secondary
		other := filepath.Join(secondary, rel)

		// Verify primary
		storedP, actualP, err := verifyFile(path)
		if err != nil {
			return fmt.Errorf("error verifying primary %s: %w", path, err)
		}

		// Verify secondary
		storedS, actualS, err := verifyFile(other)
		if err != nil {
			return fmt.Errorf("error verifying secondary %s: %w", other, err)
		}

		// Compare stored vs actual for each
		if storedP != nil && !bytes.Equal(storedP, actualP) {
			failures = append(failures, fmt.Sprintf("checksum mismatch in primary: %s", path))
		}
		if storedS != nil && !bytes.Equal(storedS, actualS) {
			failures = append(failures, fmt.Sprintf("checksum mismatch in secondary: %s", other))
		}

		// Compare primary vs secondary actual
		if !bytes.Equal(actualP, actualS) {
			failures = append(failures, fmt.Sprintf("different content: %s", path))
		}

		// Print status symbol
		if storedP == nil || storedS == nil {
			fmt.Printf("➕ %s\n", rel)
		} else if bytes.Equal(actualP, actualS) {
			fmt.Printf("✔ %s\n", rel)
		} else {
			fmt.Printf("❌ %s\n", rel)
		}

		return nil
	})

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error walking primary directory: %v\n", err)
		os.Exit(1)
	}

	if len(failures) > 0 {
		fmt.Fprintln(os.Stderr, "Failures:")
		for _, f := range failures {
			fmt.Fprintln(os.Stderr, f)
		}
		os.Exit(1)
	}
}
