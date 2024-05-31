//date: 2024-05-31T16:35:19Z
//url: https://api.github.com/gists/1cb43ecc3b1b5918325fc3c7a48f1b14
//owner: https://api.github.com/users/pavelanni

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

func hashFile(filePath string, wg *sync.WaitGroup, resultChan chan<- string, errorChan chan<- error) {
	defer wg.Done()

	file, err := os.Open(filePath)
	if err != nil {
		errorChan <- err
		return
	}
	defer file.Close()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		errorChan <- err
		return
	}

	// Send the hash to the result channel
	resultChan <- hex.EncodeToString(hasher.Sum(nil))
}

// hashDirectory calculates the SHA256 hash of all files in a directory and its subdirectories.
//
// Parameters:
// - dirPath: the path of the directory to be hashed.
//
// Returns:
// - string: the hexadecimal representation of the SHA256 hash of all files in the directory.
// - error: an error if any occurred during the hashing process.
func hashDirectory(dirPath string) (string, error) {
	hasher := sha256.New()
	var files []string

	// Walk the directory and its subdirectories
	// Create a list of files to hash
	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return "", err
	}

	sort.Strings(files) // Ensure consistent order

	// Channel to store the results
	resultChan := make(chan string, len(files))
	// Channel for the errors
	errorChan := make(chan error, len(files))
	var wg sync.WaitGroup

	// Hash all files in parallel
	for _, file := range files {
		wg.Add(1)
		go hashFile(file, &wg, resultChan, errorChan)
	}

	// Wait for all files to be hashed
	go func() {
		wg.Wait()
		close(resultChan)
		close(errorChan)
	}()

	for {
		select {
		case err := <-errorChan:
			if err != nil {
				return "", err
			}
		case fileHash, ok := <-resultChan: // Check if the channel is closed; if so, break the loop
			if !ok {
				return hex.EncodeToString(hasher.Sum(nil)), nil
			}
			hasher.Write([]byte(fileHash))
		}
	}
}

func main() {
	if len(os.Args) != 2 {
		log.Fatalf("Usage: %s <directory>", os.Args[0])
	}
	dirPath := os.Args[1]
	dirHash, err := hashDirectory(dirPath)
	if err != nil {
		log.Fatalf("Failed to hash directory: %v", err)
	}
	fmt.Printf("Hash of the directory: %s\n", dirHash)
}
