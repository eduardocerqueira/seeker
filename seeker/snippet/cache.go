//date: 2025-03-18T17:09:27Z
//url: https://api.github.com/gists/79ec0ed0af904420c0bcd0b83d7bbbed
//owner: https://api.github.com/users/esacteksab

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"
)

// Cache struct
type Cache struct {
	path string
}

// NewCache creates a new cache
func NewCache(path string) (*Cache, error) {
	err := os.MkdirAll(path, 0755)
	if err != nil {
		return nil, err
	}
	return &Cache{path: path}, nil
}

// Get retrieves data from the cache
func (c *Cache) Get(key string) ([]byte, error) {
	filePath := filepath.Join(c.path, key)
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// Set stores data in the cache
func (c *Cache) Set(key string, data []byte) error {
	filePath := filepath.Join(c.path, key)
	err := ioutil.WriteFile(filePath, data, 0644)
	if err != nil {
		return err
	}
	return nil
}

// Delete removes data from the cache
func (c *Cache) Delete(key string) error {
	filePath := filepath.Join(c.path, key)
	err := os.Remove(filePath)
	if err != nil {
		return err
	}
	return nil
}

// Exists checks if a key exists in the cache
func (c *Cache) Exists(key string) bool {
	filePath := filepath.Join(c.path, key)
	_, err := os.Stat(filePath)
	return !os.IsNotExist(err)
}

// Clear clears the entire cache
func (c *Cache) Clear() error {
	err := os.RemoveAll(c.path)
	if err != nil {
		return err
	}
    err = os.MkdirAll(c.path, 0755)
    if err != nil {
        return err
    }
    return nil
}

func main() {
	cacheDir := "mycache"
	cache, err := NewCache(cacheDir)
	if err != nil {
		fmt.Println("Error creating cache:", err)
		return
	}

	key := "mydata"
	data := []byte("Hello, cache!")

	// Set data in cache
	err = cache.Set(key, data)
	if err != nil {
		fmt.Println("Error setting data:", err)
		return
	}

    // Check if data exists
    exists := cache.Exists(key)
    fmt.Println("Exists:", exists)

	// Get data from cache
	cachedData, err := cache.Get(key)
	if err != nil {
		fmt.Println("Error getting data:", err)
		return
	}
	fmt.Println("Cached data:", string(cachedData))

	// Delete data from cache
	err = cache.Delete(key)
	if err != nil {
		fmt.Println("Error deleting data:", err)
		return
	}

    // Check if data exists after deletion
    exists = cache.Exists(key)
    fmt.Println("Exists after deletion:", exists)

    // Set more data
    err = cache.Set(key, data)
	if err != nil {
		fmt.Println("Error setting data:", err)
		return
	}

    // Clear the cache
    err = cache.Clear()
    if err != nil {
        fmt.Println("Error clearing cache:", err)
        return
    }

    // Check if data exists after clearing
    exists = cache.Exists(key)
    fmt.Println("Exists after clearing:", exists)

}
