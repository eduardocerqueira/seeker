//date: 2023-09-07T16:52:34Z
//url: https://api.github.com/gists/2d9ca5773a1958cc7a4f5cc7b3faa8d4
//owner: https://api.github.com/users/vdparikh

package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "sync"
)

// KeyValueStore is a simple in-memory key-value store.
type KeyValueStore struct {
    data map[string]string
    mu   sync.Mutex
}

// NewKeyValueStore creates a new instance of KeyValueStore.
func NewKeyValueStore() *KeyValueStore {
    return &KeyValueStore{
        data: make(map[string]string),
    }
}

// Get retrieves the value associated with a key.
func (kv *KeyValueStore) Get(key string) (string, bool) {
    kv.mu.Lock()
    defer kv.mu.Unlock()
    value, exists := kv.data[key]
    return value, exists
}

// Set stores a key-value pair.
func (kv *KeyValueStore) Set(key, value string) {
    kv.mu.Lock()
    defer kv.mu.Unlock()
    kv.data[key] = value
}

// Update updates the value associated with a key.
func (kv *KeyValueStore) Update(key, value string) bool {
    kv.mu.Lock()
    defer kv.mu.Unlock()
    _, exists := kv.data[key]
    if exists {
        kv.data[key] = value
    }
    return exists
}

// Delete removes a key-value pair from the store.
func (kv *KeyValueStore) Delete(key string) bool {
    kv.mu.Lock()
    defer kv.mu.Unlock()
    _, exists := kv.data[key]
    if exists {
        delete(kv.data, key)
    }
    return exists
}

func main() {
    kvStore := NewKeyValueStore()

    http.HandleFunc("/get", func(w http.ResponseWriter, r *http.Request) {
        key := r.URL.Query().Get("key")
        value, exists := kvStore.Get(key)
        if !exists {
            http.NotFound(w, r)
            return
        }
        fmt.Fprintf(w, "Key: %s, Value: %s\n", key, value)
    })

    http.HandleFunc("/set", func(w http.ResponseWriter, r *http.Request) {
        key := r.URL.Query().Get("key")
        value := r.URL.Query().Get("value")
        kvStore.Set(key, value)
        fmt.Fprintf(w, "Key: %s, Value: %s is set\n", key, value)
    })

    http.HandleFunc("/update", func(w http.ResponseWriter, r *http.Request) {
        key := r.URL.Query().Get("key")
        value := r.URL.Query().Get("value")
        exists := kvStore.Update(key, value)
        if !exists {
            http.NotFound(w, r)
            return
        }
        fmt.Fprintf(w, "Key: %s, Value: %s is updated\n", key, value)
    })

    http.HandleFunc("/delete", func(w http.ResponseWriter, r *http.Request) {
        key := r.URL.Query().Get("key")
        exists := kvStore.Delete(key)
        if !exists {
            http.NotFound(w, r)
            return
        }
        fmt.Fprintf(w, "Key: %s is deleted\n", key)
    })

    http.HandleFunc("/list", func(w http.ResponseWriter, r *http.Request) {
        kvStore.mu.Lock()
        defer kvStore.mu.Unlock()
        keys := make([]string, 0, len(kvStore.data))
        for key := range kvStore.data {
            keys = append(keys, key)
        }
        keysJSON, err := json.Marshal(keys)
        if err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }
        w.Header().Set("Content-Type", "application/json")
        w.Write(keysJSON)
    })

    http.ListenAndServe(":8080", nil)
}