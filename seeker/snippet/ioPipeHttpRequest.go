//date: 2024-02-15T17:03:09Z
//url: https://api.github.com/gists/2ff33dce717af25397cbf9033e054232
//owner: https://api.github.com/users/yakuter

package main

import (
    "fmt"
    "io"
    "net/http"
    "os"
)

func main() {
    filePath := "myFile.txt"

    // Create reader and writer using io.Pipe
    reader, writer := io.Pipe()

    // Read the file and write to the pipe in a goroutine
    go func() {
        file, err := os.Open(filePath)
        if err != nil {
            fmt.Println("Error opening file:", err)
            return
        }
        defer file.Close()

        // Copy file content to the pipe
        _, err = io.Copy(writer, file)
        if err != nil {
            fmt.Println("Error writing file to the pipe:", err)
            return
        }

        // Close the writer when the writing operation is done
        writer.Close()
    }()

    // Create HTTP request and send file content as body
    resp, err := http.Post("http://example.com/upload", "application/octet-stream", reader)
    if err != nil {
        fmt.Println("Error during HTTP request:", err)
        return
    }
    defer resp.Body.Close()

    fmt.Println("HTTP Status:", resp.Status)
}
