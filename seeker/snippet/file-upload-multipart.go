//date: 2022-06-07T17:09:18Z
//url: https://api.github.com/gists/d6ca0ea7a2e88cc1cb6024a6776da90a
//owner: https://api.github.com/users/noppong-tr

package main

import (
  "net/http"
  "os"
  "bytes"
  "path"
  "path/filepath"
  "mime/multipart"
  "io"
)

func main() {
  fileDir, _ := os.Getwd()
  fileName := "upload-file.txt"
  filePath := path.Join(fileDir, fileName)

  file, _ := os.Open(filePath)
  defer file.Close()

  body := &bytes.Buffer{}
  writer := multipart.NewWriter(body)
  part, _ := writer.CreateFormFile("file", filepath.Base(file.Name()))
  io.Copy(part, file)
  writer.Close()

  r, _ := http.NewRequest("POST", "http://example.com", body)
  r.Header.Add("Content-Type", writer.FormDataContentType())
  client := &http.Client{}
  client.Do(r)
}