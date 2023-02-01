//date: 2023-02-01T16:42:47Z
//url: https://api.github.com/gists/1b7d8c8202a13bdea049375fd323dac3
//owner: https://api.github.com/users/yrc0d3

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"regexp"
	"strings"
	"time"
)

// 替换markdown文件中新浪微博图床的图片
// 下载后，通过GoPic上传，然后输出替换后的内容到新的文件中
func main() {
	if len(os.Args) != 2 {
		fmt.Println("need an argument!")
		os.Exit(-1)
	}
	mdFilePath := os.Args[1]
	fileBs, err := os.ReadFile(mdFilePath)
	if err != nil {
		fmt.Printf("read file error: %v, path=%s\n", err, mdFilePath)
		os.Exit(-1)
	}
	folderPath := path.Dir(mdFilePath)

	re := regexp.MustCompile(`https://(.*jpg)`)
	res := re.FindAllSubmatch(fileBs, -1)
	fileStr := string(fileBs)
	processed := make(map[string]bool)
	for _, r := range res {
		fmt.Printf("begin to download ")
		oldURL := string(r[0])
		if processed[oldURL] {
			fmt.Println("already processed, continue to next")
			continue
		}

		fmt.Printf("%s\n", oldURL)
		cdnURL := "https://cdn.cdnjson.com/" + string(r[1])
		newURL, err := downloadThenUpload(cdnURL, folderPath)
		if err != nil {
			fmt.Printf("download error: %v\n", err)
			os.Exit(-1)
		}

		// replace oldURL with newURL
		fileStr = strings.ReplaceAll(fileStr, oldURL, newURL)
		fmt.Printf("substitute completed\n")
		// in case of antispam
		time.Sleep(1 * time.Second)
		processed[oldURL] = true
	}

	// save to new file
	newFileName := mdFilePath + ".new"
	err = os.WriteFile(newFileName, []byte(fileStr), 0644)
	if err != nil {
		fmt.Printf("write new file error: %v\n", err)
		os.Exit(-1)
	}
	fmt.Println("all completed!")
}

func downloadThenUpload(url, folderPath string) (string, error) {
	// downlaod
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", errors.New("status code is not 200")
	}

	ws := strings.Split(url, "/")
	objName := strings.Split(ws[len(ws)-1], ".")[0]
	newFilePath := path.Join(folderPath, objName+".png")
	// save to local disk
	f, err := os.Create(newFilePath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	_, err = io.Copy(f, resp.Body)
	if err != nil {
		return "", err
	}

	return upload(newFilePath)
}

func upload(filePath string) (string, error) {
	// upload through PicGo's local server
	// https://picgo.github.io/PicGo-Doc/zh/guide/advance.html#http%E8%B0%83%E7%94%A8%E4%B8%8A%E4%BC%A0%E5%85%B7%E4%BD%93%E8%B7%AF%E5%BE%84%E5%9B%BE%E7%89%87
	type PicGoReq struct {
		List []string `json:"list"`
	}
	type PicGoResp struct {
		Success bool     `json:"success"`
		Result  []string `json:"result"`
	}

	req := &PicGoReq{
		List: []string{filePath},
	}
	reqBs, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	httpResp, err := http.Post("http://127.0.0.1:36677/upload", "application/json", bytes.NewBuffer(reqBs))
	if err != nil {
		return "", err
	}
	defer httpResp.Body.Close()
	if httpResp.StatusCode != http.StatusOK {
		return "", errors.New("status code is not 200")
	}

	respBs, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return "", err
	}
	var resp PicGoResp
	err = json.Unmarshal(respBs, &resp)
	if err != nil {
		return "", err
	}
	if !resp.Success {
		return "", errors.New("upload failed")
	}
	if len(resp.Result) != 1 {
		return "", errors.New("result len is not 1")
	}

	return resp.Result[0], nil
}
