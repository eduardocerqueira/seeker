//date: 2023-02-17T16:56:56Z
//url: https://api.github.com/gists/b7c99b08fec031bc9b5164f2041183b1
//owner: https://api.github.com/users/corneliusdavid

package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	if err != nil {
		return err
	}
	return nil
}

type albumInfo struct {
	title, desc, topic, thumb, tags, src, dest string
}

func promptAlbumInfo() albumInfo {
	var a albumInfo

	// Prompt for title, description, and content topic
	fmt.Print("Enter title: ")
	a.title, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	a.title = strings.TrimSpace(a.title)

	fmt.Print("Enter description: ")
	a.desc, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	a.desc = strings.TrimSpace(a.desc)

	fmt.Print("Enter content topic\\folder under 'assets' and 'content'; e.g. 'family_events\\christmas-at-doyles-1999': ")
	a.topic, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	a.topic = strings.TrimSpace(a.topic)

	fmt.Print("Enter source folder: ")
	a.src, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	a.src = strings.TrimSpace(a.src)

	fmt.Print("Enter selected photo for album thumb: ")
	a.thumb, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	a.thumb = strings.TrimSpace(a.thumb)

	fmt.Print("Enter comma-separated tags for photos in this folder (e.g. holiday, birthday, mountains, hiking, lakes, pathfinders): ")
	a.tags, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	a.tags = strings.TrimSpace(a.tags)

	return a
}

func main() {
	webroot := `\web\www.corneliusconcepts.pictures`

	newAlbum := promptAlbumInfo()

	// create assets folder for images
	assets_destpath := filepath.Join(filepath.FromSlash(webroot), "assets", newAlbum.topic)
	err := os.Mkdir(assets_destpath, os.ModePerm)
	if err != nil {
		log.Fatal(err)
	}

	// create content folder for _index.md
	content_destpath := filepath.Join(webroot, "content", newAlbum.topic)
	fmt.Printf("-----------------New Photo Album: %s---------------------\n", newAlbum.title)
	err = os.Mkdir(content_destpath, os.ModePerm)
	if err != nil {
		log.Fatal(err)
	}

	// create content index file
	content_file, err := os.Create(filepath.Join(content_destpath, "_index.md"))
	if err != nil {
		log.Fatal(err)
	}
	defer content_file.Close()

	content_file.WriteString("---\n")
	content_file.WriteString(fmt.Sprintf("title: \"%s\"\n", newAlbum.title))
	content_file.WriteString(fmt.Sprintf("description: \"%s\"\n", newAlbum.desc))
	content_file.WriteString("draft: false\n")
	content_file.WriteString(fmt.Sprintf("albumthumb: %s\n", strings.ReplaceAll(filepath.Join(newAlbum.topic, newAlbum.thumb), `\`, `/`)))
	content_file.WriteString(fmt.Sprintf("tags: [%s]\n", newAlbum.tags))
	content_file.WriteString("resources:\n")

	// keep track of earliest file date of images to copy
	var earliest_date time.Time

	// Copy all images from source folder to destination folder
	err = filepath.Walk(newAlbum.src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Fatal(err)
		}
		if !info.IsDir() && strings.HasSuffix(strings.ToLower(path), ".jpg") {
			src, _ := filepath.Abs(path)
			dst := filepath.Join(assets_destpath, info.Name())
			fmt.Printf("Copying %s to %s...\n", src, dst)
			err = copyFile(src, dst)

			if err != nil {
				fmt.Printf("Error copying file %s: %v\n", src, err)
			} else {
				// list the resource
				content_file.WriteString(fmt.Sprintf("- %s\n", info.Name()))
				// save the earliest file date from the list
				if earliest_date.IsZero() || earliest_date.Before(info.ModTime()) {
					earliest_date = info.ModTime()
				}
			}
		}
		return err
	})
	if err != nil {
		fmt.Printf("Error copying files %v\n", err)
	} else {
		content_file.WriteString(fmt.Sprintf("date: %s\n", earliest_date.Format(time.RFC3339)))
		content_file.WriteString("---\n")
	}
}
