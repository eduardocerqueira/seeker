//date: 2024-06-26T16:36:56Z
//url: https://api.github.com/gists/e0562d4c5db32ff0ef1d79d73eb8016a
//owner: https://api.github.com/users/wjkoh

package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
)

func main() {
	r, err := os.Open("big.pdf")
	if err != nil {
		log.Fatal(err)
	}
	defer r.Close()
	w, err := os.Create("small.pdf")
	if err != nil {
		log.Fatal(err)
	}
	err = TruncatePdf(w, r, 300)
	if err != nil {
		log.Fatal(err)
	}
	err = w.Close()
	if err != nil {
		log.Fatal(err)
	}
}

func TruncatePdf(w io.Writer, r io.Reader, numPages int) error {
	input, err := ioutil.TempFile("", "input-*.pdf")
	if err != nil {
		log.Fatal(err)
	}
	defer os.Remove(input.Name())
	_, err = io.Copy(input, r)
	if err != nil {
		return err
	}
	err = input.Sync()
	if err != nil {
		return err
	}
	output, err := ioutil.TempFile("", "output-*.pdf")
	if err != nil {
		return err
	}
	defer os.Remove(output.Name())
	cmd := exec.Command("qpdf", input.Name(), "--pages", ".", fmt.Sprintf("1-%d", numPages), "--", output.Name())
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Print(string(out))
		return err
	}
	_, err = io.Copy(w, output)
	return err
}