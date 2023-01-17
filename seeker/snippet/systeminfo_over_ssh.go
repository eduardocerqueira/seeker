//date: 2023-01-17T17:02:42Z
//url: https://api.github.com/gists/4ad78bbd84a24b381e687c8f9be947df
//owner: https://api.github.com/users/MaineK00n

package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
)

func main() {
	base : "**********":\\Users\\user\\.ssh\\id_rsa", "-o", "PasswordAuthentication=no", "127.0.0.1", "powershell.exe", "-NoProfile", "-NonInteractive"}

	// case 1: 一応すべて表示される
	args := append(base, "systeminfo.exe")
	run(exec.Command("ssh.exe", args...), os.Stdout)

	// case 2: 途切れる ref: https://pbs.twimg.com/media/Fmr15EGaUAA680d?format=png&name=large
	var buf bytes.Buffer
	args = append(base, "systeminfo.exe")
	run(exec.Command("ssh.exe", args...), &buf)
	fmt.Println(buf.String())
}

func run(cmd *exec.Cmd, w io.Writer) {
	fmt.Printf("cmd: %s\n", cmd.String())

	cmd.Stdout = w
	cmd.Run()
}
 args...), &buf)
	fmt.Println(buf.String())
}

func run(cmd *exec.Cmd, w io.Writer) {
	fmt.Printf("cmd: %s\n", cmd.String())

	cmd.Stdout = w
	cmd.Run()
}
