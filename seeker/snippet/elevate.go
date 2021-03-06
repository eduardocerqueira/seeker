//date: 2022-06-01T17:13:02Z
//url: https://api.github.com/gists/39eb14afd87aac97317cb10e70bea8aa
//owner: https://api.github.com/users/lgyaxx

package main

import (
	"fmt"
	"golang.org/x/sys/windows"
	"os"
	"syscall"
	"time"
)

func main() {
	// if not elevated, relaunch by shellexecute with runas verb set
	if !amAdmin() {
		runMeElevated()
	}
	time.Sleep(10*time.Second)

}

func runMeElevated() {
	verb := "runas"
	exe, _ := os.Executable()
	cwd, _ := os.Getwd()
	args := strings.Join(os.Args[1:], " ")
	
	verbPtr, _ := syscall.UTF16PtrFromString(verb)
	exePtr, _ := syscall.UTF16PtrFromString(exe)
	cwdPtr, _ := syscall.UTF16PtrFromString(cwd)
	argPtr, _ := syscall.UTF16PtrFromString(args)
	
	var showCmd int32 = 1 //SW_NORMAL
	
	err := windows.ShellExecute(0, verbPtr, exePtr, argPtr, cwdPtr, showCmd)
	if err != nil {
		fmt.Println(err)
	}
}

func amAdmin() bool {
	_, err := os.Open("\\\\.\\PHYSICALDRIVE0")
	if err != nil {
		fmt.Println("admin no")
		return false
	}
	fmt.Println("admin yes")
	return true
}

