//date: 2023-05-18T17:04:43Z
//url: https://api.github.com/gists/43ee2b986897eb0a702c59a6ac7b0cf0
//owner: https://api.github.com/users/lmbek

/*
Solution generated with ChatGPT, use caution:

This Go program demonstrates how to use the Windows API functions SHBrowseForFolderW and SHGetPathFromIDListW to browse for a folder and retrieve its path.
The program starts by defining a struct BROWSEINFO that holds various parameters required for folder browsing. It includes the window handle, root folder, display name, title, flags, and other information.
In the main function, an instance of BROWSEINFO is created. The lpszTitle field is set using the syscall.UTF16PtrFromString function to convert the string "Select Folder" into a UTF-16 encoded pointer.
Next, the program loads the shell32.dll library using syscall.NewLazyDLL and retrieves a handle to the SHBrowseForFolderW function using shell32.NewProc.

The SHBrowseForFolderW function is called with the BROWSEINFO struct as a parameter. It opens a folder selection dialog, allowing the user to browse and select a folder. The function returns a pointer to an item identifier list (PIDL) representing the selected folder.
If a folder is selected (pidl is not zero), the program proceeds to retrieve the path of the selected folder. It creates an array path to hold the path and uses the SHGetPathFromIDListW function to populate the array with the selected folder's path.
The UTF-16 encoded path is then converted to a Go string using syscall.UTF16ToString and stored in the folderPath variable. The program logs the selected folder path using log.Println.
Finally, the program frees the allocated PIDL using the CoTaskMemFree function from the ole32.dll library. It loads the ole32.dll library using syscall.NewLazyDLL and retrieves a handle to the CoTaskMemFree function using ole32.NewProc. The CoTaskMemFree function is called with the pidl parameter to release the memory.
If no folder is selected (pidl is zero), the program logs that the folder selection was canceled.

Overall, this program provides a basic example of using the Windows API functions to browse for a folder and obtain its path.

*/

package main

import (
	"log"
	"syscall"
	"unsafe"
)

type BROWSEINFO struct {
	hwndOwner      syscall.Handle
	pidlRoot       uintptr
	pszDisplayName *uint16
	lpszTitle      *uint16
	ulFlags        uint32
	lpfn           uintptr
	lParam         uintptr
	iImage         int32
}

func main() {
	var bi BROWSEINFO
	bi.lpszTitle, _ = syscall.UTF16PtrFromString("Select Folder")
	bi.ulFlags = 0x00000001 | 0x00000040 // BIF_RETURNONLYFSDIRS | BIF_USENEWUI

	// Get a handle to the shell32.dll library
	shell32 := syscall.NewLazyDLL("shell32.dll")

	// Get a handle to the SHBrowseForFolderW function
	shBrowseForFolder := shell32.NewProc("SHBrowseForFolderW")

	// Call the SHBrowseForFolderW function
	pidl, _, _ := shBrowseForFolder.Call(uintptr(unsafe.Pointer(&bi)))

	if pidl != 0 {
		var path [syscall.MAX_PATH]uint16

		// Get the selected folder path using SHGetPathFromIDListW function
		shGetPathFromIDList := shell32.NewProc("SHGetPathFromIDListW")
		_, _, _ = shGetPathFromIDList.Call(pidl, uintptr(unsafe.Pointer(&path[0])))

		// Convert the UTF-16 encoded path to a Go string
		folderPath := syscall.UTF16ToString(path[:])

		log.Println("Selected folder: ", folderPath)

		// Free the allocated PIDL using CoTaskMemFree function from ole32.dll
		ole32 := syscall.NewLazyDLL("ole32.dll")
		coTaskMemFree := ole32.NewProc("CoTaskMemFree")
		_, _, _ = coTaskMemFree.Call(pidl)
	} else {
		log.Println("Folder selection canceled.")
	}
}
