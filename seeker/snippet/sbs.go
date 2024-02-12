//date: 2024-02-12T16:56:37Z
//url: https://api.github.com/gists/74f6bcb99edf77cbdb50560e9b5394b4
//owner: https://api.github.com/users/alanyang

package main

import (
	"fmt"
	"reflect"
	"unsafe"

)

func main() {

	s := "abcd"
	fmt.Println(s)

	b := String2Bytes(s)
	fmt.Println(b)
	fmt.Println(bytes2String(b))
}

func String2Bytes(s string) []byte {
	if len(s) == 0 {
		return nil
	}

	return unsafe.Slice(unsafe.StringData(s), len(s))
}

func Bytes2String(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	return unsafe.String(unsafe.SliceData(b), len(b))
}

func string2Bytes(s string) []byte {
	if len(s) == 0 {
		return nil
	}

	h := *(*reflect.StringHeader)(unsafe.Pointer(&s))
	return *(*[]byte)(unsafe.Pointer(&reflect.SliceHeader{
		Data: h.Data,
		Len:  h.Len,
		Cap:  h.Len,
	}))
}

func bytes2String(b []byte) string {
	if len(b) == 0 {
		return ""
	}

	h := *(*reflect.SliceHeader)(unsafe.Pointer(&b))

	return *(*string)(unsafe.Pointer(&reflect.StringHeader{
		Len:  h.Len,
		Data: h.Data,
	}))
}
