//date: 2022-06-06T16:50:15Z
//url: https://api.github.com/gists/89d0cdf4134157662cab35a8b4798469
//owner: https://api.github.com/users/unakatsuo

package main

/*
#include <stdlib.h>
#include <string.h>

struct AAA {
  char a[10];
};
*/
import "C"
import (
        "fmt"
        "unsafe"
)


func main(){
  aaa := (*C.struct_AAA)(C.calloc(C.sizeof_struct_AAA, 1))

  // build fail: cannot convert aaa.a (type [32]_Ctype_int) to type unsafe.Pointer
  //fmt.Printf("aaa.a=%p, &aaa.a[0]=%p\n", unsafe.Pointer(aaa.a), unsafe.Pointer(&aaa.a[0]))
  fmt.Printf("&aaa.a=%p, &aaa.a[0]=%p\n", unsafe.Pointer(&aaa.a), unsafe.Pointer(&aaa.a[0]))

  src := [10]byte{1,2,3,4,5,6,7,8,9,10}
  psrc := &src
  fmt.Printf("&src=%p, &scr[0]=%p\n", unsafe.Pointer(&src), unsafe.Pointer(&src[0]))
  fmt.Printf("psrc=%p, &psrc[0]=%p,  &psrc[1]=%p\n", unsafe.Pointer(psrc), unsafe.Pointer(&psrc[0]), unsafe.Pointer(&psrc[1]))
  fmt.Printf("len(src)=%d, len(psrc)=%d\n", len(src), len(psrc))

  // Copy from Go fixed array pointer to C array.
  C.memcpy(unsafe.Pointer(&aaa.a), unsafe.Pointer(psrc), C.size_t(len(psrc)))

  fmt.Println(aaa.a)
}