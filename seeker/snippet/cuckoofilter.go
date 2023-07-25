//date: 2023-07-25T16:48:25Z
//url: https://api.github.com/gists/3d779d2f24d11f6f4fd853808ec70037
//owner: https://api.github.com/users/leometzger

package main

import (
	"fmt"

	cuckoo "github.com/seiflotfy/cuckoofilter"
)

func main() {
	filter := cuckoo.NewFilter(cuckoo.DefaultCapacity)
	filter.Insert([]byte("foo"))
	filter.Insert([]byte("bar"))

	fmt.Println(filter.Count())               // 2
	fmt.Println(filter.Lookup([]byte("foo"))) // true
	fmt.Println(filter.Lookup([]byte("bar"))) // true
	fmt.Println(filter.Lookup([]byte("baz"))) // false

	filter.Delete([]byte("foo"))
	fmt.Println(filter.Count())               // 1
	fmt.Println(filter.Lookup([]byte("foo"))) // false
	fmt.Println(filter.Lookup([]byte("bar"))) // true
}