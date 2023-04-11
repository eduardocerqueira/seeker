//date: 2023-04-11T16:48:25Z
//url: https://api.github.com/gists/caa93e0f10708cd79e408ccaa2d64084
//owner: https://api.github.com/users/NoobforAl

package main

import (
	"fmt"

	"github.com/kr/pretty"
)

type Value interface {
	int | int64 | int32 | int16 | int8 |
		float64 | float32 | byte | string
}

type LinkedList[T Value] struct {
	Val  T
	Next *LinkedList[T]
}

func (list *LinkedList[T]) Insert(Val T) {
	for tmp := list; tmp != nil; tmp = tmp.Next {
		if tmp.Next == nil {
			tmp.Next = &LinkedList[T]{Val, nil}
			break
		}
	}
}

func (list *LinkedList[T]) Pop() {
	for tmp := list; tmp.Next != nil; tmp = tmp.Next {
		if tmp.Next.Next == nil {
			tmp.Next = nil
			break
		}
	}
}

func (list *LinkedList[T]) Find(Val T) *LinkedList[T] {
	for tmp := list; tmp != nil; tmp = tmp.Next {
		if tmp.Val == Val {
			return tmp
		}
	}
	return nil
}

func main() {
	// Make Head
	list := LinkedList[int]{1, nil}

	// Insert 2 Value
	list.Insert(2)
	list.Insert(3)

	// Pop 2 Value
	list.Pop()

	// Find One Value
	f := list.Find(1)
	fmt.Printf("Find: %# v\n", pretty.Formatter(f))

	fmt.Printf("All Value: %# v", pretty.Formatter(list))
}
