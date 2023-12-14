//date: 2023-12-14T16:41:40Z
//url: https://api.github.com/gists/25d6b6b70be50fc3fa90db03e22e7f19
//owner: https://api.github.com/users/jittakal

// You can edit this code!
// Click here and start typing.
package main

import "fmt"

// Stack implementation using singly linked list

// Stack interface
type Stack[T any] interface {
	// Push
	Push(item T)
	// Pop
	Pop() (T, error)
	// Is Empty
	IsEmpty() bool
}

// Stack LinkedList Node structure
type StackDSNode[T any] struct {
	Val  T
	Next *StackDSNode[T]
}

// Stack DataStructure
type StackDS[T any] struct {
	Top *StackDSNode[T]
}

// New function for StackDSNode
func NewStackDSNode[T any](item T) *StackDSNode[T] {
	return &StackDSNode[T]{
		Val: item,
	}
}

// New function for StackDS
func NewStackDS[T any]() *StackDS[T] {
	return &StackDS[T]{}
}

// Implement Stack interface
// Push
func (s *StackDS[T]) Push(item T) {
	stackNode := NewStackDSNode[T](item)
	// Is Empty
	if s.IsEmpty() {
		s.Top = stackNode
	} else {
		stackNode.Next = s.Top
		s.Top = stackNode
	}
}

// Pop
func (s *StackDS[T]) Pop() (T, error) {
	if s.IsEmpty() {
		var zeroT T
		return zeroT, fmt.Errorf("stack is empty")
	}

	item := s.Top.Val
	s.Top = s.Top.Next

	return item, nil
}

// Is Empty
func (s *StackDS[T]) IsEmpty() bool {
	return s.Top == nil
}

func main() {
	stack := NewStackDS[int]()

	fmt.Println(stack.IsEmpty())

	stack.Push(10)
	stack.Push(20)

	fmt.Println(stack.Pop())
	fmt.Println(stack.Pop())
	fmt.Println(stack.Pop())
}
