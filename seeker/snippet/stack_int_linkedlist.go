//date: 2023-12-14T16:41:00Z
//url: https://api.github.com/gists/1ea3627c7582f167e1bacd904dea9d75
//owner: https://api.github.com/users/jittakal

// You can edit this code!
// Click here and start typing.
package main

import "fmt"

// Implement stack using linked List

type Stack interface {
	Push(item int)
	Pop() (int, error)
	IsEmpty() bool
}

type StackNode struct {
	Val  int
	Next *StackNode
}

type StackDS struct {
	Top *StackNode
}

func NewStack() *StackDS {
	return &StackDS{}
}

func NewStackNode(item int) *StackNode {
	return &StackNode{
		Val: item,
	}
}

func (s *StackDS) Push(item int) {
	stackNode := NewStackNode(item)

	if s.IsEmpty() {
		s.Top = stackNode
	} else {
		stackNode.Next = s.Top
		s.Top = stackNode
	}
}

func (s *StackDS) Pop() (int, error) {
	if s.IsEmpty() {
		return 0, fmt.Errorf("stack is empty")
	}

	item := s.Top.Val
	s.Top = s.Top.Next

	return item, nil
}

func (s *StackDS) IsEmpty() bool {
	return s.Top == nil
}

func main() {
	stack := NewStack()

	stack.Push(10)
	stack.Push(20)

	fmt.Println(stack.Pop())
	fmt.Println(stack.Pop())
	fmt.Println(stack.Pop())
}
