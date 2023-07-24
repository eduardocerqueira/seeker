//date: 2023-07-24T16:52:18Z
//url: https://api.github.com/gists/e3b699df045fd8a895ddefe7b92b9234
//owner: https://api.github.com/users/maxclav

package stack

// Stack represents a basic stack data structure.
type Stack struct {
	items []any
}

// Push adds an item to the top of the stack.
func (s *Stack) Push(item any) {
	s.items = append(s.items, item)
}

// Pop removes and returns the item from the top of the stack.
// It also returns a boolean value indicating whether the stack is empty.
func (s *Stack) Pop() (any, bool) {
	if len(s.items) == 0 {
		return nil, false
	}
	index := len(s.items) - 1
	item := s.items[index]
	s.items = s.items[:index]
	return item, true
}

// IsEmpty checks if the stack is empty.
func (s *Stack) IsEmpty() bool {
	return len(s.items) == 0
}

// Size returns the number of elements in the stack.
func (s *Stack) Size() int {
	return len(s.items)
}

// Top returns the item at the top of the stack without removing it.
func (s *Stack) Top() (any, bool) {
	if len(s.items) == 0 {
		return nil, false
	}
	return s.items[len(s.items)-1], true
}