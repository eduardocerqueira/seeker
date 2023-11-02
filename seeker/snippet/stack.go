//date: 2023-11-02T16:33:56Z
//url: https://api.github.com/gists/be12e78061eddcf69b89d56587641a87
//owner: https://api.github.com/users/mmtdebojit

package iterator

type Stack struct {
	items []interface{}
}

func (s *Stack) Push(item interface{}) {
	s.items = append(s.items, item)
}

func (s *Stack) Pop() interface{} {
	if s.IsEmpty() {
		return nil
	}
	index := len(s.items) - 1
	item := s.items[index]
	s.items = s.items[:index]
	return item
}

func (s *Stack) IsEmpty() bool {
	return len(s.items) == 0
}

func (s *Stack) Size() int {
	return len(s.items)
}
