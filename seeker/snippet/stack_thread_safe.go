//date: 2022-12-08T16:53:18Z
//url: https://api.github.com/gists/736e887eecaaa6cfe7096562a0c4dbe9
//owner: https://api.github.com/users/v-for-vasiliev

package main

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

type StackTreadSafe struct {
	lock      sync.Mutex
	container []int // Array grows exponentially, so we'll get very few allocations in practice
}

func NewStack() *StackTreadSafe {
	return &StackTreadSafe{sync.Mutex{}, make([]int, 0)}
}

func (s *StackTreadSafe) Push(v int) {
	s.lock.Lock()
	defer s.lock.Unlock()

	s.container = append(s.container, v)
}

func (s *StackTreadSafe) Pop() (int, error) {
	s.lock.Lock()
	defer s.lock.Unlock()

	l := len(s.container)
	if l == 0 {
		return -1, errors.New("empty stack")
	}

	res := s.container[l-1]
	s.container = s.container[:l-1]

	return res, nil
}

func doPush(s *StackTreadSafe, count int, wg *sync.WaitGroup) {
	defer wg.Done()
	n := 0
	for n < count {
		s.Push(n)
		n += 1
		fmt.Printf("push\t%d\n", n)
		time.Sleep(250 * time.Millisecond)
	}
}

func doPop(s *StackTreadSafe, count int, wg *sync.WaitGroup) {
	defer wg.Done()
	n := 0
	for n < count {
		v, err := s.Pop()
		n += 1
		if err != nil {
			fmt.Printf("pop error: %v\n", err)
		} else {
			fmt.Printf("pop\t%d\n", v)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func main() {
	s := NewStack()
	var wg sync.WaitGroup
	wg.Add(2)
	go doPush(s, 50, &wg)
	go doPop(s, 51, &wg)
	wg.Wait()
	fmt.Println("done")
}
