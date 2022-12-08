//date: 2022-12-08T16:57:10Z
//url: https://api.github.com/gists/65529c500e84711d0adadaa47742e112
//owner: https://api.github.com/users/v-for-vasiliev

package main

import (
	"errors"
	"fmt"
	"strconv"
	"sync"
	"time"
)

type StackGeneric[T any] struct {
	lock      sync.Mutex
	container []T
}

func NewStack[T any]() *StackGeneric[T] {
	return &StackGeneric[T]{lock: sync.Mutex{}, container: []T{}}
}

func (s *StackGeneric[T]) Push(element T) {
	s.lock.Lock()
	defer s.lock.Unlock()

	s.container = append(s.container, element)
}

func (s *StackGeneric[T]) Pop() (T, error) {
	s.lock.Lock()
	defer s.lock.Unlock()

	l := len(s.container)
	if l == 0 {
		var empty T
		return empty, errors.New("empty stack")
	}

	res := s.container[l-1]
	s.container = s.container[:l-1]

	return res, nil
}

func doPush(s *StackGeneric[string], count int, wg *sync.WaitGroup) {
	defer wg.Done()
	n := 0
	for n < count {
		s.Push(strconv.Itoa(n))
		n += 1
		fmt.Printf("push\t%d\n", n)
		time.Sleep(250 * time.Millisecond)
	}
}

func doPop(s *StackGeneric[string], count int, wg *sync.WaitGroup) {
	defer wg.Done()
	n := 0
	for n < count {
		v, err := s.Pop()
		n += 1
		if err != nil {
			fmt.Printf("pop error: %v\n", err)
		} else {
			fmt.Printf("pop\t%s\n", v)
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func main() {
	s := NewStack[string]()
	var wg sync.WaitGroup
	wg.Add(2)
	go doPush(s, 50, &wg)
	go doPop(s, 51, &wg)
	wg.Wait()
	fmt.Println("done")
}
