//date: 2022-05-04T17:06:26Z
//url: https://api.github.com/gists/e307e458b466f067c3c868aced14346e
//owner: https://api.github.com/users/EbicHecker

package queue

import (
	"errors"
	"fmt"
	"sync"
)

type ConcurrentQueue struct {
	rwmutex sync.RWMutex
	slice   []any
}

func NewConcurrentQueue() *ConcurrentQueue {
	return &ConcurrentQueue{slice: make([]any, 0)}
}

func (queue *ConcurrentQueue) Dequeue() any {
	queue.rwmutex.Lock()
	defer queue.rwmutex.Unlock()
	firstElement := queue.slice[0]
	queue.slice = queue.slice[1:]
	return firstElement
}

func (queue *ConcurrentQueue) TryDeque() (any, error) {
	queue.rwmutex.Lock()
	defer queue.rwmutex.Unlock()
	if queue.IsEmpty() {
		return nil, errors.New("queue empty")
	}
	firstElement := queue.slice[0]
	queue.slice = queue.slice[1:]
	return firstElement, nil
}

func (queue *ConcurrentQueue) Enqueue(item any) {
	queue.rwmutex.Lock()
	defer queue.rwmutex.Unlock()
	queue.slice = append(queue.slice, item)
}

func (queue *ConcurrentQueue) Peek() any {
	return queue.ElementAt(0)
}

func (queue *ConcurrentQueue) TryPeek() (any, error) {
	return queue.TryElementAt(0)
}

func (queue *ConcurrentQueue) ElementAt(index int) any {
	queue.rwmutex.RLock()
	defer queue.rwmutex.RUnlock()
	return queue.slice[index]
}

func (queue *ConcurrentQueue) TryElementAt(index int) (any, error) {
	queue.rwmutex.RLock()
	defer queue.rwmutex.RUnlock()
	if queue.Len() >= index {
		return queue.slice[index], nil
	}
	return nil, fmt.Errorf("no element in queue at index %v", index)
}

func (queue *ConcurrentQueue) Len() int {
	queue.rwmutex.RLock()
	defer queue.rwmutex.RUnlock()
	return len(queue.slice)
}

func (queue *ConcurrentQueue) IsEmpty() bool {
	return queue.Len() == 0
}

func (queue *ConcurrentQueue) ToSlice() []any {
	queue.rwmutex.RLock()
	defer queue.rwmutex.RUnlock()
	slice := make([]any, queue.Len())
	copy(slice, queue.slice)
	return slice
}

func (queue *ConcurrentQueue) CopyTo(dst []any) {
	queue.rwmutex.RLock()
	defer queue.rwmutex.RUnlock()
	copy(dst, queue.slice)
}
