//date: 2023-11-02T16:33:56Z
//url: https://api.github.com/gists/be12e78061eddcf69b89d56587641a87
//owner: https://api.github.com/users/mmtdebojit

package iterator

import (
	"fmt"
)

type Queue struct {
	items []interface{}
}

func (q *Queue) Enqueue(item interface{}) {
	q.items = append(q.items, item)
}

func (q *Queue) Dequeue() interface{} {
	if len(q.items) == 0 {
		return nil
	}
	item := q.items[0]
	q.items = q.items[1:]
	return item
}

func (q *Queue) IsEmpty() bool {
	return len(q.items) == 0
}

func (q *Queue) Size() int {
	return len(q.items)
}

func main() {
	queue := Queue{}

	queue.Enqueue(1)
	queue.Enqueue(2)
	queue.Enqueue(3)

	fmt.Println("Queue size:", queue.Size())

	for !queue.IsEmpty() {
		item := queue.Dequeue()
		fmt.Println("Dequeued:", item)
	}

	fmt.Println("Queue size:", queue.Size())
}
