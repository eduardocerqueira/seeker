//date: 2022-10-18T17:31:36Z
//url: https://api.github.com/gists/5c6b8afc13ab34ebbb2bf4bdd6c3f70f
//owner: https://api.github.com/users/PedroGaletti

/*
Queue - Standard

It's like a Stack, but in the Stack, we point to the last item that is entered the Stack.
In the Queue, we will point to the first item that is entered the Queue.
*/

package main

import "fmt"

type Queue struct {
	Head *Node
	Tail *Node
}

type Node struct {
	Val  string
	Next *Node
}

func (q *Queue) Enqueue(name string) {
	node := Node{Val: name} // Creating a new node

	if q.Head == nil { // Check if my Queue is empty, if is empty the element will be the head
		q.Head = &node
		q.Tail = &node
	} else { // If head is not empty, the Tail.Next and the Tail will receive the new node
		q.Tail.Next = &node
		q.Tail = &node
	}
}

func (q *Queue) Dequeue() string {
	if q.Head == nil { // If Queue is empty return ""
		return ""
	}

	node := q.Head       // node will receive the Head -- (Remove the first item that is entered in the queue)
	q.Head = q.Head.Next // Head will receive the next node of Head

	if q.Head == nil { // If Head is nil, queue is empty, so remove the Tail
		q.Tail = nil
	}

	return node.Val
}

func main() {
	queue := Queue{}

	queue.Enqueue("Pedro")
	queue.Enqueue("John")
	queue.Enqueue("Derek")
	queue.Enqueue("Clint")
	queue.Enqueue("Brian")

	fmt.Println(queue.Dequeue())
	fmt.Println(queue.Dequeue())
	fmt.Println(queue.Dequeue())
	fmt.Println(queue.Dequeue())
	fmt.Println(queue.Dequeue())
}
