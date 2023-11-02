//date: 2023-11-02T16:33:56Z
//url: https://api.github.com/gists/be12e78061eddcf69b89d56587641a87
//owner: https://api.github.com/users/mmtdebojit

package iterator

import "fmt"

//faciliates the traversal of some data structure

type Collection interface {
	getIterator(string) Iterator
}

type Iterator interface {
	getNext() *Node
	hasMore() bool
}

type Node struct {
	data        int64
	left, right *Node
}

func NewNode(data int64) *Node {
	return &Node{data, nil, nil}
}

//implements Collection interface
type Tree struct {
	root *Node
}

func (tree *Tree) getIterator(traversalType string) Iterator {
	switch traversalType {
	case "bfs":
		return NewBfsIterator(tree)
	case "random":
		return NewRandomIterator(tree)
	}
	//default
	return NewRandomIterator(tree)
}

type BfsIterator struct {
	queue Queue
	tree  *Tree
}

func (bfs *BfsIterator) getNext() *Node {
	if bfs.queue.IsEmpty() {
		return nil
	}
	nextNode := bfs.queue.Dequeue().(*Node)
	if nextNode.left != nil {
		bfs.queue.Enqueue(nextNode.left)
	}
	if nextNode.right != nil {
		bfs.queue.Enqueue(nextNode.right)
	}
	return nextNode
}

func (bfs *BfsIterator) hasMore() bool {
	return !bfs.queue.IsEmpty()
}

func NewBfsIterator(tree *Tree) *BfsIterator {
	queue := Queue{}
	queue.Enqueue(tree.root)
	return &BfsIterator{queue, tree}
}

type RandomIterator struct {
	stack Stack
	tree  *Tree
}

func (random *RandomIterator) getNext() *Node {
	if random.stack.IsEmpty() {
		return nil
	}
	nextNode := random.stack.Pop().(*Node)
	if nextNode.left != nil {
		random.stack.Push(nextNode.left)
	}
	if nextNode.right != nil {
		random.stack.Push(nextNode.right)
	}
	return nextNode
}

func (random *RandomIterator) hasMore() bool {
	return !random.stack.IsEmpty()
}

func NewRandomIterator(tree *Tree) *RandomIterator {
	stack := Stack{}
	stack.Push(tree.root)
	return &RandomIterator{stack, tree}
}

func ClientCode() {
	root := NewNode(1)
	root.left = NewNode(2)
	root.right = NewNode(3)
	root.left.left = NewNode(4)
	root.left.right = NewNode(5)
	root.right.left = NewNode(6)
	root.right.right = NewNode(7)
	tree := Tree{root}

	fmt.Println("BFS Iterator...")
	bfsIterator := tree.getIterator("bfs")
	for bfsIterator.hasMore() {
		currentNode := bfsIterator.getNext()
		fmt.Printf("%d, ", currentNode.data)
	}
	fmt.Println("\n")

	fmt.Println("Random Iterator...")
	dfsIterator := tree.getIterator("random")
	for dfsIterator.hasMore() {
		currentNode := dfsIterator.getNext()
		fmt.Printf("%d, ", currentNode.data)
	}
	fmt.Println("\n")
}
