//date: 2024-01-08T17:09:20Z
//url: https://api.github.com/gists/e7199f5793d103891527fdeff4530814
//owner: https://api.github.com/users/cruzelx

func (dll *DoublyLinkedList[K, V]) Remove(node *Node[K, V]) {

	if node == dll.Head {
		dll.Head = node.Next
	}
	if node == dll.Tail {
		dll.Tail = node.Prev
	}
	if node.Prev != nil {
		node.Prev.Next = node.Next
	}
	if node.Next != nil {
		node.Next.Prev = node.Prev
	}
	node.Prev = nil
	node.Next = nil

}
