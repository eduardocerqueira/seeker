//date: 2024-01-08T17:01:17Z
//url: https://api.github.com/gists/8ecad37ba212ef797edb79c3d716c23a
//owner: https://api.github.com/users/cruzelx

func (dll *DoublyLinkedList[K, V]) Prepend(node *Node[K, V]) {

	// If the DLL is empty
	if dll.Head == nil {
		dll.Head, dll.Tail = node, node
		return
	}

	node.Next = dll.Head
	node.Prev = nil
	dll.Head.Prev = node
	dll.Head = node

}