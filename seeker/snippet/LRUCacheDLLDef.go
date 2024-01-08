//date: 2024-01-08T16:50:59Z
//url: https://api.github.com/gists/633afc2ba212b82136b935332005426e
//owner: https://api.github.com/users/cruzelx

type Node[K comparable, V any] struct {
	Key   K
	Value V

	// Time of creation or last access
	TimeStamp time.Time

	// Time-To-Live; keys expire if the current time > Timestamp + TTL
	TTL time.Duration

	Prev *Node[K, V]
	Next *Node[K, V]
}

type DoublyLinkedList[K comparable, V any] struct {
	Head *Node[K, V]
	Tail *Node[K, V]
}

func NewNode[K comparable, V any](key K, value V, ttl time.Duration) *Node[K, V] {
	return &Node[K, V]{
		Key:       key,
		Value:     value,
		TTL:       ttl,
		TimeStamp: time.Now(),
	}
}

func NewDLL[K comparable, V any]() *DoublyLinkedList[K, V] {
	return &DoublyLinkedList[K, V]{}
}