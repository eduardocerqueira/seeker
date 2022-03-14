//date: 2022-03-14T17:07:36Z
//url: https://api.github.com/gists/54554ab385bcec53ab28ba0707767a8b
//owner: https://api.github.com/users/prembhaskal

package parallel

import "sync"

type clientsync struct {
	val int
}

func (c *clientsync) Add(x int) int {
	return c.val + x
}

type librarySync struct {
	c *clientsync
	mx sync.Mutex
}

func NewLibrarySync() *librarySync {
	c := &clientsync{
		val: 0,
	}

	return &librarySync{
		c: c,
	}
}

func (l *librarySync) C() *clientsync {
	l.mx.Lock()
	defer l.mx.Unlock()
	return l.c
}

func (l *librarySync) UpdateClient() {
	l.mx.Lock()
	defer l.mx.Unlock()
	newVal := l.c.val + 1
	l.c = &clientsync{newVal}
}
