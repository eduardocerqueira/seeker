//date: 2026-01-08T17:20:17Z
//url: https://api.github.com/gists/aec4874952ea84d40120f98351ab5421
//owner: https://api.github.com/users/retpolanne

type LRUCache struct {
    cache map[int]int
    lruHeap *MinHeap
    capacity int
    currentSize int
}


func Constructor(capacity int) LRUCache {
    cache := make(map[int]int)
    minHeap := &MinHeap{}
    minHeap.create()
    return LRUCache{
        cache: cache,
        lruHeap: minHeap,
        capacity: capacity,
        currentSize: 0,
    }
}


func (this *LRUCache) Get(key int) int {
    if val, found := this.cache[key]; found {
        idx := this.lruHeap.search(key)
        this.lruHeap.heap[idx].lastAccessTs = int(time.Now().UnixNano())
        // Since we've accessed recently, we need to move this down. 
        this.lruHeap.moveDown(idx)
        return val
    }
    return -1
}


func (this *LRUCache) Put(key int, value int)  {
    if _, found := this.cache[key]; found {
        this.cache[key] = value
        // Move down since we've updated the item. 
        idx := this.lruHeap.search(key)
        this.lruHeap.heap[idx].lastAccessTs = int(time.Now().UnixNano())
        this.lruHeap.moveDown(idx)
    } else {
        if this.currentSize < this.capacity {
            this.cache[key] = value
            this.lruHeap.add(key)
            this.currentSize++
        } else {
            // LRU cache evict logic here. 
            fmt.Printf("Evicting to add %d\n", key)
            ref := this.lruHeap.pop()
            delete(this.cache, ref.key)
            fmt.Printf("Evicted key %d\n", ref.key)
            this.cache[key] = value
            this.lruHeap.add(key)
        }
    }
}

// This is where we'll store keys alongside an access counter
type MapRef struct {
    key int
    lastAccessTs int
}

type MinHeap struct {
    heap []*MapRef
    heapTail int
}

func (m *MinHeap) create() {
    m.heapTail = 0
    m.heap = []*MapRef{}
}

func (m *MinHeap) isLeaf(p int) bool {
    return m.leftChildIndex(p) > m.heapTail
}

func (m *MinHeap) parent(p int) int {
    return (p - 1) / 2
}

func (m *MinHeap) leftChildIndex(p int) int {
    return 2 * p + 1
}

func (m *MinHeap) rightChildIndex(p int) int {
    return 2 * p + 2
}

func (m *MinHeap) swap(s, d int) {
    tmp := m.heap[d]
    m.heap[d] = m.heap[s]
    m.heap[s] = tmp
}

// This will help us keep the heap property.
// It starts comparing the current item with its parent. 
// Reminder:
// -1 if x is less than y,
// 0 if x equals y,
// +1 if x is greater than y.
// So, if current is smaller than the parent, we swap them. 
// We now change the current pointer to point to the parent, 
// and parent to point to the parent of the current pointer (which went up). 
// We do this until p == 0. 
func (m *MinHeap) moveUp(p int) {
    parent := m.parent(p)
    for {
        if cmp.Compare(m.heap[p].lastAccessTs, m.heap[parent].lastAccessTs) < 0 && p > 0 {
            m.swap(p, parent)
            p = parent
            parent = m.parent(p)
        } else {
            break
        }
    }
}

// We'll use this alongside the pop function to remove the item from the top of the heap. 
func (m *MinHeap) moveDown(p int) {
    counter := 0
    for {
        tmp := -1
        leftIndex := m.leftChildIndex(p)
        rightIndex := m.rightChildIndex(p)
        // We check if there's a right child
        if rightIndex < m.heapTail {
            // There's a right child, so we need to get the smallest between
            // left and right and then compare with parent.
            if cmp.Compare(m.heap[leftIndex].lastAccessTs, m.heap[rightIndex].lastAccessTs) < 0 {
                // Left is smaller
                tmp = leftIndex
            } else {
                // Right is smaller
                tmp = rightIndex
            }
            if cmp.Compare(m.heap[tmp].lastAccessTs, m.heap[p].lastAccessTs) < 0 {
                m.swap(tmp, p)
                p = tmp
            }

        } else if leftIndex < m.heapTail {
            // There's a left child, but no right child
            if cmp.Compare(m.heap[leftIndex].lastAccessTs, m.heap[p].lastAccessTs) < 0 {
                m.swap(leftIndex, p)
                p = leftIndex
            } else {
                // p is already the smallest
                break
            }
        } else {
            // We probably have a leaf!
            break
        }
        // If we didn't fall into these if clauses, that means parent is probably the smallest. 
        if p >= m.heapTail - 1 || counter >= m.heapTail - 1 {
            break
        }
        counter++
    }
}


// This is a temp 0(n) function to search for a key on the tree. 
// I'll put this here while I search for a more efficient way to find values. 
func (m *MinHeap) search(k int) int {
    for i := 0; i < m.heapTail; i++ {
        if m.heap[i].key == k {
            return i
        }
    }
    return -1
}

func (m *MinHeap) add(i int) {
    mapRef := &MapRef{
        key: i,
        lastAccessTs: int(time.Now().UnixNano()),
    }
    m.heap = append(m.heap, mapRef)
    m.heapTail++
    m.moveUp(m.heapTail - 1)
}

func (m *MinHeap) pop() *MapRef {
    res := m.heap[0]
    m.heapTail--
    m.heap[0] = m.heap[m.heapTail]
    m.heap = m.heap[:m.heapTail]
    m.moveDown(0)
    return res
}


/**
 * Your LRUCache object will be instantiated and called as such:
 * obj := Constructor(capacity);
 * param_1 := obj.Get(key);
 * obj.Put(key,value);
 */