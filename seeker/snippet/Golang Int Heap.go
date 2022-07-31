//date: 2022-07-31T03:12:07Z
//url: https://api.github.com/gists/06f354190568313e177d5ed0f274d3dc
//owner: https://api.github.com/users/evlic

// IntHeap 实现堆
type IntHeap []int

func (h IntHeap) Len() int { return len(h) }

func (h IntHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

// 大顶堆 如果分数相同，Name 字典序小的更大
func (h IntHeap) Less(i, j int) bool {
	return h[i] < h[j]
}

// Push 放入尾部，并调整
func (h *IntHeap) Push(v interface{}) {
	*h = append(*h, v.(int))
}

// Pop 删除最后一个
func (h *IntHeap) Pop() interface{} {
	a := *h
	v := a[len(a)-1]
	*h = a[:len(a)-1]
	return v
}