//date: 2022-07-31T03:12:20Z
//url: https://api.github.com/gists/fc66e341dda7d97fec554456410671e8
//owner: https://api.github.com/users/evlic

type Custom struct {
    
}
// IntHeap 实现堆
type CustomHeap []*Custom

func (h CustomHeap) Len() int { 
    return len(h) 
}

func (h IntHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

// 大顶堆 如果分数相同，Name 字典序小的更大
func (h IntHeap) Less(i, j int) bool {
	return 
}

// Push 放入尾部，并调整
func (h *IntHeap) Push(v interface{}) {
	*h = append(*h, v.(*Custom))
}

// Pop 删除最后一个
func (h *IntHeap) Pop() interface{} {
	a := *h
	v := a[len(a)-1]
	*h = a[:len(a)-1]
	return v
}