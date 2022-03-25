//date: 2022-03-25T17:03:45Z
//url: https://api.github.com/gists/2594b48376a1a949e3e2d459bb84d0ca
//owner: https://api.github.com/users/rbranson

type ary[T any] struct {
	members [arraySize]T
	len     uint16
}

func (a *ary[T]) length() int {
	return int(a.len)
}

func (a *ary[T]) set(index int, item T) {
	if asserts && index >= int(a.len) {
		panic(fmt.Sprintf("index %v beyond len %v", index, a.len))
	}
	a.members[index] = item
}

func (a *ary[T]) get(index int) T {
	if asserts && index >= int(a.len) {
		panic(fmt.Sprintf("index %v beyond len %v", index, a.len))
	}
	return a.members[index]
}

func (a *ary[T]) ptr(index int) *T {
	if asserts && index >= int(a.len) {
		panic(fmt.Sprintf("index %v beyond len %v", index, a.len))
	}
	return &a.members[index]
}

func (a *ary[T]) zeroRange(from int, to int) {
	if to < from {
		return
	}
	var empty [arraySize]T
	copy(a.members[from:to], empty[:])
}

func (a *ary[T]) zero(index int) {
	var empty T
	a.members[index] = empty
}

func (a *ary[T]) insertAt(index int, item T) {
	if asserts {
		if index > int(a.len)+1 {
			panic(fmt.Sprintf("index %v beyond len %v+1", index, a.len))
		}
	}
	if index != int(a.len) {
		copy(a.members[index+1:a.len+1], a.members[index:a.len])
	}
	a.members[index] = item
	a.len++
}

func (a *ary[T]) removeAt(index int) (removed T, ok bool) {
	if asserts && index < 0 {
		panic(fmt.Sprintf("index %v < 0", index))
	}
	if index >= int(a.len) {
		return
	}
	if asserts && a.len <= 0 {
		panic(fmt.Sprintf("a.len %v <= 0", a.len))
	}
	removed, ok = a.members[index], true
	maxIdx := int(a.len) - 1
	if index < maxIdx {
		copy(a.members[index:maxIdx], a.members[index+1:maxIdx+1])
	}
	// zero calls are so pointers don't get hidden beyond the len and never GC'd
	a.zero(maxIdx)
	a.len--
	return
}

func (a *ary[T]) pop() (removed T, ok bool) {
	if a.len == 0 {
		return
	}
	idx := int(a.len) - 1
	removed, ok = a.members[idx], true
	a.zero(idx)
	a.len--
	return
}

func (a *ary[T]) truncate(length int) {
	if asserts && length < 0 {
		panic(fmt.Sprintf("truncate length %v < 0", length))
	}
	a.zeroRange(length, a.length())
	a.len = uint16(length)
}

func (a *ary[T]) copyResize(offset int, src *ary[T], from, to int) {
	cnt := to - from
	newLength := offset + cnt
	if asserts {
		if offset > int(a.len) {
			panic(fmt.Sprintf("offset %v > a.len %v", offset, a.len))
		}
		if newLength > len(a.members) {
			panic(fmt.Sprintf("endOffset %v >= len(a.members) %v", newLength, len(a.members)))
		}
		if to < from {
			panic(fmt.Sprintf("to %v < from %v", to, from))
		}
		if from >= int(src.len) {
			panic(fmt.Sprintf("from %v >= src.len %v", from, src.len))
		}
		if to > int(src.len) {
			panic(fmt.Sprintf("to %v >= src.len %v", to, src.len))
		}
	}
	copy(a.members[offset:newLength], src.members[from:to])
	a.truncate(newLength)
}

func (a *ary[T]) append(values ...T) {
	if asserts {
		newLen := int(a.len) + len(values)
		if newLen > len(a.members) {
			panic(fmt.Sprintf("newLen %v > len(a.members) %v", newLen, len(a.members)))
		}
	}
	copy(a.members[a.len:int(a.len)+len(values)], values)
	a.len += uint16(len(values))
}

func (a *ary[T]) toSlice() []T {
	out := make([]T, a.len)
	copy(out, a.members[:a.len])
	return out
}
