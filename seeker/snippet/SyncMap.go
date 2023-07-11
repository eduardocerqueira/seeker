//date: 2023-07-11T17:04:46Z
//url: https://api.github.com/gists/ad00b4ed1cd878a6da8c1b0329915068
//owner: https://api.github.com/users/mniak

package utils

import "sync"

type SyncMap[K comparable, V any] struct {
	inner sync.Map
}

func (m *SyncMap[K, V]) convertValue(anyval any) V {
	if anyval == nil {
		var empty V
		return empty
	}
	return anyval.(V)
}

func (m *SyncMap[K, V]) Load(key K) (value V, ok bool) {
	anyval, ok := m.inner.Load(key)
	return m.convertValue(anyval), ok
}

func (m *SyncMap[K, V]) Store(key K, value V) {
	m.inner.Store(key, value)
}

func (m *SyncMap[K, V]) LoadOrStore(key K, value V) (V, bool) {
	anyactual, loaded := m.inner.LoadOrStore(key, value)
	return m.convertValue(anyactual), loaded
}

func (m *SyncMap[K, V]) LoadAndDelete(key K) (value V, loaded bool) {
	anyval, loaded := m.inner.LoadAndDelete(key)
	return m.convertValue(anyval), loaded
}

func (m *SyncMap[K, V]) Delete(key K) {
	m.inner.Delete(key)
}

func (m *SyncMap[K, V]) Swap(key K, value V) (previous V, loaded bool) {
	anyprevious, loaded := m.inner.Swap(key, value)
	return m.convertValue(anyprevious), loaded
}

func (m *SyncMap[K, V]) CompareAndSwap(key K, old, new V) bool {
	return m.inner.CompareAndSwap(key, old, new)
}

func (m *SyncMap[K, V]) CompareAndDelete(key K, old V) (deleted bool) {
	return m.inner.CompareAndDelete(key, old)
}

func (m *SyncMap[K, V]) Range(fn func(key K, value V) bool) {
	m.inner.Range(func(anykey, anyval any) bool {
		key := anykey.(K)
		val := m.convertValue(anyval)
		return fn(key, val)
	})
}
