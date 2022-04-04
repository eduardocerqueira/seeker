//date: 2022-04-04T17:11:17Z
//url: https://api.github.com/gists/bf66a7371bee2a10ff313055860137e1
//owner: https://api.github.com/users/aktuba

package main

import "fmt"

type CacheInterface interface {
	has(key string) bool
	get(key string) string
	set(key, value string) bool
	delete(key string) bool
}

type cacheItem struct {
	key   string
	value string
	next  *cacheItem
	prev  *cacheItem
}

type cacheMap = map[string]cacheItem

type Cache struct {
	cap   int
	cache cacheMap
	first *cacheItem
	last  *cacheItem
}

func (c *Cache) has(key string) bool {
	_, ok := c.cache[key]
	return ok
}

func (c *Cache) get(key string) string {
	if item, ok := c.cache[key]; ok {
		if item.prev != nil {
			item.prev.next = item.next
		}

		if item.next != nil {
			item.next.prev = item.prev
		}

		item.prev = c.last
		item.next = nil

		if c.last != nil {
			c.last.next = &item
		}
		c.last = &item

		return item.value
	}
	return ""
}

func (c *Cache) set(key, value string) bool {
	if len(c.cache)+1 > c.cap && !c.delete(c.first.key) {
		return false
	}

	item := cacheItem{
		key:   key,
		value: value,
		prev:  c.last,
		next:  nil,
	}

	if c.first == nil {
		c.first = &item
		c.last = c.first
	} else {
		c.last.next = &item
		c.last = &item
	}

	c.cache[key] = item

	return true
}

func (c *Cache) delete(key string) bool {
	item := c.cache[key]

	if item.prev != nil {
		item.prev.next = item.next
	} else {
		c.first = item.next
	}

	if item.next != nil {
		item.next.prev = item.prev
	} else {
		c.last = item.prev
	}

	delete(c.cache, key)

	return true
}

func newCache(cap int) CacheInterface {
	if cap == 0 {
		cap = 1000
	}

	cache := new(Cache)
	cache.cap = cap
	cache.cache = make(cacheMap, cap)
	cache.first = nil
	cache.last = nil
	return cache
}

func main() {
	cache := newCache(2)

	cache.set("key1", "value1")
	cache.set("key2", "value2")
	cache.set("key3", "value3")

	fmt.Println("key1: ", cache.has("key1"))
	fmt.Println("key2: ", cache.has("key2"))
	fmt.Println("key3: ", cache.has("key3"))

	cache.delete("key3")
	fmt.Println("key1: ", cache.has("key1"))
	fmt.Println("key2: ", cache.has("key2"))
	fmt.Println("key3: ", cache.has("key3"))
	fmt.Println("value for key2: ", cache.get("key2"))

	cache.set("key4", "value4")
	fmt.Println("key1: ", cache.has("key1"))
	fmt.Println("key2: ", cache.has("key2"))
	fmt.Println("key3: ", cache.has("key3"))
	fmt.Println("key4: ", cache.has("key4"))
	fmt.Println("value for key2: ", cache.get("key2"))
	fmt.Println("value for key4: ", cache.get("key4"))

	_ = cache.get("key2")       // подняли
	cache.set("key5", "value5") // тут key4 должен исчезнуть
	fmt.Println("key1: ", cache.has("key1"))
	fmt.Println("key2: ", cache.has("key2"))
	fmt.Println("key3: ", cache.has("key3"))
	fmt.Println("key4: ", cache.has("key4"))
	fmt.Println("key5: ", cache.has("key5"))
}
