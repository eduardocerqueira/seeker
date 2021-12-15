//date: 2021-12-15T17:00:36Z
//url: https://api.github.com/gists/92b8ee3da3d5623eebf0d8341ca5ad89
//owner: https://api.github.com/users/quii

package main

import "fmt"

func main() {
	someNumbers := []int64{1, 2, 3}
	someNames := []string{"John", "Chris", "Mary"}

	fmt.Println(Reduce(someNumbers, 0, func(a, b int64) int64 {
		return a + b
	}))

	fmt.Println(Reduce(someNames, "", func(a, b string) string {
		return a + " " + b
	}))

	fmt.Println("lets find names beginning with J", Find(someNames, func(name string) bool {
		return name[0] == 'J'
	}))

	//demo, show map (shout names)
	fmt.Println("lets shout names", Map(someNames, func(name string) string {
		return name + "!"
	}))

}

func Map[T any](items []T, f func(T) T) []T {
	result := make([]T, len(items))
	for i, v := range items {
		result[i] = f(v)
	}
	return result
}

func Reduce[T any](numbers []T, initialValue T, fn func(T, T) T) T {
	var result T
	result = initialValue
	for _, number := range numbers {
		result = fn(result, number)
	}
	return result
}

func Find[T any](haystack []T, predicate func(T) bool) []T {
	var result []T
	for _, x := range haystack {
		if predicate(x) {
			result = append(result, x)
		}
	}
	return result
}
