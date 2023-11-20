//date: 2023-11-20T17:03:13Z
//url: https://api.github.com/gists/ce9d0ea2c87402fabaa38d0a7d5941a3
//owner: https://api.github.com/users/rvflash

// Package slices defines various functions useful with slices of any type.
package slices

import "math"

// Chunk splits a slice into uniform chunks of the requested size.
func Chunk[V any](x []V, size int) [][]V {
	n := len(x)
	if n == 0 || size <= 0 {
		return nil
	}
	if n <= size {
		return [][]V{x}
	}
	var (
		res = make([][]V, 0, int(math.Ceil(float64(n)/float64(size))))
		end int
	)
	for i := 0; i < n; i += size {
		end = i + size
		if end > n {
			end = n
		}
		res = append(res, x[i:end])
	}
	return res
}
