//date: 2024-12-17T16:51:22Z
//url: https://api.github.com/gists/8864899dbd8f490140a1c38b809bc91a
//owner: https://api.github.com/users/alexandreLamarre

package main

import (
	"fmt"
	"testing"
)

func main() {
	resFewSmall := testing.Benchmark(BenchmarkFewSmall)
	ppBench("Elems : 10, Size : 10", resFewSmall)
	resManySmall := testing.Benchmark(BenchmarkManySmall)
	ppBench("Elems : 10000, Size : 10", resManySmall)
	resFewBig := testing.Benchmark(BenchmarkFewBig)
	ppBench("Elems : 10, Size : 10000", resFewBig)
	resManyBig := testing.Benchmark(BenchmarkManyBig)
	ppBench("Elems : 10000, Size : 10000", resManyBig)

}

func ppBench(name string, res testing.BenchmarkResult) {
	fmt.Printf("Benchmark for : %s \n", name)
	fmt.Printf("Memory allocations : %d \n", res.MemAllocs)
	fmt.Printf("Number of bytes allocated: %d \n", res.Bytes)
	fmt.Printf("Number of runs: %d \n", res.N)
	fmt.Printf("Time taken: %s \n", res.T)
	fmt.Printf("=================\n")
}

type BufferElement struct {
	buf []byte
}

var (
	fewSmall  []BufferElement
	manySmall []BufferElement
	fewBig    []BufferElement
	manyBig   []BufferElement
)

func init() {
	fewSmall = setupBuffer(10, 10)
	manySmall = setupBuffer(10000, 10)
	fewBig = setupBuffer(10, 10000)
	manyBig = setupBuffer(10000, 10000)
}

func setupBuffer(n int, size int) []BufferElement {
	ret := make([]BufferElement, n)
	for i, _ := range ret {
		ret[i] = BufferElement{
			buf: make([]byte, size),
		}
	}
	return ret
}

func ShallowCopy(buf []BufferElement) {
	newBuf := make([]BufferElement, len(buf))
	for i, x := range buf {
		newBuf[i] = x
	}
}

func BenchmarkFewSmall(b *testing.B) {
	for n := 0; n < b.N; n++ {
		ShallowCopy(fewSmall)
	}
}

func BenchmarkManySmall(b *testing.B) {
	for n := 0; n < b.N; n++ {
		ShallowCopy(manySmall)
	}
}

func BenchmarkManyBig(b *testing.B) {
	for n := 0; n < b.N; n++ {
		ShallowCopy(manyBig)
	}
}

func BenchmarkFewBig(b *testing.B) {
	for n := 0; n < b.N; n++ {
		ShallowCopy(fewBig)
	}
}
