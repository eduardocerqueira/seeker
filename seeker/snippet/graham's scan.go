//date: 2022-05-02T17:12:07Z
//url: https://api.github.com/gists/edab6f920fd372ae506a29d2d654ce0f
//owner: https://api.github.com/users/keshavchand

package main

import (
	"fmt"
	"log"
	_ "log"
	"math/rand"
	"sort"
	"time"
)

type Point struct {
	x, y float32
}

type PointByXY []Point

func (p PointByXY) Print() {
	for idx, i := range p {
		if idx > 0 {
			fmt.Printf(", ")
		}
		fmt.Printf("(%f, %f)", i.x, i.y)
	}
	fmt.Println("")
}

func (p PointByXY) Len() int {
	return len(p)
}

func (p PointByXY) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func (p PointByXY) Less(i, j int) bool {
	if p[i].x < p[j].x {
		return true
	}
	if p[i].x == p[j].x {
		return p[i].y < p[j].y
	}
	return false
}

func main() {
  t := time.Now()
	rand.Seed(t.UnixNano())
	points := make([]Point, 0, 1000)
	for i := 0; i < 1000; i++ {
		p := Point{rand.Float32()*255 - 100, rand.Float32()*255 - 100}
		points = append(points, p)
	}
	sort.Sort(PointByXY(points))

	wrapped := Wrap(points)
	PointByXY(wrapped).Print()
	log.Printf("%0.2f%%", float32(len(wrapped))*100.0/float32(len(points)))
}

func Wrap(points []Point) []Point {
	if len(points) <= 3 {
		return points
	}
	upper := generateUpper(points)
	lower := generateLower(points)

	for i := 0; i < len(lower)/2; i++ {
		lower[i], lower[len(lower)-1-i] = lower[len(lower)-1-i], lower[i]
	}

	result := append(upper, lower[1:len(lower)-1]...)
	return result
}

func generateUpper(points []Point) []Point {
	temp1 := make([]Point, len(points)) // TODO: what if it is very large
	shouldSwap := func(p1, p2, p3 Point) bool {
		return ((p2.x-p1.x)*(p3.y-p1.y) - (p2.y-p1.y)*(p3.x-p1.x)) >= 0
	}
	// Upper Part
	copy(temp1, points[0:2])
	writer := 2

	for i := 2; i < len(points); i++ {
		temp1[writer] = points[i]
		p1, p2, p3 := temp1[writer-2], temp1[writer-1], temp1[writer]
		for writer >= 2 && shouldSwap(p1, p2, p3) {
			writer -= 1
			temp1[writer] = temp1[writer+1]
			if writer < 2 {
				break
			}
			p1, p2, p3 = temp1[writer-2], temp1[writer-1], temp1[writer]
		}
		writer += 1
	}
	return (temp1[:writer])
}

func generateLower(points []Point) []Point {
	temp2 := make([]Point, len(points)) // TODO: what if it is very large
	shouldSwap := func(p1, p2, p3 Point) bool {
		return ((p2.x-p1.x)*(p3.y-p1.y) - (p2.y-p1.y)*(p3.x-p1.x)) <= 0
	}
	// Lower Part
	copy(temp2, points[0:2])
	writer := 2

	for i := 2; i < len(points); i++ {
		temp2[writer] = points[i]
		p1, p2, p3 := temp2[writer-2], temp2[writer-1], temp2[writer]
		for writer >= 2 && shouldSwap(p1, p2, p3) {
			writer -= 1
			temp2[writer] = temp2[writer+1]
			if writer < 2 {
				break
			}
			p1, p2, p3 = temp2[writer-2], temp2[writer-1], temp2[writer]
		}
		writer += 1
	}

	return (temp2[:writer])
}
