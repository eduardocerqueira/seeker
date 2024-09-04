//date: 2024-09-04T16:45:56Z
//url: https://api.github.com/gists/a7d946707b8d8a7f235895dc0cc2b569
//owner: https://api.github.com/users/tiagoncardoso

import (
	"fmt"
	"math"
)

type Shape interface {
	Area() float64
}

type Rectangle struct {
	Width float64
	Height float64
}

type Circle struct {
	Radius float64
}

type Line struct {
	Length float64
}

func (r Rectangle) Area() float64 {
	return r.Width * r.Height
}

func (c Circle) Area() float64 {
	return math.Pi * c.Radius * c.Radius
}

func (l Line) Area() float64 {
	return 0
}

func CalculateArea(s Shape) {
	fmt.Printf("Area: %2f\n", s.Area())
}

func main() {
	r := Rectangle{Width: 10, Height: 20}
	c := Circle{Radius: 10}
	l := Line{Length: 10}

	CalculateArea(r)
	CalculateArea(c)
	CalculateArea(l)
}
