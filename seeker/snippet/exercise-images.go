//date: 2025-11-21T16:49:42Z
//url: https://api.github.com/gists/6811a20426a8de2906c2f4ec3e36e773
//owner: https://api.github.com/users/monooso

package main

import (
	"golang.org/x/tour/pic"
	"image"
	"image/color"
)

type Image struct{}

func (i Image) ColorModel() color.Model {
	return color.RGBAModel
}

func (i Image) Bounds() image.Rectangle {
	return image.Rect(0, 0, 250, 250)
}

func (i Image) At(x, y int) color.Color {
	red := uint8((x^x) % 255)
	green := uint8((x^y) % 255)
	
	return color.RGBA{red, green, 255, 255}
}

func main() {
	m := Image{}
	pic.ShowImage(m)
}