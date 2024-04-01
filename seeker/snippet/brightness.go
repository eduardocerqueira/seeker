//date: 2024-04-01T16:53:33Z
//url: https://api.github.com/gists/6966957459fb1aa0a174bf1061beb9d0
//owner: https://api.github.com/users/mattismoel

package img

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"math"
)

type Pixel struct {
	R, G, B, A uint8
}

type Pixels [][]Pixel

// Converts RGB value into floating point representation in interval [0.0; 1.0]
//
// The input values are expected to be in interval [0;255].
func (p Pixel) Decimal() (float64, float64, float64) {
	vr := float64(p.R) / 255.0
	vg := float64(p.G) / 255.0
	vb := float64(p.B) / 255.0

	return vr, vg, vb
}

// Gets the pixels of an input [io.Reader], and returns it as a [Pixels] struct.
//
// For using the manipulations methods of the [Pixels] struct, resize the
// reader (image) to a lower resolution.
func GetPixels(f io.Reader) (Pixels, error) {
	image.RegisterFormat("jpeg", "jpeg", jpeg.Decode, jpeg.DecodeConfig)
	image.RegisterFormat("png", "png", png.Decode, png.DecodeConfig)
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("could not decode input file: %v", err)
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	var pixels [][]Pixel
	for y := bounds.Min.Y; y < height; y++ {
		var row []Pixel
		for x := bounds.Min.X; x < width; x++ {
			row = append(row, rgbaToPixel(img.At(x, y).RGBA()))
		}
		pixels = append(pixels, row)
	}
	return pixels, nil
}

// Returns the perceived brightness of a set of [Pixels].
func (pxls Pixels) Brightness() (float64, error) {
	avgR, avgG, avgB, _ := pxls.Average()
	brightness := math.Sqrt(
		0.241*math.Sqrt(float64(avgR)) +
			0.691*math.Pow(float64(avgG), 2) +
			0.068*math.Pow(float64(avgB), 2),
	)

	return brightness, nil
}

// Returns the RGBA components individual average color.
func (pxls Pixels) Average() (uint8, uint8, uint8, uint8) {
	width := len(pxls)
	height := len(pxls[0])
	var pxCount = width * height

	var avgR, avgG, avgB, avgA int
	for y := range height - 1 {
		for x := range width - 1 {
			p := pxls[x][y]
			avgR += int(p.R)
			avgG += int(p.G)
			avgB += int(p.B)
			avgA += int(p.A)
		}
	}

	r := uint32(avgR) / uint32(pxCount)
	g := uint32(avgG) / uint32(pxCount)
	b := uint32(avgB) / uint32(pxCount)
	a := uint32(avgA) / uint32(pxCount)

	return uint8(r), uint8(g), uint8(b), uint8(a)

}

func rgbaToPixel(r, g, b, a uint32) Pixel {
	return Pixel{uint8(r / 257), uint8(g / 257), uint8(b / 257), uint8(a / 257)}
}
