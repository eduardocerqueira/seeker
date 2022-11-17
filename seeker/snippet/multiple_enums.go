//date: 2022-11-17T17:11:40Z
//url: https://api.github.com/gists/12397cb6df62883fa7645108ddd724c6
//owner: https://api.github.com/users/robertsmoto

package main

import "fmt"

type Season int
type Direction int
type Color int

const (
	Spring Season = iota
	Summer
	Winter
	Fall
	North Direction = iota
	South
	East
	West
	Red Color = iota
	White
	Blue
)

func main() {
	fmt.Printf("Summer %[1]d, %[1]T", Summer)
	fmt.Printf("\nFall %[1]d, %[1]T", Fall)
	fmt.Printf("\nSouth %[1]d, %[1]T", South)
	fmt.Printf("\nBlue %[1]d, %[1]T", Blue)
}