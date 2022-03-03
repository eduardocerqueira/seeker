//date: 2022-03-03T16:57:01Z
//url: https://api.github.com/gists/99358e2529a5887ae13a936a7f1c8ae5
//owner: https://api.github.com/users/qdm12

package potato

import "fmt"

type Potato uint8

const (
	APotato Potato = iota
)

type Frie struct {
	Cooked bool
}

type Cutter interface {
	Cut(potato Potato) (fries []Frie, err error)
}

type Fryer interface {
	Fry(uncookedFries []Frie) (cookedFries []Frie, err error)
}

func CutAndFry(cutter Cutter, fryer Fryer, potatoes []Potato) (fries []Frie, err error) {
	for _, potato := range potatoes {
		uncookedFries, err := cutter.Cut(potato)
		if err != nil {
			return nil, fmt.Errorf("cannot cut potato: %w", err)
		}

		cookedFries, err := fryer.Fry(uncookedFries)
		if err != nil {
			return nil, fmt.Errorf("cannot fry raw fries: %w", err)
		}

		fries = append(fries, cookedFries...)
	}

	return fries, nil
}

// ==================================
// ==================================
// ==================================
// Part 2 mocks returning other mocks
// ==================================
// ==================================
// ==================================

type Fetcher interface {
	FetchTools() (cutter Cutter, fryer Fryer, err error)
}

func MakeFries(fetcher Fetcher, potatoes []Potato) (fries []Frie, err error) {
	cutter, fryer, err := fetcher.FetchTools()
	if err != nil {
		return nil, fmt.Errorf("cannot get our tools: %w", err)
	}

	return CutAndFry(cutter, fryer, potatoes)
}
