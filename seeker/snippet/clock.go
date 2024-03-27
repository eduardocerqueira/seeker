//date: 2024-03-27T17:04:12Z
//url: https://api.github.com/gists/3f8b8731054006d8fbd7b13b06d1b632
//owner: https://api.github.com/users/Farhaduneci

package main

import (
	"fmt"
)

type Clock struct {
	hours   int
	minutes int
}

func New(hours, minutes int) Clock {
	h := (hours + minutes/60) % 24
	m := minutes % 60

	if m < 0 {
		m += 60
		h -= 1
	}
	if h < 0 {
		h += 24
	}

	return Clock{h, m}
}

func (c Clock) Add(minutes int) Clock {
	return New(c.hours, c.minutes+minutes)
}

func (c Clock) Subtract(minutes int) Clock {
	return New(c.hours, c.minutes-minutes)
}

func (c Clock) String() string {
	return fmt.Sprintf("%02d:%02d", c.hours, c.minutes)
}

