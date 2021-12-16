//date: 2021-12-16T16:51:27Z
//url: https://api.github.com/gists/343af9be3fdd76d644865a7052953701
//owner: https://api.github.com/users/ImShakthi

package main

import (
	"context"
	"fmt"
)

type ctxKey int

const (
	log ctxKey = iota
	pref
)

func main() {
	ctx := context.Background()

	ctx = context.WithValue(ctx, log, "10")
	fmt.Printf("log :: %+v\n", ctx.Value(log))
	fmt.Printf("context :: %+v\n", ctx)

	ctx = context.WithValue(ctx, log, "18")
	fmt.Printf("log :: %+v\n", ctx.Value(log))
	fmt.Printf("context :: %+v\n", ctx)

	
	ctx = context.WithValue(ctx, log, "1")
	ctx = context.WithValue(ctx, log, "2")
	ctx = context.WithValue(ctx, pref, "3")
	fmt.Printf("log :: %+v\n", ctx.Value(log))
	fmt.Printf("pref :: %+v\n", ctx.Value(pref))
	fmt.Printf("context :: %+v\n", ctx)
}
