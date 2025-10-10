//date: 2025-10-10T17:07:37Z
//url: https://api.github.com/gists/91b2291ead3008137b2531532f80d851
//owner: https://api.github.com/users/goodevilgenius

package ctx

import "context"

// Value returns a strongly-typed value from the given context using ctx.Value.
// ok will be true if found in the context and if the correct type.
func Value[T any](ctx context.Context, key any) (val T, ok bool) {
	if val, ok = ctx.Value(key).(T); ok {
		return val, ok
	}
	var zero T
	return zero, false
}

// AssignValue will set the value of ptr to the value from Value(ctx, key), if it's
// found in the context and the same type as ptr.
func AssignValue[T any](ctx context.Context, key any, ptr *T) bool {
	val, ok := Value[T](ctx, key)
	if ok {
		*ptr = val
	}
	return ok
}