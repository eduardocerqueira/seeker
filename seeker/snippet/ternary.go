//date: 2025-03-26T16:55:34Z
//url: https://api.github.com/gists/a4786eb2dff87b4e62d1d711da77221d
//owner: https://api.github.com/users/ttys3

package ternary

func If[T any](condition bool, ifTrue Supplier[T], ifFalse Supplier[T]) T {
	if condition {
		return ifTrue.Get()
	}
	return ifFalse.Get()
}

func Func[T any](supplier func() T) Supplier[T] {
	return FuncSupplier[T](supplier)
}

func Value[T any](value T) Supplier[T] {
	return ValueSupplier[T]{Value: value}
}

type Supplier[T any] interface {
	Get() T
}

type FuncSupplier[T any] func() T

func (s FuncSupplier[T]) Get() T {
	return s()
}

type ValueSupplier[T any] struct {
	Value T
}

func (s ValueSupplier[T]) Get() T {
	return s.Value
}
