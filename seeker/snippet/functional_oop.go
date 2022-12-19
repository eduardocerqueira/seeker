//date: 2022-12-19T16:42:06Z
//url: https://api.github.com/gists/672f592aab6f7da514490acc4ed529b1
//owner: https://api.github.com/users/packrat386

package main

import (
	"fmt"
)

// an object is a function that returns its methods, which have names
type object func() map[string]method

type method func(args ...interface{}) []interface{}

// an object can be sent a named method with its arguments
// and get back return values
func send(o object, method string, args ...interface{}) []interface{} {
	m := o()[method]
	if m == nil {
		panic("NoMethodError lol")
	}

	return m(args...)
}

func constructObject(methods map[string]method) object {
	return func() map[string]method {
		return methods
	}
}

// one object can inherit from another by keeping some of its
// methods and either overriding or adding some
func inherit(parent object, methods map[string]method) object {
	base := parent()

	for name, method := range methods {
		base[name] = method
	}

	return constructObject(base)
}

// wallet interface, defined on object.
//
// limitation: we can only have one signature per name in go
//
// limitation: we kinda have to deal with interface{} a lot because
// go does not give us any alternative
//
// limitation: we would have to actually type these out every time
// there's no reason that these could be guessed by the compiler
func getBalance(o object) int64 {
	ret := send(o, "getBalance")

	return ret[0].(int64)
}

func deposit(o object, i int64) {
	send(o, "deposit", i)
}

func withdraw(o object, i int64) error {
	ret := send(o, "withdraw", i)

	// nil loses any previous identity once it is put into
	// a []interface{}, and the cast back to error will fail
	if ret[0] == nil {
		return nil
	}

	return ret[0].(error)
}

func newWallet(initialBalance int64) object {
	balance := initialBalance

	return constructWallet(&balance)
}

// We use this "construct" function in all of our objects so
// that we can be clear about what memory is potentially
// shareable. This interface is functionally private
func constructWallet(balance *int64) object {
	getBalance := func(args ...interface{}) []interface{} {
		return []interface{}{*balance}
	}

	deposit := func(args ...interface{}) []interface{} {
		i := args[0].(int64)
		*balance += i

		return []interface{}{}
	}

	withdraw := func(args ...interface{}) []interface{} {
		i := args[0].(int64)
		if i > *balance {
			return []interface{}{fmt.Errorf("cannot withdraw %d tokens, wallet only contains %d")}
		}

		*balance -= i

		return []interface{}{nil}
	}

	return constructObject(map[string]method{
		"getBalance": getBalance,
		"deposit":    deposit,
		"withdraw":   withdraw,
	})
}

// overdraftingWallet overrides withdraw to allow for withdrawals
// to go into the negative (and not error out)

func newOverdraftingWallet(initialBalance int64) object {
	balance := initialBalance

	return constructOverdraftingWallet(&balance)
}

// we have to use the parents private constructor to ensure that
// our balance is shared across the two
func constructOverdraftingWallet(balance *int64) object {
	parent := constructWallet(balance)

	withdraw := func(args ...interface{}) []interface{} {
		i := args[0].(int64)
		*balance -= i
		return []interface{}{nil}
	}

	return inherit(parent, map[string]method{
		"withdraw": withdraw,
	})
}

// namedWallet adds this to the interface

func getName(o object) string {
	ret := send(o, "getName")

	return ret[0].(string)
}

func setName(o object, name string) {
	send(o, "setName", name)
}

func newNamedWallet(initialBalance int64, initialName string) object {
	balance := initialBalance
	name := initialName

	return constructNamedWallet(&balance, &name)
}

func constructNamedWallet(balance *int64, name *string) object {
	parent := constructWallet(balance)

	getName := func(args ...interface{}) []interface{} {
		return []interface{}{*name}
	}

	setName := func(args ...interface{}) []interface{} {
		n := args[0].(string)
		*name = n

		return []interface{}{}
	}

	return inherit(parent, map[string]method{
		"getName": getName,
		"setName": setName,
	})

}

// Right now all the types only work in here via divinely preordained
// harmony, but there's no reason they couldn't be statically type
// checked. We know the type of our wallets when we construct them
// and so we know which methods they support.
func main() {
	fmt.Println("BASE WALLET")
	w := newWallet(20)

	fmt.Println(getBalance(w))

	deposit(w, 20)

	fmt.Println(getBalance(w))

	err := withdraw(w, 10)
	if err != nil {
		panic(err)
	}

	fmt.Println(getBalance(w))

	err = withdraw(w, 50)
	if err == nil {
		panic("should have errored here?")
	}

	fmt.Println(getBalance(w))

	fmt.Println("OVERDRAFTING WALLET")
	odw := newOverdraftingWallet(20)
	fmt.Println(getBalance(odw))

	err = withdraw(odw, 30)
	if err != nil {
		panic(err)
	}

	fmt.Println(getBalance(odw))

	deposit(odw, 5)
	fmt.Println(getBalance(odw))

	fmt.Println("NAMED WALLET")
	nw := newNamedWallet(20, "my wallet")
	fmt.Println(getBalance(nw))

	fmt.Println(getName(nw))
	setName(nw, "my favorite wallet")
	fmt.Println(getName(nw))
}
