//date: 2023-03-22T16:50:49Z
//url: https://api.github.com/gists/16cba4b19090df78e66ab15fca70da21
//owner: https://api.github.com/users/dannyvelas

// You can edit this code!
// Click here and start typing.
package main

import "fmt"

type S struct {
	Name string
}

type T struct {
	FieldsS []S
}

func main() {
	// Proof that an object that is copied at the top of a for-each loop, cannot have a property changed within that loop
	ss := []S{{"hi"}}
	for _, s := range ss {
		s.Name = "yoooo"
		fmt.Println(s.Name)
	}
	for _, s := range ss {
		fmt.Println(s.Name)
	}

	/*
		output:
			yooo
			hi
	*/

	// Proof that an object that is copied at the top of a for-each loop, can have _members_ of its property changed within that loop, as long as the inner loop uses indexing instead of a for-each loop
	ts := []T{T{FieldsS: []S{S{"hi"}, S{"bye"}} /*close S arr*/} /*closeT */} /*close T arr*/

	for _, copyOfT := range ts {
		copyOfPtrToFieldsS := copyOfT.FieldsS
		for j := range copyOfPtrToFieldsS {
			copyOfPtrToFieldsS[j].Name = "yooo"
		}
	}
	fmt.Println(ts)

	/*
		output:
			[{[{yooo} {yooo}]}]
	*/

	// Proof that an array can have their members changed as long as a new array is using pointers to its members
	ss1 := []S{{"HI"}, {"BYE"}}
	ss2 := []S{{"HOLA"}, {"CHAO"}}
	allSs := make([]*S, 0, len(ss1)+len(ss2))
	for i := range ss1 {
		allSs = append(allSs, &ss1[i])
	}
	for i := range ss2 {
		allSs = append(allSs, &ss2[i])
	}

	for _, s := range allSs {
		s.Name = "yoo"
	}

	fmt.Println(ss1)
	fmt.Println(ss2)
	/*
		 output:
			[{yoo} {yoo}]
			[{yoo} {yoo}]
	*/
}