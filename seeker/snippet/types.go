//date: 2022-01-28T17:13:23Z
//url: https://api.github.com/gists/4be452bbb48d71c6776af6ebb740b8f7
//owner: https://api.github.com/users/benmoss

// +kubebuilder:object:generate=true
// +groupName=foo
package foo

// This type shouldn't be documented
type EmbeddedFields struct {
	// Conditions are the conditions
	Conditions []string
}

// +kubebuilder:object:root=true
// Bar is a bar
type Bar struct {
	// Name is the name
	Name string
	EmbeddedFields
}
