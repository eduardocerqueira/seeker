//date: 2026-02-25T17:52:07Z
//url: https://api.github.com/gists/946294a5cfdf298f6c4b2ad045a11ae0
//owner: https://api.github.com/users/saniales

package mongo

type Document struct {
	ID   string `json:"id" bson:"_id,omitempty"`
	Name string `json:"name"`
}
