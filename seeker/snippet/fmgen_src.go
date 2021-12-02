//date: 2021-12-02T17:05:17Z
//url: https://api.github.com/gists/c8831c7027d8e1398ea7be5e347828fa
//owner: https://api.github.com/users/ryan-holcombe

package sample

// Sample a sample struct
type Sample struct {
    ID          int64 `fmgen:"-"`
    Name        string
    Age         int64 `fmgen:"optional"`
    LastUpdated time.Time
}

// Simple struct with just a name
type Simple struct {
    Name string
}
