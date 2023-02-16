//date: 2023-02-16T16:58:23Z
//url: https://api.github.com/gists/bd79fdd1613db934cebbf330a97d2196
//owner: https://api.github.com/users/Hiccup1234

package main
import "fmt"
func testCase(N int) {
    if N <= 0 {
        return
    }
    var X int
    fmt.Scanf("%d", &X)
    fmt.Println(sOs(X))
    testCase(N-1)
}
func sOs(X int) int {
    if X == 0 {
        return 0
    }
    var Y int
    fmt.Scanf("%d", &Y)
    if Y > 0 {
        return Y*Y + sOs(X-1)
    }
    return sOs(X-1)
}
func main() {
    var N int
    fmt.Scanf("%d", &N)
    testCase(N)
}