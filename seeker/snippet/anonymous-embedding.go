//date: 2024-06-11T17:08:59Z
//url: https://api.github.com/gists/4ef6fe92df86865f7c804f98a8bf812e
//owner: https://api.github.com/users/docsallover

type Customer struct {
  Name       string
  Address string
  *PaymentInfo // Anonymous embedding of PaymentInfo struct
}

type PaymentInfo struct {
  Cardholder string
  CardNumber string
}