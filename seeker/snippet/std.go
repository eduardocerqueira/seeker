//date: 2024-10-25T14:42:07Z
//url: https://api.github.com/gists/f28f4b12864fd40a2ca6b6c20294da27
//owner: https://api.github.com/users/moul

// just reorganizing the current `std` into "categories", not yet improving the API.

// chain
type AddressList []Address
    func NewAddressList() *AddressList
type AddressSet interface{ ... }
type RawAddress [RawAddressSize]byte
const RawAddressSize = 20
func GetChainID() string
func GetHeight() int64
type Address string
    func DerivePkgAddr(pkgPath string) Address
    func EncodeBech32(prefix string, bz [20]byte) Address
    func GetCallerAt(n int) Address
    func GetOrigCaller() Address
    func GetOrigPkgAddr() Address
func DecodeBech32(addr Address) (prefix string, bz [20]byte, ok bool)
func Emit(typ string, attrs ...string)

// chain/runtime
type Realm struct{ ... }
    func CurrentRealm() Realm
    func PrevRealm() Realm
func IsOriginCall() bool
func AssertOriginCall()

// chain/banker
type Banker interface{ ... }
    func GetBanker(bt BankerType) Banker
type BankerType uint8
    const BankerTypeReadonly BankerType = iota ...
type Coin struct{ ... }
    func NewCoin(denom string, amount int64) Coin
type Coins []Coin
    func GetOrigSend() Coins
    func NewCoins(coins ...Coin) Coins

// chain/params
func SetParamBool(key string, val bool)
func SetParamBytes(key string, val []byte)
func SetParamInt64(key string, val int64)
func SetParamString(key string, val string)
func SetParamUint64(key string, val uint64)