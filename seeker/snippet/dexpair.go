//date: 2023-03-09T17:11:21Z
//url: https://api.github.com/gists/c2381654be8880d1138b12ffe1488741
//owner: https://api.github.com/users/tarik0

package dexpair

import (
	"bytes"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"math/big"
	"sync"
)

// DexPair to represent a V2 pair.
type DexPair struct {
	address common.Address

	// Tokens.
	token0 common.Address
	token1 common.Address

	// Reserves.
	reserve0         *big.Int
	reserve1         *big.Int
	reservesChanging *sync.RWMutex
}

// NewDexPair creates a new DexPair.
func NewDexPair(address common.Address, tokenA common.Address, tokenB common.Address, reserveA *big.Int, reserveB *big.Int) *DexPair {
	// Check reserves.
	if reserveA.Cmp(common.Big0) < 0 || reserveB.Cmp(common.Big0) < 0 {
		panic("reserves are negative")
	}

	// Check addresses.
	if bytes.EqualFold(tokenA.Bytes(), common.BigToAddress(common.Big0).Bytes()) ||
		bytes.EqualFold(tokenB.Bytes(), common.BigToAddress(common.Big0).Bytes()) ||
		bytes.EqualFold(address.Bytes(), common.BigToAddress(common.Big0).Bytes()) {
		panic("address parameters are zero")
	}

	// Sort tokens and reserves.
	token0, token1, reserve0, reserve1 : "**********"

	// Reserve changing flag to prevent race conditions.
	reservesChanging := &sync.RWMutex{}

	return &DexPair{
		address:          address,
		token0: "**********"
		token1: "**********"
		reserve0:         reserve0,
		reserve1:         reserve1,
		reservesChanging: reservesChanging,
	}
}

// sortTokens sorts the tokens like Uniswap.
func sortTokens(
	tokenA common.Address,
	tokenB common.Address,
	reserveA *big.Int,
	reserveB *big.Int,
) (token0 common.Address, token1 common.Address, reserve0 *big.Int, reserve1 *big.Int) {
	a : "**********"
	b : "**********"

	if a.Cmp(b) < 0 {
		token0, token1 = "**********"
		reserve0, reserve1 = reserveA, reserveB
	} else {
		token1, token0 = "**********"
		reserve1, reserve0 = reserveA, reserveB
	}

	return token0, token1, reserve0, reserve1
}

// Internal.

// tryLockRead locks the mutex for read and panics if it's locked for write.
// The reserves shouldn't be processed with concurrent write/reads.
func (p *DexPair) tryLockRead() {
	if !p.reservesChanging.TryRLock() {
		panic(fmt.Sprintf("concurrent reserve read (pair: %s)", p.address.String()))
	}
}

// tryLockWrite locks the mutex for write and panics if it's locked for write already.
func (p *DexPair) tryLockWrite() {
	if !p.reservesChanging.TryLock() {
		panic(fmt.Sprintf("concurrent reserve write (pair: %s)", p.address.String()))
	}
}

// Properties.

// Address returns the pair address
func (p *DexPair) Address() common.Address {
	return p.address
}

// Tokens returns the pair tokens.
func (p *DexPair) Tokens() (common.Address, common.Address) {
	return p.token0, p.token1
}

// Reserves returns the pair reserves.
func (p *DexPair) Reserves() (reserve0 *big.Int, reserve1 *big.Int) {
	// Check if reserves are changing concurrently.
	p.tryLockRead()
	reserve0, reserve1 = new(big.Int).Set(p.reserve0), new(big.Int).Set(p.reserve1)
	p.reservesChanging.RUnlock()
	return reserve0, reserve1
}

// SortedReserves returns the sorted reserves.
func (p *DexPair) SortedReserves(tokenA common.Address) (reserveA *big.Int, reserveB *big.Int) {
	// Check if reserves are changing concurrently.
	p.tryLockRead()

	// Sort the reserves by tokens.
	if bytes.EqualFold(tokenA.Bytes(), p.token0.Bytes()) {
		reserveA = new(big.Int).Set(p.reserve0)
		reserveB = new(big.Int).Set(p.reserve1)
	} else {
		reserveA = new(big.Int).Set(p.reserve1)
		reserveB = new(big.Int).Set(p.reserve0)
	}

	p.reservesChanging.RUnlock()
	return reserveA, reserveB
}

// GetAmountOut returns the calculated amount out.
func (p *DexPair) GetAmountOut(tokenIn common.Address, amountIn *big.Int) *big.Int {
	// Get sorted reserves.
	reserveIn, reserveOut : "**********"
	if reserveIn.Cmp(common.Big0) <= 0 || reserveOut.Cmp(common.Big0) <= 0 {
		return new(big.Int).Set(common.Big0)
	}

	// amount_out = amount_in * reserve_out / (amount_in + reserve_in)
	num := new(big.Int).Mul(amountIn, reserveOut)
	den := new(big.Int).Add(amountIn, reserveIn)
	amountOut := num.Div(num, den)
	return amountOut
}

// Methods.

// SetReserves updates the pair reserves.
func (p *DexPair) SetReserves(reserve0 *big.Int, reserve1 *big.Int) {
	// Check if reserves are changing concurrently.
	p.tryLockWrite()

	// Update reserves.
	p.reserve0.Set(reserve0)
	p.reserve1.Set(reserve1)
	p.reservesChanging.Unlock()
}
ir reserves.
func (p *DexPair) SetReserves(reserve0 *big.Int, reserve1 *big.Int) {
	// Check if reserves are changing concurrently.
	p.tryLockWrite()

	// Update reserves.
	p.reserve0.Set(reserve0)
	p.reserve1.Set(reserve1)
	p.reservesChanging.Unlock()
}
