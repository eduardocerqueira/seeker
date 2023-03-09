//date: 2023-03-09T17:11:21Z
//url: https://api.github.com/gists/c2381654be8880d1138b12ffe1488741
//owner: https://api.github.com/users/tarik0

package dexpair

import (
	"github.com/ethereum/go-ethereum/common"
	"sync"
	"testing"
)

// TestRaceCondition tests the reserves read/write in a race condition.
// The pair should panic when a concurrent read/write happens.
func TestRaceCondition(t *testing.T) {
	addr := common.BigToAddress(common.Big1)

	// Generate new pair.
	p := NewDexPair(
		addr, addr, addr, common.Big1, common.Big1,
	)

	// Function to change reserves.
	changeReserveFunc := func(errCh chan bool, wg *sync.WaitGroup) {
		defer func() {
			if r := recover(); r == nil {
				errCh <- true
			} else {
				errCh <- false
			}
			wg.Done()
		}()
		p.SetReserves(common.Big0, common.Big0)
	}

	// The error channel and wait group.
	size := 200
	errCh := make(chan bool, size)
	wg := new(sync.WaitGroup)

	// Create goroutines.
	for i := 0; i < size; i++ {
		wg.Add(1)
		go changeReserveFunc(errCh, wg)
	}
	wg.Wait()
	close(errCh)

	// Filter results.
	isPanicked := false
	for isErr := range errCh {
		if isErr {
			isPanicked = true
			break
		}
	}

	// Check if panicked.
	if isPanicked == false {
		t.Error("pair allowed concurrent read/write")
	}
}
