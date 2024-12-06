//date: 2024-12-06T16:50:32Z
//url: https://api.github.com/gists/91b7174fba003fe62863a9aace4a1997
//owner: https://api.github.com/users/shackra

package seed

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewWalletPhantomCompatible(t *testing.T) {
	bip39seed := "dash hair garment vehicle keen wine effort hole stay similar end double"
	expectedAddress := "3BLVofLEJGsb9bnQXPn9ds7oFzTyLw2Qk4Sa7uVSaWwY"

	wallet, err := GenerateWalletFromMnemonicSeed(bip39seed, "")

	require.NoError(t, err)

	require.Equal(t, expectedAddress, wallet.PublicKey().String())
}
