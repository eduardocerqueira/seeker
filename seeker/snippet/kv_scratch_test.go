//date: 2023-11-13T16:56:27Z
//url: https://api.github.com/gists/f2ac1c99c63291692b18fdae715338af
//owner: https://api.github.com/users/mickmister

package main

import (
	"testing"

	"github.com/mattermost/mattermost/server/public/plugin/plugintest"
	"github.com/stretchr/testify/require"
)

func TestKVScratch(t *testing.T) {
	p := &Plugin{}
	api := &plugintest.API{}
	p.SetAPI(api)

	testStore := makeTestKVStore(api, testKVStore{
		"initial": []byte("initial value"),
	})

	p.KVScratch()

	require.Equal(t, "my value", string(testStore["mykey"]))
	require.Equal(t, "initial value", string(testStore["fetched"]))
}
