//date: 2022-12-20T16:50:14Z
//url: https://api.github.com/gists/fb65eaf808d02044a5edd7508b0f7188
//owner: https://api.github.com/users/lazyfrosch

package net_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetNetworkRange(t *testing.T) {
	first, last, err := net.GetNetworkRange("10.0.0.0/23")
	assert.NoError(t, err)
	assert.Equal(t, "10.0.0.2", first.String())
	assert.Equal(t, "10.0.1.254", last.String())

	first, last, err = net.GetNetworkRange("172.16.0.0/12")
	assert.NoError(t, err)
	assert.Equal(t, "172.16.0.2", first.String())
	assert.Equal(t, "172.31.255.254", last.String())
}
