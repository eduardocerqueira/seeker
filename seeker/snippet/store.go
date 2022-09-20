//date: 2022-09-20T17:14:53Z
//url: https://api.github.com/gists/ea3d8983510866582d7c2b1db638439a
//owner: https://api.github.com/users/marbar3778

package store

import (
	"github.com/cosmos/cosmos-sdk/codec"
	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkerrors "github.com/cosmos/cosmos-sdk/types/errors"
	"github.com/tendermint/tendermint/libs/log"

	"github.com/berachain/berachain-node/internal/store/key"
)

// KVStore is a wrapper around the cosmos-sdk KVStore to provide more safety regarding key management and better ease-of-use
type KVStore struct {
	sdk.KVStore
	cdc codec.BinaryCodec
}

// NewNormalizedStore returns a new KVStore
func NewNormalizedStore(store sdk.KVStore, cdc codec.BinaryCodec) KVStore {
	return KVStore{
		KVStore: store,
		cdc:     cdc,
	}
}

// SetNew marshals the value and stores it under the given key
func (store KVStore) SetNew(k key.Key, value codec.ProtoMarshaler) {
	store.KVStore.Set(k.Bytes(), store.cdc.MustMarshalLengthPrefixed(value))
}

// GetNew unmarshals the raw bytes stored under the given key into the value object. Returns true if the key exists.
func (store KVStore) GetNew(key key.Key, value codec.ProtoMarshaler) bool {
	value.Reset()

	bz := store.KVStore.Get(key.Bytes())
	if bz == nil {
		return false
	}
	store.cdc.MustUnmarshalLengthPrefixed(bz, value)
	return true
}

// Has returns true if the key exists.
func (store KVStore) Has(key key.Key) bool {
	return store.KVStore.Has(key.Bytes())
}

// Delete deletes the value stored under the given key, if it exists
func (store KVStore) Delete(key key.Key) {
	store.KVStore.Delete(key.Bytes())
}

// IteratorNew returns an Iterator that can handle a structured Key
func (store KVStore) IteratorNew(prefix key.Key) Iterator {
	iter := sdk.KVStorePrefixIterator(store.KVStore, append(prefix.Bytes(), []byte(key.DefaultDelimiter)...))
	return iterator{Iterator: iter, cdc: store.cdc}
}

// ReverseIterator returns an Iterator that can handle a structured Key and
// interate reversely
func (store KVStore) ReverseIteratorNew(prefix key.Key) Iterator {
	iter := sdk.KVStoreReversePrefixIterator(store.KVStore, append(prefix.Bytes(), []byte(key.DefaultDelimiter)...))
	return iterator{Iterator: iter, cdc: store.cdc}
}

// Iterator is an easier and safer to use sdk.Iterator extension
type Iterator interface {
	sdk.Iterator
	UnmarshalValue(marshaler codec.ProtoMarshaler)
	GetKey() key.Key
}

type iterator struct {
	sdk.Iterator
	cdc codec.BinaryCodec
}

// UnmarshalValue returns the value marshaled into the given type
func (i iterator) UnmarshalValue(value codec.ProtoMarshaler) {
	value.Reset()
	i.cdc.MustUnmarshalLengthPrefixed(i.Value(), value)
}

// GetKey returns the key of the current iterator value
func (i iterator) GetKey() key.Key {
	return key.FromBz(i.Key())
}

// CloseLogError closes the given iterator and logs if an error is returned
func CloseLogError(iter sdk.Iterator, logger log.Logger) {
	err := iter.Close()
	if err != nil {
		logger.Error(sdkerrors.Wrap(err, "failed to close kv store iterator").Error())
	}
}