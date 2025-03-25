//date: 2025-03-25T17:04:05Z
//url: https://api.github.com/gists/179b7759698793755ff66e038640b1c1
//owner: https://api.github.com/users/rgolangh

package populator

//go:generate mockgen -destination=mocks/storage_mock_client.go -package=storage_mocks . StorageApi
type StorageApi interface {
	StorageMapper
	StorageResolver
}

type StorageAdapterID string

type StorageMapper interface {
	// EnsureClonnerIgroup creates or updates an initiator group with the clonnerIqn
	// and returns list of IDs and an error.
	EnsureClonnerIgroup(initiatorGroup string, clonnerIqn string) ([]StorageAdapterID, error)
	// Map is responsible to mapping an initiator group to a LUN
	// ids is a list of storage adapter IDs returned by the EnsureClonnerIgroup
	Map(initatorGroup string, targetLUN LUN, ids []StorageAdapterID) error
	// UnMap is responsible to unmapping an initiator group from a LUN
	// ids is a list of storage adapter IDs returned by the EnsureClonnerIgroup
	UnMap(initatorGroup string, targetLUN LUN, ids []StorageAdapterID) error
	// CurrentMappedGroups returns the initiator groups the LUN is mapped to
	CurrentMappedGroups(targetLUN LUN) ([]string, []StorageAdapterID, error)
}

type StorageResolver interface {
	ResolveVolumeHandleToLUN(volumeHandle string) (LUN, error)
}
	