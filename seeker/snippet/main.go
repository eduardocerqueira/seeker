//date: 2022-01-24T17:08:58Z
//url: https://api.github.com/gists/04a621a29a7788ce04edc44699f56da1
//owner: https://api.github.com/users/achilleas-k

package main

import (
	"encoding/json"
	"fmt"
	"sort"
)

type EntityType uint8

const (
	ETPartitionTable EntityType = iota
	ETPartition
)

const (
	BIOSBootPartitionGUID = "21686148-6449-6E6F-744E-656564454649"
	BIOSBootPartitionUUID = "FAC7F1FB-3E8D-4137-A512-961DE09A5549"

	FilesystemDataGUID = "0FC63DAF-8483-4772-8E79-3D69D8477DE4"
	FilesystemDataUUID = "CB07C243-BC44-4717-853E-28852021225B"

	EFISystemPartitionGUID = "C12A7328-F81F-11D2-BA4B-00A0C93EC93B"
	EFISystemPartitionUUID = "68B2905B-DF3E-4FB3-80FA-49D1E773AA33"
	EFIFilesystemUUID      = "7B77-95E7"

	RootPartitionUUID = "6264D520-3FB9-423F-8AB8-7A0A8E3D3562"
)

type Entity interface {
	GetType() EntityType
}

type Container interface {
	Entity
	GetItemCount() uint
	GetChild(n uint) Entity
}

type Sizeable interface {
	EnsureSize(size uint64) bool
	GetSize() uint64
}

type Mountable interface {
	GetMountpoint() string
}

type VolumeContainer interface {
	CreateVolume(mountpoint string, size uint64) (Entity, error)
}

type PartitionTable struct {
	Size       uint64 // Size of the disk (in bytes).
	UUID       string // Unique identifier of the partition table (GPT only).
	Type       string // Partition table type, e.g. dos, gpt.
	Partitions []Partition

	SectorSize   uint64 // Sector size in bytes
	ExtraPadding uint64 // Extra space at the end of the partition table (sectors)
}

func (pt *PartitionTable) GetType() EntityType {
	return EntityType(0)
}

func (pt *PartitionTable) GetItemCount() uint {
	return uint(len(pt.Partitions))
}

func (pt *PartitionTable) GetChild(n uint) Entity {
	return &pt.Partitions[n]
}

func (pt *PartitionTable) GetSize() uint64 {
	return pt.Size
}

func (pt *PartitionTable) EnsureSize(s uint64) bool {
	if s > pt.Size {
		pt.Size = s
		return true
	}
	return false
}

func (pt *PartitionTable) CreateVolume(mountpoint string, size uint64) (Entity, error) {
	filesystem := Filesystem{
		Type:         "xfs",
		Mountpoint:   mountpoint,
		FSTabOptions: "defaults",
		FSTabFreq:    0,
		FSTabPassNo:  0,
	}

	partition := Partition{
		Size:    size,
		Payload: &filesystem,
	}

	n := len(pt.Partitions)
	var maxNo int

	if pt.Type == "gpt" {
		partition.Type = FilesystemDataGUID
		maxNo = 128
	} else {
		maxNo = 4
	}

	if n == maxNo {
		return nil, fmt.Errorf("maximum number of partitions reached (%d)", maxNo)
	}

	pt.Partitions = append(pt.Partitions, partition)

	return &pt.Partitions[len(pt.Partitions)-1], nil
}

type Partition struct {
	Start    uint64 // Start of the partition in bytes
	Size     uint64 // Size of the partition in bytes
	Type     string // `gpt` or `dos`
	Bootable bool   // `Legacy BIOS bootable` (GPT) or `active` (DOS) flag
	// ID of the partition, dos doesn't use traditional UUIDs, therefore this
	// is just a string.
	UUID string
	// If nil, the partition is raw; It doesn't contain a filesystem.
	Payload Entity
}

func (p *Partition) GetType() EntityType {
	return EntityType(0)
}

func (pt *Partition) GetItemCount() uint {
	if pt.Payload == nil {
		return 0
	}
	return 1
}

func (p *Partition) GetChild(n uint) Entity {
	if n != 0 {
		panic("wat")
	}
	return p.Payload
}

func (p *Partition) GetSize() uint64 {
	return p.Size
}

func (p *Partition) EnsureSize(s uint64) bool {
	if s > p.Size {
		p.Size = s
		return true
	}
	return false
}

type Filesystem struct {
	Type string
	// ID of the filesystem, vfat doesn't use traditional UUIDs, therefore this
	// is just a string.
	UUID       string
	Label      string
	Mountpoint string
	// The fourth field of fstab(5); fs_mntops
	FSTabOptions string
	// The fifth field of fstab(5); fs_freq
	FSTabFreq uint64
	// The sixth field of fstab(5); fs_passno
	FSTabPassNo uint64
}

func (fs *Filesystem) GetType() EntityType {
	return EntityType(0)
}

func (fs *Filesystem) GetMountpoint() string {
	return fs.Mountpoint
}

type LUKSContainer struct {
	UUID string
	// TODO: Fill in osbuild options

	Payload Entity
}

func (lc *LUKSContainer) GetType() EntityType {
	return EntityType(0)
}

func (lc *LUKSContainer) GetItemCount() uint {
	if lc.Payload == nil {
		return 0
	}
	return 1
}

func (lc *LUKSContainer) GetChild(n uint) Entity {
	if n != 0 {
		panic("wat")
	}
	return lc.Payload
}

type LVMVolumeGroup struct {
	Name        string
	Description string

	LogicalVolumes []LVMLogicalVolume
}

func (vg *LVMVolumeGroup) GetType() EntityType {
	return EntityType(0)
}

func (vg *LVMVolumeGroup) GetItemCount() uint {
	return uint(len(vg.LogicalVolumes))
}

func (vg *LVMVolumeGroup) GetChild(n uint) Entity {
	return &vg.LogicalVolumes[n]
}

func (vg *LVMVolumeGroup) CreateVolume(mountpoint string, size uint64) (Entity, error) {
	filesystem := Filesystem{
		Type:         "xfs",
		Mountpoint:   mountpoint,
		FSTabOptions: "defaults",
		FSTabFreq:    0,
		FSTabPassNo:  0,
	}

	lv := LVMLogicalVolume{
		Size:    size,
		Payload: &filesystem,
	}

	vg.LogicalVolumes = append(vg.LogicalVolumes, lv)

	return &vg.LogicalVolumes[len(vg.LogicalVolumes)-1], nil
}

type LVMLogicalVolume struct {
	Size    uint64
	Payload Entity
}

func (lv *LVMLogicalVolume) GetType() EntityType {
	return EntityType(0)
}

func (lv *LVMLogicalVolume) GetItemCount() uint {
	if lv.Payload == nil {
		return 0
	}
	return 1
}

func (lv *LVMLogicalVolume) GetChild(n uint) Entity {
	if n != 0 {
		panic("wat")
	}
	return lv.Payload
}

func (lv *LVMLogicalVolume) GetSize() uint64 {
	return lv.Size
}

func (lv *LVMLogicalVolume) EnsureSize(s uint64) bool {
	if s > lv.Size {
		lv.Size = s
		return true
	}
	return false
}

type Btrfs struct {
	UUID       string
	Label      string
	Mountpoint string
	Subvolumes []BtrfsSubvolumes
}

func (b *Btrfs) GetType() EntityType {
	return EntityType(0)
}

type BtrfsSubvolumes struct {
	Size       uint64
	Mountpoint string
	GroupID    uint64
}

func (subvol *BtrfsSubvolumes) GetType() EntityType {
	return EntityType(0)
}

// -----------------------------

var partitionTables = map[string]PartitionTable{
	"plain": {
		UUID: "D209C89E-EA5E-4FBD-B161-B461CCE297E0",
		Type: "gpt",
		Partitions: []Partition{
			{
				Size:     2048, // 1MB
				Bootable: true,
				Type:     BIOSBootPartitionGUID,
				UUID:     BIOSBootPartitionUUID,
			},
			{
				Size: 409600, // 200 MB
				Type: EFISystemPartitionGUID,
				UUID: EFISystemPartitionUUID,
				Payload: &Filesystem{
					Type:         "vfat",
					UUID:         EFIFilesystemUUID,
					Mountpoint:   "/boot/efi",
					Label:        "EFI-SYSTEM",
					FSTabOptions: "defaults,uid=0,gid=0,umask=077,shortname=winnt",
					FSTabFreq:    0,
					FSTabPassNo:  2,
				},
			},
			{
				Size: 1024000, // 500 MB
				Type: FilesystemDataGUID,
				UUID: FilesystemDataUUID,
				Payload: &Filesystem{
					Type:         "xfs",
					Mountpoint:   "/boot",
					Label:        "boot",
					FSTabOptions: "defaults",
					FSTabFreq:    0,
					FSTabPassNo:  0,
				},
			},
			{
				Type: FilesystemDataGUID,
				UUID: RootPartitionUUID,
				Payload: &Filesystem{
					Type:         "xfs",
					Label:        "root",
					Mountpoint:   "/",
					FSTabOptions: "defaults",
					FSTabFreq:    0,
					FSTabPassNo:  0,
				},
			},
		},
	},

	"luks": {
		UUID: "D209C89E-EA5E-4FBD-B161-B461CCE297E0",
		Type: "gpt",
		Partitions: []Partition{
			{
				Size:     2048, // 1MB
				Bootable: true,
				Type:     BIOSBootPartitionGUID,
				UUID:     BIOSBootPartitionUUID,
			},
			{
				Size: 409600, // 200 MB
				Type: EFISystemPartitionGUID,
				UUID: EFISystemPartitionUUID,
				Payload: &Filesystem{
					Type:         "vfat",
					UUID:         EFIFilesystemUUID,
					Mountpoint:   "/boot/efi",
					Label:        "EFI-SYSTEM",
					FSTabOptions: "defaults,uid=0,gid=0,umask=077,shortname=winnt",
					FSTabFreq:    0,
					FSTabPassNo:  2,
				},
			},
			{
				Size: 1024000, // 500 MB
				Type: FilesystemDataGUID,
				UUID: FilesystemDataUUID,
				Payload: &Filesystem{
					Type:         "xfs",
					Mountpoint:   "/boot",
					Label:        "boot",
					FSTabOptions: "defaults",
					FSTabFreq:    0,
					FSTabPassNo:  0,
				},
			},
			{
				Type: FilesystemDataGUID,
				UUID: RootPartitionUUID,
				Payload: &LUKSContainer{
					UUID: "",
					Payload: &Filesystem{
						Type:         "xfs",
						Label:        "root",
						Mountpoint:   "/",
						FSTabOptions: "defaults",
						FSTabFreq:    0,
						FSTabPassNo:  0,
					},
				},
			},
		},
	},
	"luks+lvm": {
		UUID: "D209C89E-EA5E-4FBD-B161-B461CCE297E0",
		Type: "gpt",
		Partitions: []Partition{
			{
				Size:     2048, // 1MB
				Bootable: true,
				Type:     BIOSBootPartitionGUID,
				UUID:     BIOSBootPartitionUUID,
			},
			{
				Size: 409600, // 200 MB
				Type: EFISystemPartitionGUID,
				UUID: EFISystemPartitionUUID,
				Payload: &Filesystem{
					Type:         "vfat",
					UUID:         EFIFilesystemUUID,
					Mountpoint:   "/boot/efi",
					Label:        "EFI-SYSTEM",
					FSTabOptions: "defaults,uid=0,gid=0,umask=077,shortname=winnt",
					FSTabFreq:    0,
					FSTabPassNo:  2,
				},
			},
			{
				Size: 1024000, // 500 MB
				Type: FilesystemDataGUID,
				UUID: FilesystemDataUUID,
				Payload: &Filesystem{
					Type:         "xfs",
					Mountpoint:   "/boot",
					Label:        "boot",
					FSTabOptions: "defaults",
					FSTabFreq:    0,
					FSTabPassNo:  0,
				},
			},
			{
				Type: FilesystemDataGUID,
				UUID: RootPartitionUUID,
				Size: 5 * 1024 * 1024 * 1024,
				Payload: &LUKSContainer{
					UUID: "",
					Payload: &LVMVolumeGroup{
						Name:        "",
						Description: "",
						LogicalVolumes: []LVMLogicalVolume{
							{
								Size: 2 * 1024 * 1024 * 1024,
								Payload: &Filesystem{
									Type:         "xfs",
									Label:        "root",
									Mountpoint:   "/",
									FSTabOptions: "defaults",
									FSTabFreq:    0,
									FSTabPassNo:  0,
								},
							},
							{
								Size: 2 * 1024 * 1024 * 1024,
								Payload: &Filesystem{
									Type:         "xfs",
									Label:        "root",
									Mountpoint:   "/home",
									FSTabOptions: "defaults",
									FSTabFreq:    0,
									FSTabPassNo:  0,
								},
							},
						},
					},
				},
			},
		},
	},
}

func entityPath(ent Entity, target string) []Entity {
	switch e := ent.(type) {
	case Mountable:
		if target == e.GetMountpoint() {
			return []Entity{ent}
		}
	case Container:
		for idx := uint(0); idx < e.GetItemCount(); idx++ {
			child := e.GetChild(idx)
			path := entityPath(child, target)
			if path != nil {
				path = append(path, e)
				return path
			}
		}
	}
	return nil
}

type FSCustomization struct {
	Mountpoint string
	Size       uint64
}

type Blueprint []FSCustomization

func _CreatePartitionTable(bp Blueprint, basePT *PartitionTable) *PartitionTable {
	for _, c := range bp {
		if path := entityPath(basePT, c.Mountpoint); path != nil {
			for _, element := range path {
				if sz, ok := element.(Sizeable); ok {
					sz.EnsureSize(c.Size)
					break
				}
			}
		}
	}
	return basePT
}

func resizeEntityBranch(path []Entity, size uint64) {
	var changed bool
	for _, element := range path {
		if changed {
			changed = false
			c, ok := element.(Container)
			if !ok {
				panic("WAT30: parent of sizeable should be container")
			}
			size = 0
			for idx := uint(0); idx < c.GetItemCount(); idx++ {
				s, ok := c.GetChild(idx).(Sizeable)
				if !ok {
					panic("WAT32: child of container should be sizeable")
				}
				size += s.GetSize()
			}
			if s, ok := element.(Sizeable); ok {
				s.EnsureSize(size)
			}
		} else if sz, ok := element.(Sizeable); ok {
			changed = sz.EnsureSize(size)
			if !changed {
				break
			}
		}
	}
}

func createFilesystem(c FSCustomization, pt *PartitionTable) {
	rootPath := entityPath(pt, "/")
	if rootPath == nil {
		panic("WAT42: No root??")
	}

	var vc VolumeContainer
	var entity Entity
	var idx int
	for idx, entity = range rootPath {
		var ok bool
		if vc, ok = entity.(VolumeContainer); ok {
			break
		}
	}

	if vc == nil {
		panic("WAT89: could not find root volume container")
	}

	newVol, err := vc.CreateVolume(c.Mountpoint, 0)
	if err != nil {
		panic("WAT99: failed creating volume: " + err.Error())
	}
	vcPath := append([]Entity{newVol}, rootPath[idx:]...)
	resizeEntityBranch(vcPath, c.Size)
}

func CreatePartitionTable(bp Blueprint, basePT *PartitionTable) *PartitionTable {
	for _, c := range bp {
		if path := entityPath(basePT, c.Mountpoint); path != nil {
			resizeEntityBranch(path, c.Size)
		} else {
			createFilesystem(c, basePT)
		}
	}
	return basePT
}

func dumpPaths() {
	for name, pt := range partitionTables {
		fmt.Printf("Pathing partition table %s\n", name)
		path := entityPath(&pt, "/")
		for idx, p := range path {
			fmt.Printf("  %d %T\n", idx, p)
		}
	}
}

func main() {
	bp := []FSCustomization{
		{
			Mountpoint: "/",
			Size:       10 * 1024 * 1024 * 1024,
		},
		{
			Mountpoint: "/home",
			Size:       20 * 1024 * 1024 * 1024,
		},
		{
			Mountpoint: "/opt",
			Size:       7 * 1024 * 1024 * 1024,
		},
	}

	names := make([]string, 0, len(partitionTables))
	for name := range partitionTables {
		names = append(names, name)
	}

	sort.Strings(names)

	for _, name := range names {
		pt := partitionTables[name]
		fmt.Printf("Modifying partition table %s\n", name)
		mpt := CreatePartitionTable(bp, &pt)
		m, _ := json.MarshalIndent(mpt, "", "  ")
		fmt.Printf("%+v\n", string(m))
	}
}
