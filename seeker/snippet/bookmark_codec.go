//date: 2022-08-29T17:00:29Z
//url: https://api.github.com/gists/e3c942dd9f4af0ef6321f309ad20df83
//owner: https://api.github.com/users/pgaskin

// Package crb decodes and encodes Chrome's BookmarkCodec.
//
// Mostly round-trip equivalent, but formatting is slightly different.
//
// Written against aabc286, Chrome 106, 2022-08-22. Should work at least as far
// back as 2014.
//
// https://source.chromium.org/chromium/chromium/src/+/main:components/bookmarks/browser/bookmark_codec.cc
package crb

import (
	"crypto/md5"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"time"
	"unicode/utf16"
)

type Bookmarks struct {
	Checksum string `json:"checksum"`
	Roots    struct {
		BookmarkBar    BookmarkNode `json:"bookmark_bar"`
		Other          BookmarkNode `json:"other"`
		MobileBookmark BookmarkNode `json:"synced"`
	} `json:"roots"`
	SyncMetadata     Bytes             `json:"sync_metadata,omitempty"`
	Version          Version           `json:"version"`
	MetaInfo         map[string]string `json:"meta_info,omitempty"`
	UnsyncedMetaInfo map[string]string `json:"unsynced_meta_info,omitempty"`
}

type BookmarkNode struct {
	Children         *[]BookmarkNode   `json:"children,omitempty"` // Type == NodeTypeFolder
	DateAdded        Time              `json:"date_added"`
	DateLastUsed     Time              `json:"date_last_used,omitempty"`
	DateModified     Time              `json:"date_modified,omitempty"`
	GUID             GUID              `json:"guid"`
	ID               int               `json:"id,string"`
	Name             string            `json:"name"`
	Type             NodeType          `json:"type"`
	URL              string            `json:"url,omitempty"`
	MetaInfo         map[string]string `json:"meta_info,omitempty"`
	UnsyncedMetaInfo map[string]string `json:"unsynced_meta_info,omitempty"`
}

func Decode(r io.Reader) (*Bookmarks, bool, error) {
	var b Bookmarks
	d := json.NewDecoder(r)
	d.DisallowUnknownFields()
	if err := d.Decode(&b); err != nil {
		return nil, false, nil
	}
	return &b, b.Checksum == b.CalculateChecksum(), nil
}

func Encode(w io.Writer, b *Bookmarks) error {
	e := json.NewEncoder(w)
	e.SetIndent("", "   ")
	e.SetEscapeHTML(false)
	return e.Encode(b)
}

func (b Bookmarks) CalculateChecksum() string {
	h := md5.New()
	b.Walk(func(n BookmarkNode, parents ...string) error {
		switch n.Type {
		case "url":
			h.Write([]byte(strconv.Itoa(n.ID)))
			h.Write(u16string(n.Name))
			h.Write([]byte(n.Type))
			h.Write([]byte(n.URL))
		case "folder":
			h.Write([]byte(strconv.Itoa(n.ID)))
			h.Write(u16string(n.Name))
			h.Write([]byte(n.Type))
		}
		return nil
	})
	return hex.EncodeToString(h.Sum(nil))
}

var ErrBreak = errors.New("break")

type WalkFunc func(n BookmarkNode, parents ...string) error

func (b Bookmarks) Walk(fn WalkFunc) error {
	if err := b.Roots.BookmarkBar.walk(fn); err != nil {
		if err == ErrBreak {
			err = nil
		}
		return err
	}
	if err := b.Roots.Other.walk(fn); err != nil {
		if err == ErrBreak {
			err = nil
		}
		return err
	}
	if err := b.Roots.MobileBookmark.walk(fn); err != nil {
		if err == ErrBreak {
			err = nil
		}
		return err
	}
	return nil
}

func (n BookmarkNode) Walk(fn WalkFunc) error {
	if err := n.walk(fn); err != nil && err != ErrBreak {
		return err
	}
	return nil
}

func (n BookmarkNode) walk(fn WalkFunc, parents ...string) error {
	if err := fn(n); err != nil {
		return err
	}
	if n.Type == NodeTypeFolder {
		for _, c := range *n.Children {
			if err := c.walk(fn, append(parents, c.Name)...); err != nil {
				return err
			}
		}
	}
	return nil
}

type Version int

const CurrentVersion Version = 1

func (v Version) Valid() error {
	if v != 1 {
		return fmt.Errorf("unsupported bookmarks version %d", v)
	}
	return nil
}

func (v Version) MarshalJSON() ([]byte, error) {
	if err := v.Valid(); err != nil {
		return nil, err
	}
	return json.Marshal(int(v))
}

func (v *Version) UnmarshalJSON(b []byte) error {
	if err := json.Unmarshal(b, (*int)(v)); err != nil {
		return err
	}
	if err := v.Valid(); err != nil {
		return err
	}
	return nil
}

type NodeType string

const (
	NodeTypeURL    NodeType = "url"
	NodeTypeFolder NodeType = "folder"
)

func (t NodeType) Valid() error {
	switch t {
	case NodeTypeURL, NodeTypeFolder:
		return nil
	default:
		return fmt.Errorf("unrecognized node type %q", string(t))
	}
}

func (t NodeType) MarshalJSON() ([]byte, error) {
	if err := t.Valid(); err != nil {
		return nil, err
	}
	return json.Marshal(string(t))
}

func (t *NodeType) UnmarshalJSON(b []byte) error {
	if err := json.Unmarshal(b, (*string)(t)); err != nil {
		return err
	}
	if err := t.Valid(); err != nil {
		return err
	}
	return nil
}

type Bytes []byte

func (c Bytes) MarshalJSON() ([]byte, error) {
	switch {
	case c == nil:
		return []byte(`null`), nil
	case len(c) == 0:
		return []byte(`""`), nil
	default:
		return json.Marshal(base64.StdEncoding.EncodeToString(c))
	}
}

func (c *Bytes) UnmarshalJSON(b []byte) error {
	var s *string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	switch {
	case s == nil:
		*c = nil
	case *s == "":
		*c = []byte{}
	default:
		if b, err := base64.StdEncoding.DecodeString(*s); err != nil {
			return err
		} else {
			*c = b
		}
	}
	return nil
}

type Time int64

const crTimeEpochDelta int64 = 11644473600 // seconds

func (t Time) IsZero() bool {
	return t == 0
}

func (t Time) MarshalJSON() ([]byte, error) {
	if t.IsZero() {
		return nil, nil
	}
	return json.Marshal(strconv.FormatInt(int64(t), 10))
}

func (t *Time) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	if v, err := strconv.ParseInt(s, 10, 64); err != nil {
		return err
	} else {
		*t = Time(v)
	}
	return nil
}

func (t *Time) SetTime(x time.Time) {
	*t = Time(x.UnixMicro() - (crTimeEpochDelta * 1000 * 1000))
}

func (t Time) Time() time.Time {
	// microsecond time with an epoch of 1601-01-01 UTC
	return time.Unix(int64(t)/1000/1000-crTimeEpochDelta, int64(t)%(1000*1000))
}

func (t Time) String() string {
	return t.Time().String()
}

type GUID string

func (s GUID) String() string {
	v, _ := s.Canonical()
	return v
}

func (s GUID) Valid() bool {
	_, ok := s.Bytes()
	return ok
}

func (s GUID) Canonical() (string, bool) {
	u, ok := s.Bytes()
	return fmt.Sprintf("%x-%x-%x-%x-%x", u[0:4], u[4:6], u[6:8], u[8:10], u[10:]), ok
}

func (s GUID) Bytes() ([16]byte, bool) {
	var u [16]byte
	if len(s) != 36 || s[8] != '-' || s[13] != '-' || s[18] != '-' || s[23] != '-' {
		return u, false
	}
	for i, x := range [16]int{0, 2, 4, 6, 9, 11, 14, 16, 19, 21, 24, 26, 28, 30, 32, 34} {
		v, ok := xtob(s[x], s[x+1])
		if !ok {
			return u, false
		}
		u[i] = v
	}
	return u, true
}

func u16string(s string) []byte {
	codes := utf16.Encode([]rune(s))
	b := make([]byte, len(codes)*2)
	for i, r := range codes {
		b[i*2] = byte(r)
		b[i*2+1] = byte(r >> 8)
	}
	return b
}

var xv = [256]byte{
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 255, 255, 255, 255, 255, 255,
	255, 10, 11, 12, 13, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 10, 11, 12, 13, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
}

func xtob(x1, x2 byte) (byte, bool) {
	b1 := xv[x1]
	b2 := xv[x2]
	return (b1 << 4) | b2, b1 != 255 && b2 != 255
}
