//date: 2025-12-31T16:43:01Z
//url: https://api.github.com/gists/275fa7d5a9855d4548877ea024bc4f3b
//owner: https://api.github.com/users/eric2788

package flvfixer

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"sync"
)

const (
	TagTypeAudio  = 0x08
	TagTypeVideo  = 0x09
	TagTypeScript = 0x12

	JumpThreshold = 500

	AudioDurationFallback = 22
	AudioDurationMin      = 20
	AudioDurationMax      = 24

	VideoDurationFallback = 33
	VideoDurationMin      = 15
	VideoDurationMax      = 50
)

var (
	FlvHeader = []byte{'F', 'L', 'V', 0x01, 0x05, 0x00, 0x00, 0x00, 0x09}

	ErrNotFlvFile = errors.New("not a valid FLV file")
	ErrInvalidTag = errors.New("invalid FLV tag")
)

type Tag struct {
	Type       byte
	DataSize   uint32
	Timestamp  int32
	StreamID   [3]byte
	Data       []byte
	IsHeader   bool
	IsKeyframe bool
}

type TimestampStore struct {
	FirstChunk          bool
	LastOriginal        int32
	CurrentOffset       int32
	NextTimestampTarget int32
}

func (ts *TimestampStore) Reset() {
	ts.FirstChunk = true
	ts.LastOriginal = 0
	ts.CurrentOffset = 0
	ts.NextTimestampTarget = 0
}

// =====================================================
// REALTIME FIXER - 逐個 Tag 修復並輸出
// =====================================================

type RealtimeFixer struct {
	mu            sync. Mutex
	tsStore       *TimestampStore
	buffer        *bytes.Buffer
	headerWritten bool
}

func NewRealtimeFixer() *RealtimeFixer {
	return &RealtimeFixer{
		tsStore:        &TimestampStore{FirstChunk: true},
		buffer:        new(bytes.Buffer),
		headerWritten: false,
	}
}

func (rf *RealtimeFixer) Fix(input []byte) ([]byte, error) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	rf.buffer.Write(input)
	output := new(bytes.Buffer)

	// Write FLV header once
	if ! rf.headerWritten && rf. buffer.Len() >= 9 {
		header := rf. buffer.Next(9)
		if !bytes.Equal(header[: 3], []byte{'F', 'L', 'V'}) {
			return nil, ErrNotFlvFile
		}
		output. Write(header)
		output.Write([]byte{0, 0, 0, 0})
		rf.headerWritten = true
	}

	// Parse complete tags from buffer
	for {
		if rf.buffer.Len() < 15 {
			break
		}

		savedBuffer := make([]byte, rf.buffer. Len())
		copy(savedBuffer, rf.buffer.Bytes())

		rf.buffer.Next(4) // Skip PreviousTagSize

		if rf.buffer.Len() < 11 {
			rf.buffer = bytes.NewBuffer(savedBuffer)
			break
		}

		headerBytes := make([]byte, 11)
		rf.buffer.Read(headerBytes)

		tagType := headerBytes[0]
		dataSize := uint32(headerBytes[1])<<16 | uint32(headerBytes[2])<<8 | uint32(headerBytes[3])

		if rf.buffer.Len() < int(dataSize) {
			rf.buffer = bytes.NewBuffer(savedBuffer)
			break
		}

		tagData := make([]byte, dataSize)
		rf.buffer.Read(tagData)

		timestamp := int32(headerBytes[7])<<24 | int32(headerBytes[4])<<16 |
			int32(headerBytes[5])<<8 | int32(headerBytes[6])

		tag := &Tag{
			Type:      tagType,
			DataSize:  dataSize,
			Timestamp:  timestamp,
			Data:      tagData,
		}
		copy(tag.StreamID[:], headerBytes[8:11])

		if len(tagData) >= 2 {
			if tagType == TagTypeVideo {
				firstByte := tagData[0]
				secondByte := tagData[1]
				tag.IsKeyframe = (firstByte & 0xF0) == 0x10
				tag.IsHeader = secondByte == 0x00
			} else if tagType == TagTypeAudio {
				if (tagData[0]>>4) == 10 && len(tagData) >= 2 {
					tag.IsHeader = tagData[1] == 0x00
				}
			}
		}

		rf.fixTimestamp(tag)

		if err := writeTag(output, tag); err != nil {
			return nil, err
		}
	}

	return output.Bytes(), nil
}

func (rf *RealtimeFixer) fixTimestamp(tag *Tag) {
	ts := rf.tsStore
	currentTimestamp := tag.Timestamp

	if ts.FirstChunk {
		ts.FirstChunk = false
		ts.CurrentOffset = currentTimestamp
	}

	diff := currentTimestamp - ts.LastOriginal

	if diff < -JumpThreshold || (ts.LastOriginal == 0 && diff < 0) {
		ts.CurrentOffset = currentTimestamp - ts.NextTimestampTarget
	} else if diff > JumpThreshold {
		ts.CurrentOffset = currentTimestamp - ts.NextTimestampTarget
	}

	ts.LastOriginal = currentTimestamp
	tag.Timestamp -= ts.CurrentOffset
	ts.NextTimestampTarget = calculateNextTarget(tag, ts.NextTimestampTarget)
}

func calculateNextTarget(tag *Tag, currentTarget int32) int32 {
	duration := int32(VideoDurationFallback)
	if tag.Type == TagTypeAudio {
		duration = AudioDurationFallback
	}
	return tag.Timestamp + duration
}

// =====================================================
// ACCUMULATE FIXER - 累積 X MB 後批次處理
// =====================================================

type AccumulateFixer struct {
	mu             sync.Mutex
	tsStore        *TimestampStore
	buffer         *bytes. Buffer
	chunkSizeBytes int
	headerWritten  bool
	totalProcessed int64
}

func NewAccumulateFixer(chunkSizeMB int) *AccumulateFixer {
	return &AccumulateFixer{
		tsStore:        &TimestampStore{FirstChunk: true},
		buffer:         new(bytes. Buffer),
		chunkSizeBytes: chunkSizeMB * 1024 * 1024,
		headerWritten:   false,
	}
}

// Accumulate adds data and returns true if ready to flush
func (af *AccumulateFixer) Accumulate(input []byte) (bool, error) {
	af.mu.Lock()
	defer af.mu.Unlock()

	af.buffer.Write(input)
	return af.buffer.Len() >= af.chunkSizeBytes, nil
}

// Flush processes accumulated data (call this when Accumulate returns true OR at EOF)
func (af *AccumulateFixer) Flush() ([]byte, error) {
	af.mu.Lock()
	defer af.mu.Unlock()

	return af.flushInternal()
}

// FlushRemaining processes all remaining data (call at EOF)
func (af *AccumulateFixer) FlushRemaining() ([]byte, error) {
	af.mu.Lock()
	defer af.mu. Unlock()

	// Force flush even if buffer is small
	return af.flushInternal()
}

func (af *AccumulateFixer) flushInternal() ([]byte, error) {
	if af.buffer. Len() == 0 {
		return nil, nil
	}

	output := new(bytes.Buffer)

	// Write header once globally (not per flush)
	if !af.headerWritten {
		if af.buffer.Len() < 9 {
			// Not enough data yet, keep waiting
			return nil, nil
		}

		header := make([]byte, 9)
		copy(header, af.buffer. Bytes()[:9])

		if ! bytes.Equal(header[:3], []byte{'F', 'L', 'V'}) {
			return nil, ErrNotFlvFile
		}

		output.Write(header)
		output.Write([]byte{0, 0, 0, 0})
		af.headerWritten = true
		af.buffer. Next(9) // Consume header from buffer
	}

	// Parse all complete tags
	tags := make([]*Tag, 0, 100)

	for af.buffer.Len() >= 15 {
		startLen := af.buffer.Len()

		// Skip PreviousTagSize
		prevTagSizeBytes := make([]byte, 4)
		af.buffer.Read(prevTagSizeBytes)

		if af.buffer.Len() < 11 {
			// Restore
			af.buffer = bytes.NewBuffer(append(prevTagSizeBytes, af.buffer.Bytes()...))
			break
		}

		headerBytes := make([]byte, 11)
		af.buffer.Read(headerBytes)

		dataSize := uint32(headerBytes[1])<<16 | uint32(headerBytes[2])<<8 | uint32(headerBytes[3])

		if af.buffer.Len() < int(dataSize) {
			// Incomplete tag, restore buffer
			restoredData := append(prevTagSizeBytes, headerBytes...)
			restoredData = append(restoredData, af.buffer. Bytes()...)
			af.buffer = bytes.NewBuffer(restoredData)
			break
		}

		tagData := make([]byte, dataSize)
		af.buffer.Read(tagData)

		timestamp := int32(headerBytes[7])<<24 | int32(headerBytes[4])<<16 |
			int32(headerBytes[5])<<8 | int32(headerBytes[6])

		tag := &Tag{
			Type:      headerBytes[0],
			DataSize:  dataSize,
			Timestamp: timestamp,
			Data:       tagData,
		}
		copy(tag.StreamID[: ], headerBytes[8:11])

		if len(tagData) >= 2 {
			if tag.Type == TagTypeVideo {
				tag.IsKeyframe = (tagData[0] & 0xF0) == 0x10
				tag. IsHeader = tagData[1] == 0x00
			} else if tag.Type == TagTypeAudio && (tagData[0]>>4) == 10 {
				tag.IsHeader = tagData[1] == 0x00
			}
		}

		tags = append(tags, tag)

		// Safety check
		if af.buffer.Len() > startLen {
			return nil, errors.New("buffer corruption detected")
		}
	}

	// Fix timestamps for all tags
	af.fixTimestamps(tags)

	// Write all fixed tags
	for _, tag := range tags {
		if err := writeTag(output, tag); err != nil {
			return nil, err
		}
	}

	af.totalProcessed += int64(output.Len())

	return output.Bytes(), nil
}

func (af *AccumulateFixer) fixTimestamps(tags []*Tag) {
	if len(tags) == 0 {
		return
	}

	ts := af.tsStore

	// First chunk:  find minimum timestamp
	if ts.FirstChunk {
		ts.FirstChunk = false
		minTs := tags[0].Timestamp
		for _, t := range tags {
			if t.Timestamp < minTs {
				minTs = t.Timestamp
			}
		}
		ts.CurrentOffset = minTs
	}

	for _, tag := range tags {
		currentTimestamp := tag.Timestamp
		diff := currentTimestamp - ts.LastOriginal

		if diff < -JumpThreshold || (ts. LastOriginal == 0 && diff < 0) {
			ts.CurrentOffset = currentTimestamp - ts.NextTimestampTarget
		} else if diff > JumpThreshold {
			ts. CurrentOffset = currentTimestamp - ts.NextTimestampTarget
		}

		ts.LastOriginal = tag.Timestamp
		tag. Timestamp -= ts.CurrentOffset
	}
	
	ts.NextTimestampTarget = calculateNextTargetAdvanced(tags)
}

// GetStats returns processing statistics
func (af *AccumulateFixer) GetStats() (buffered int, processed int64) {
	af.mu.Lock()
	defer af.mu.Unlock()
	return af.buffer.Len(), af.totalProcessed
}

// =====================================================
// Helper:  Write Tag to Stream
// =====================================================

func writeTag(w io.Writer, tag *Tag) error {
	header := make([]byte, 11)
	header[0] = tag.Type

	header[1] = byte(tag. DataSize >> 16)
	header[2] = byte(tag.DataSize >> 8)
	header[3] = byte(tag.DataSize)

	header[4] = byte(tag. Timestamp >> 16)
	header[5] = byte(tag.Timestamp >> 8)
	header[6] = byte(tag.Timestamp)
	header[7] = byte(tag.Timestamp >> 24)

	copy(header[8:11], tag.StreamID[:])

	if _, err := w.Write(header); err != nil {
		return err
	}

	if _, err := w.Write(tag.Data); err != nil {
		return err
	}

	prevTagSize := make([]byte, 4)
	binary.BigEndian.PutUint32(prevTagSize, uint32(11+len(tag.Data)))
	if _, err := w.Write(prevTagSize); err != nil {
		return err
	}

	return nil
}

// =====================================================
// Advanced: Group-based Timestamp Calculation
// =====================================================

func calculateNextTargetAdvanced(tags []*Tag) int32 {
	videoTags := make([]*Tag, 0)
	audioTags := make([]*Tag, 0)

	for _, tag := range tags {
		if tag.Type == TagTypeVideo {
			videoTags = append(videoTags, tag)
		} else if tag.Type == TagTypeAudio {
			audioTags = append(audioTags, tag)
		}
	}

	videoDuration := int32(0)
	if len(videoTags) >= 2 {
		duration := videoTags[1].Timestamp - videoTags[0].Timestamp
		if duration >= VideoDurationMin && duration <= VideoDurationMax {
			videoDuration = videoTags[len(videoTags)-1].Timestamp + duration
		} else {
			videoDuration = videoTags[len(videoTags)-1].Timestamp + VideoDurationFallback
		}
	} else if len(videoTags) == 1 {
		videoDuration = videoTags[0]. Timestamp + VideoDurationFallback
	}

	audioDuration := int32(0)
	if len(audioTags) >= 2 {
		duration := audioTags[1]. Timestamp - audioTags[0]. Timestamp
		if duration >= AudioDurationMin && duration <= AudioDurationMax {
			audioDuration = audioTags[len(audioTags)-1].Timestamp + duration
		} else {
			audioDuration = audioTags[len(audioTags)-1].Timestamp + AudioDurationFallback
		}
	} else if len(audioTags) == 1 {
		audioDuration = audioTags[0]. Timestamp + AudioDurationFallback
	}

	return int32(math.Max(float64(videoDuration), float64(audioDuration)))
}