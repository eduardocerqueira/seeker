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

	// 🔥 優化: Buffer 大小常量
	DefaultBufferSize = 8 * 1024      // 8KB - Raspberry Pi 友好
	MaxBufferSize     = 64 * 1024     // 64KB - 最大緩衝
	TagHeaderSize     = 11
	FlvHeaderSize     = 9
	PrevTagSizeBytes  = 4
)

var (
	FlvHeader = []byte{'F', 'L', 'V', 0x01, 0x05, 0x00, 0x00, 0x00, 0x09}

	ErrNotFlvFile = errors.New("not a valid FLV file")
	ErrInvalidTag = errors.New("invalid FLV tag")

	// 🔥 優化: sync.Pool 用於復用 buffer 和對象
	byteBufferPool = sync.Pool{
		New: func() interface{} {
			return bytes.NewBuffer(make([]byte, 0, DefaultBufferSize))
		},
	}

	tagPool = sync.Pool{
		New: func() interface{} {
			return &Tag{}
		},
	}

	headerBytesPool = sync.Pool{
		New: func() interface{} {
			b := make([]byte, TagHeaderSize)
			return &b
		},
	}

	smallBytesPool = sync.Pool{
		New: func() interface{} {
			b := make([]byte, PrevTagSizeBytes)
			return &b
		},
	}
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

// 🔥 優化:  重置 Tag 以便復用
func (t *Tag) Reset() {
	t.Type = 0
	t.DataSize = 0
	t. Timestamp = 0
	t.StreamID = [3]byte{0, 0, 0}
	t.Data = nil
	t.IsHeader = false
	t.IsKeyframe = false
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
		buffer:        byteBufferPool.Get().(*bytes.Buffer), // 🔥 優化: 從 pool 取得
		headerWritten: false,
	}
}

// 🔥 優化:  釋放資源
func (rf *RealtimeFixer) Close() {
	rf.mu.Lock()
	defer rf.mu.Unlock()
	
	if rf.buffer != nil {
		rf.buffer.Reset()
		byteBufferPool.Put(rf.buffer)
		rf.buffer = nil
	}
}

func (rf *RealtimeFixer) Fix(input []byte) ([]byte, error) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	rf.buffer. Write(input)
	
	// 🔥 優化: 從 pool 取得輸出 buffer
	output := byteBufferPool.Get().(*bytes.Buffer)
	output.Reset()
	defer func() {
		// 注意: output 會被返回，所以不能立即放回 pool
		// 調用者需要負責處理
	}()

	// Write FLV header once
	if !rf.headerWritten && rf.buffer. Len() >= FlvHeaderSize {
		header := rf.buffer.Next(FlvHeaderSize)
		if !bytes.Equal(header[: 3], []byte{'F', 'L', 'V'}) {
			return nil, ErrNotFlvFile
		}
		output.Write(header)
		output.Write([]byte{0, 0, 0, 0})
		rf.headerWritten = true
	}

	// Parse complete tags from buffer
	for {
		if rf.buffer.Len() < 15 {
			break
		}

		// 🔥 優化: 避免完整拷貝，使用切片
		bufLen := rf.buffer.Len()
		
		// Skip PreviousTagSize
		rf.buffer.Next(PrevTagSizeBytes)

		if rf.buffer.Len() < TagHeaderSize {
			// 🔥 優化:  使用 Grow + 手動回退而非完整拷貝
			rf.buffer.Reset()
			remaining := input[len(input)-(bufLen):]
			rf.buffer.Write(remaining)
			break
		}

		// 🔥 優化: 從 pool 取得 header buffer
		headerBytesPtr := headerBytesPool.Get().(*[]byte)
		headerBytes := *headerBytesPtr
		rf.buffer.Read(headerBytes)

		tagType := headerBytes[0]
		dataSize := uint32(headerBytes[1])<<16 | uint32(headerBytes[2])<<8 | uint32(headerBytes[3])

		if rf.buffer.Len() < int(dataSize) {
			// 恢復:  需要更多數據
			headerBytesPool.Put(headerBytesPtr)
			
			// 🔥 優化: 手動構建最小恢復
			tempBuf := byteBufferPool.Get().(*bytes.Buffer)
			tempBuf.Reset()
			tempBuf.Write([]byte{0, 0, 0, 0}) // PrevTagSize
			tempBuf.Write(headerBytes)
			tempBuf.Write(rf.buffer. Bytes())
			
			rf.buffer. Reset()
			rf.buffer.Write(tempBuf.Bytes())
			
			tempBuf.Reset()
			byteBufferPool.Put(tempBuf)
			break
		}

		// 🔥 優化: 複用 tag 對象但數據還是要拷貝 (因為會被修改)
		tagData := make([]byte, dataSize)
		rf.buffer.Read(tagData)

		timestamp := int32(headerBytes[7])<<24 | int32(headerBytes[4])<<16 |
			int32(headerBytes[5])<<8 | int32(headerBytes[6])

		tag := tagPool.Get().(*Tag)
		tag.Reset()
		tag.Type = tagType
		tag.DataSize = dataSize
		tag.Timestamp = timestamp
		tag.Data = tagData
		copy(tag.StreamID[:], headerBytes[8:11])

		// 返還 header buffer
		headerBytesPool.Put(headerBytesPtr)

		// 檢測標誌
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

		if err := writeTagOptimized(output, tag); err != nil {
			tagPool.Put(tag)
			return nil, err
		}

		// 🔥 優化:  返還 tag 到 pool (但保留 Data 因為已經寫入)
		tagPool.Put(tag)
	}

	// 🔥 優化:  返回複製的數據，這樣 output buffer 可以被復用
	result := make([]byte, output.Len())
	copy(result, output.Bytes())
	
	output.Reset()
	byteBufferPool.Put(output)

	return result, nil
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
	buffer         *bytes.Buffer
	chunkSizeBytes int
	headerWritten  bool
	totalProcessed int64
	
	// 🔥 優化: 預分配 tag slice
	tagCache       []*Tag
	tagCacheSize   int
}

func NewAccumulateFixer(chunkSizeMB int) *AccumulateFixer {
	// 🔥 優化:  估算可能的 tag 數量並預分配
	estimatedTags := (chunkSizeMB * 1024 * 1024) / 1024 // 假設平均 1KB/tag
	
	return &AccumulateFixer{
		tsStore:        &TimestampStore{FirstChunk: true},
		buffer:         byteBufferPool.Get().(*bytes.Buffer),
		chunkSizeBytes:  chunkSizeMB * 1024 * 1024,
		headerWritten:  false,
		tagCache:       make([]*Tag, 0, estimatedTags),
		tagCacheSize:   estimatedTags,
	}
}

// 🔥 優化:  釋放資源
func (af *AccumulateFixer) Close() {
	af.mu.Lock()
	defer af.mu.Unlock()
	
	if af.buffer != nil {
		af.buffer.Reset()
		byteBufferPool.Put(af.buffer)
		af.buffer = nil
	}
	
	// 返還所有 tag 到 pool
	for _, tag := range af. tagCache {
		if tag != nil {
			tagPool. Put(tag)
		}
	}
	af.tagCache = nil
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

	return af.flushInternal()
}

func (af *AccumulateFixer) flushInternal() ([]byte, error) {
	if af.buffer.Len() == 0 {
		return nil, nil
	}

	// 🔥 優化: 從 pool 取得 output buffer
	output := byteBufferPool.Get().(*bytes.Buffer)
	output.Reset()

	// Write header once globally (not per flush)
	if !af.headerWritten {
		if af.buffer.Len() < FlvHeaderSize {
			byteBufferPool.Put(output)
			return nil, nil
		}

		header := make([]byte, FlvHeaderSize)
		copy(header, af.buffer. Bytes()[:FlvHeaderSize])

		if !bytes.Equal(header[:3], []byte{'F', 'L', 'V'}) {
			byteBufferPool.Put(output)
			return nil, ErrNotFlvFile
		}

		output.Write(header)
		output.Write([]byte{0, 0, 0, 0})
		af.headerWritten = true
		af.buffer.Next(FlvHeaderSize)
	}

	// 🔥 優化: 重用 tag cache
	tags := af.tagCache[:0] // 保留容量，清空長度

	for af.buffer.Len() >= 15 {
		startLen := af.buffer.Len()

		// 🔥 優化: 從 pool 取得小 buffer
		prevTagSizeBytesPtr := smallBytesPool.Get().(*[]byte)
		prevTagSizeBytes := *prevTagSizeBytesPtr
		af.buffer.Read(prevTagSizeBytes)

		if af.buffer.Len() < TagHeaderSize {
			// Restore
			tempBuf := byteBufferPool.Get().(*bytes.Buffer)
			tempBuf.Reset()
			tempBuf.Write(prevTagSizeBytes)
			tempBuf.Write(af.buffer.Bytes())
			af.buffer.Reset()
			af.buffer.Write(tempBuf.Bytes())
			tempBuf.Reset()
			byteBufferPool.Put(tempBuf)
			smallBytesPool.Put(prevTagSizeBytesPtr)
			break
		}

		headerBytesPtr := headerBytesPool.Get().(*[]byte)
		headerBytes := *headerBytesPtr
		af. buffer.Read(headerBytes)

		dataSize := uint32(headerBytes[1])<<16 | uint32(headerBytes[2])<<8 | uint32(headerBytes[3])

		if af.buffer.Len() < int(dataSize) {
			// Incomplete tag, restore buffer
			tempBuf := byteBufferPool.Get().(*bytes.Buffer)
			tempBuf.Reset()
			tempBuf.Write(prevTagSizeBytes)
			tempBuf.Write(headerBytes)
			tempBuf.Write(af.buffer.Bytes())
			af.buffer.Reset()
			af.buffer.Write(tempBuf.Bytes())
			tempBuf.Reset()
			byteBufferPool. Put(tempBuf)
			headerBytesPool.Put(headerBytesPtr)
			smallBytesPool.Put(prevTagSizeBytesPtr)
			break
		}

		tagData := make([]byte, dataSize)
		af.buffer.Read(tagData)

		timestamp := int32(headerBytes[7])<<24 | int32(headerBytes[4])<<16 |
			int32(headerBytes[5])<<8 | int32(headerBytes[6])

		// 🔥 優化:  從 pool 取得 tag
		tag := tagPool.Get().(*Tag)
		tag.Reset()
		tag.Type = headerBytes[0]
		tag.DataSize = dataSize
		tag.Timestamp = timestamp
		tag.Data = tagData
		copy(tag. StreamID[:], headerBytes[8:11])

		if len(tagData) >= 2 {
			if tag.Type == TagTypeVideo {
				tag.IsKeyframe = (tagData[0] & 0xF0) == 0x10
				tag. IsHeader = tagData[1] == 0x00
			} else if tag.Type == TagTypeAudio && (tagData[0]>>4) == 10 {
				tag.IsHeader = tagData[1] == 0x00
			}
		}

		tags = append(tags, tag)

		headerBytesPool.Put(headerBytesPtr)
		smallBytesPool.Put(prevTagSizeBytesPtr)

		// Safety check
		if af.buffer.Len() > startLen {
			byteBufferPool.Put(output)
			return nil, errors.New("buffer corruption detected")
		}
	}

	// Fix timestamps for all tags
	af.fixTimestamps(tags)

	// Write all fixed tags
	for _, tag := range tags {
		if err := writeTagOptimized(output, tag); err != nil {
			byteBufferPool.Put(output)
			return nil, err
		}
	}

	af.totalProcessed += int64(output.Len())

	// 🔥 優化: 保存 tag cache 供下次使用
	af.tagCache = tags

	// 🔥 優化: 返回複製的數據
	result := make([]byte, output.Len())
	copy(result, output.Bytes())
	
	output.Reset()
	byteBufferPool. Put(output)

	return result, nil
}

func (af *AccumulateFixer) fixTimestamps(tags []*Tag) {
	if len(tags) == 0 {
		return
	}

	ts := af.tsStore

	// First chunk: find minimum timestamp
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
// Helper:  Write Tag to Stream (優化版本)
// =====================================================

func writeTagOptimized(w io.Writer, tag *Tag) error {
	// 🔥 優化: 從 pool 取得 header buffer
	headerPtr := headerBytesPool.Get().(*[]byte)
	header := *headerPtr
	defer headerBytesPool.Put(headerPtr)

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

	// 🔥 優化:  從 pool 取得 prevTagSize buffer
	prevTagSizePtr := smallBytesPool.Get().(*[]byte)
	prevTagSize := *prevTagSizePtr
	defer smallBytesPool.Put(prevTagSizePtr)

	binary.BigEndian.PutUint32(prevTagSize, uint32(11+len(tag.Data)))
	if _, err := w.Write(prevTagSize); err != nil {
		return err
	}

	return nil
}

// 🔥 保留舊版本以保持兼容性
func writeTag(w io.Writer, tag *Tag) error {
	return writeTagOptimized(w, tag)
}

// =====================================================
// Advanced: Group-based Timestamp Calculation
// =====================================================

func calculateNextTargetAdvanced(tags []*Tag) int32 {
	// 🔥 優化:  預分配精確大小的 slice
	videoCount := 0
	audioCount := 0
	for _, tag := range tags {
		if tag.Type == TagTypeVideo {
			videoCount++
		} else if tag.Type == TagTypeAudio {
			audioCount++
		}
	}

	videoTags := make([]*Tag, 0, videoCount)
	audioTags := make([]*Tag, 0, audioCount)

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
		duration := audioTags[1].Timestamp - audioTags[0]. Timestamp
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