//date: 2026-03-09T17:33:31Z
//url: https://api.github.com/gists/903ced297d388200805900b368e643cd
//owner: https://api.github.com/users/garudamods

// ScreenAI OCR [Go Implementation]
// Copyright (C) 2026 Fazx - Garudamods
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// this program. If not, see <https://www.gnu.org/licenses/>.
//
// ----------------------------------------------------------------------------
// Based on:
//   AuroraWright (https://github.com/AuroraWright/owocr)
//   Source: https://github.com/AuroraWright/owocr/blob/1.26.1/owocr/ocr.py#L820
//   License: GNU General Public License v3.0
//
// This file is a rewrite of the Chr*me Screen AI OCR integration originally
// implemented in Python by AuroraWright as part of the owocr project.
// The Go rewrite was authored by Fazx - Garudamods.
// ----------------------------------------------------------------------------

package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

type skColorInfo struct {
	fColorSpace uintptr
	fColorType  int32
	fAlphaType  int32
}

type skISize struct {
	fWidth  int32
	fHeight int32
}

type skImageInfo struct {
	fColorInfo  skColorInfo
	fDimensions skISize
}

type skPixmap struct {
	fPixels   uintptr
	fRowBytes uintptr
	fInfo     skImageInfo
}

type skBitmap struct {
	fPixelRef uintptr
	fPixmap   skPixmap
	fFlags    uint32
	_         [4]byte
}

var version = "v1.0.0"

var (
	screenAILib                       *syscall.LazyDLL
	procSetFileContentFunctions       *syscall.LazyProc
	procInitOCRUsingCallback          *syscall.LazyProc
	procSetOCRLightMode               *syscall.LazyProc
	procPerformOCR                    *syscall.LazyProc
	procFreeLibraryAllocatedCharArray *syscall.LazyProc

	gModelDir string

	ocrMu sync.Mutex

	workerMgr *ocrWorkerMgr
)

type protoFields map[int][]interface{}

func readVarint(data []byte, pos int) (uint64, int) {
	var result uint64
	var shift uint
	for pos < len(data) {
		b := data[pos]
		pos++
		result |= uint64(b&0x7F) << shift
		shift += 7
		if b&0x80 == 0 {
			break
		}
	}
	return result, pos
}

func parseProto(data []byte) protoFields {
	f := make(protoFields)
	pos := 0
	for pos < len(data) {
		start := pos
		tag, npos := readVarint(data, pos)
		if npos == start {
			break
		}
		pos = npos
		fn := int(tag >> 3)
		wt := tag & 0x7

		var val interface{}
		switch wt {
		case 0:
			v, np := readVarint(data, pos)
			pos = np
			val = v
		case 1:
			if pos+8 > len(data) {
				return f
			}
			val = binary.LittleEndian.Uint64(data[pos:])
			pos += 8
		case 2:
			length, np := readVarint(data, pos)
			pos = np
			end := pos + int(length)
			if end > len(data) {
				return f
			}
			b := make([]byte, length)
			copy(b, data[pos:end])
			pos = end
			val = b
		case 5:
			if pos+4 > len(data) {
				return f
			}
			val = binary.LittleEndian.Uint32(data[pos:])
			pos += 4
		default:
			return f
		}
		f[fn] = append(f[fn], val)
	}
	return f
}

func pBytes(f protoFields, fn int) []byte {
	if vals, ok := f[fn]; ok && len(vals) > 0 {
		if b, ok := vals[0].([]byte); ok {
			return b
		}
	}
	return nil
}

func pString(f protoFields, fn int) string {
	b := pBytes(f, fn)
	if b == nil {
		return ""
	}
	return string(b)
}

func pUint64(f protoFields, fn int) uint64 {
	if vals, ok := f[fn]; ok && len(vals) > 0 {
		switch v := vals[0].(type) {
		case uint64:
			return v
		case uint32:
			return uint64(v)
		}
	}
	return 0
}

func pFixed32(f protoFields, fn int) uint32 {
	if vals, ok := f[fn]; ok && len(vals) > 0 {
		switch v := vals[0].(type) {
		case uint32:
			return v
		case uint64:
			return uint32(v)
		}
	}
	return 0
}

type BBox struct {
	X      int     `json:"x"`
	Y      int     `json:"y"`
	Width  int     `json:"width"`
	Height int     `json:"height"`
	Angle  float32 `json:"angle"`
}

type Word struct {
	Text       string  `json:"text"`
	BBox       BBox    `json:"bbox"`
	Confidence float32 `json:"confidence"`
}

type Line struct {
	Text       string  `json:"text"`
	BBox       BBox    `json:"bbox"`
	X          int     `json:"x"`
	Y          int     `json:"y"`
	W          int     `json:"w"`
	H          int     `json:"h"`
	A          float32 `json:"a"`
	Language   string  `json:"language"`
	Confidence float32 `json:"confidence"`
	Words      []Word  `json:"words,omitempty"`
}

func parseBBox(data []byte) BBox {
	if len(data) < 2 {
		return BBox{}
	}
	f := parseProto(data)
	return BBox{
		X:      int(pUint64(f, 1)),
		Y:      int(pUint64(f, 2)),
		Width:  int(pUint64(f, 3)),
		Height: int(pUint64(f, 4)),
		Angle:  math.Float32frombits(pFixed32(f, 5)),
	}
}

func parseWordProto(data []byte) Word {
	f := parseProto(data)
	bboxData := pBytes(f, 1)
	if bboxData == nil {
		bboxData = []byte{}
	}
	return Word{
		Text:       pString(f, 2),
		BBox:       parseBBox(bboxData),
		Confidence: math.Float32frombits(pFixed32(f, 3)),
	}
}

func parseLineProto(data []byte) Line {
	f := parseProto(data)
	bboxData := pBytes(f, 2)
	if bboxData == nil {
		bboxData = []byte{}
	}
	bboxParse := parseBBox(bboxData)
	line := Line{
		Text:       pString(f, 3),
		BBox:       bboxParse,
		X:          bboxParse.X,
		Y:          bboxParse.Y,
		W:          bboxParse.Width,
		H:          bboxParse.Height,
		A:          bboxParse.Angle,
		Language:   pString(f, 4),
		Confidence: math.Float32frombits(pFixed32(f, 10)),
	}

	if innerData := pBytes(f, 1); innerData != nil {
		inner := parseProto(innerData)
		for _, v := range inner[1] {
			if wData, ok := v.([]byte); ok {
				line.Words = append(line.Words, parseWordProto(wData))
			}
		}
	}
	return line
}

func parseVisualAnnotation(data []byte) []Line {
	f := parseProto(data)
	var lines []Line
	for _, v := range f[2] {
		if lineData, ok := v.([]byte); ok {
			lines = append(lines, parseLineProto(lineData))
		}
	}
	return lines
}

func cStr(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Pointer(ptr + uintptr(n))) != 0 {
		n++
	}
	return string((*[1 << 30]byte)(unsafe.Pointer(ptr))[:n:n])
}

const libName = "chrome_screen_ai.dll"

func searchBase(base string) (modelDir, libPath string) {
	if _, err := os.Stat(base); os.IsNotExist(err) {
		return "", ""
	}
	if entries, err := os.ReadDir(base); err == nil {
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].Name() > entries[j].Name()
		})
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			candidate := filepath.Join(base, e.Name())
			lib := filepath.Join(candidate, libName)
			if _, err := os.Stat(lib); err == nil {
				return candidate, lib
			}
		}
	}
	lib := filepath.Join(base, libName)
	if _, err := os.Stat(lib); err == nil {
		return base, lib
	}
	return "", ""
}

func downloadFileWithProgress(url, dst, label string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned %s", resp.Status)
	}

	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()

	total := resp.ContentLength
	var downloaded int64

	spinFrames := []string{"⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"}
	spinIdx := 0
	startTime := time.Now()
	lastTime := startTime
	lastBytes := int64(0)
	speed := 0.0
	const barWidth = 25

	printBar := func() {
		spin := spinFrames[spinIdx%len(spinFrames)]
		spinIdx++
		dlMB := float64(downloaded) / (1024 * 1024)
		if total > 0 {
			totalMB := float64(total) / (1024 * 1024)
			pct := int(downloaded * 100 / total)
			filled := barWidth * pct / 100
			if filled > barWidth {
				filled = barWidth
			}
			bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)
			fmt.Fprintf(os.Stderr, "\r%s %s %.1f/%.1f MB  │ %.1f MB/s  %s  %d%%   ",
				spin, label, dlMB, totalMB, speed, bar, pct)
		} else {
			fmt.Fprintf(os.Stderr, "\r%s %s %.1f MB  │ %.1f MB/s   ",
				spin, label, dlMB, speed)
		}
	}

	buf := make([]byte, 32*1024)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := f.Write(buf[:n]); writeErr != nil {
				fmt.Fprintln(os.Stderr)
				return writeErr
			}
			downloaded += int64(n)

			now := time.Now()
			if elapsed := now.Sub(lastTime).Seconds(); elapsed >= 0.15 {
				speed = float64(downloaded-lastBytes) / elapsed / (1024 * 1024)
				lastBytes = downloaded
				lastTime = now
				printBar()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			fmt.Fprintln(os.Stderr)
			return readErr
		}
	}

	if total > 0 {
		downloaded = total
	}
	speed = float64(downloaded) / time.Since(startTime).Seconds() / (1024 * 1024)
	printBar()
	fmt.Fprintln(os.Stderr)

	totalMB := float64(downloaded) / (1024 * 1024)
	if elapsed := time.Since(startTime).Seconds(); elapsed > 0 {
		log.Printf("%s done – %.1f MB in %.1fs (avg %.1f MB/s)",
			label, totalMB, elapsed, float64(downloaded)/elapsed/(1024*1024))
	}
	return nil
}

func downloadViaCIPD(exeDir string) (modelDir, libPath string) {
	const (
		packageName = "chromium/third_party/screen-ai/windows-amd64"
		cipdURL     = "https://chrome-infra-packages.appspot.com/client?platform=windows-amd64&version=latest"
	)
	targetDir := filepath.Join(exeDir, "screen_ai")

	log.Println("Screen AI not found locally. Trying to download via CIPD...")

	tmpDir, err := os.MkdirTemp("", "cipd-*")
	if err != nil {
		log.Printf("CIPD download failed: cannot create temp dir: %v", err)
		return "", ""
	}
	defer os.RemoveAll(tmpDir)

	cipdBin := filepath.Join(tmpDir, "cipd.exe")
	log.Printf("Downloading CIPD client...")
	if err := downloadFileWithProgress(cipdURL, cipdBin, "  CIPD client:"); err != nil {
		log.Printf("CIPD download failed: %v", err)
		return "", ""
	}

	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		log.Printf("CIPD download failed: cannot create target dir: %v", err)
		return "", ""
	}

	ensureContent := packageName + " latest\n"
	log.Printf("Downloading Screen AI package (%s@latest)...", packageName)

	cmd := exec.Command(cipdBin, "export", "-root", targetDir, "-ensure-file", "-")
	cmd.Stdin = strings.NewReader(ensureContent)
	cmd.Stdout = log.Writer()
	cmd.Stderr = log.Writer()

	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}

	if err := cmd.Run(); err != nil {
		log.Printf("CIPD package download failed: %v", err)
		return "", ""
	}

	log.Printf("Screen AI downloaded to: %s", targetDir)

	return searchBase(targetDir)
}

func resolveScreenAI(exeDir string) (modelDir, libPath string) {

	if mDir, lib := searchBase(filepath.Join(exeDir, "screen_ai")); mDir != "" {
		return mDir, lib
	}

	return downloadViaCIPD(exeDir)
}

func initScreenAI(libPath, mDir string) error {
	gModelDir = mDir
	log.Printf("Loading Screen AI library: %s", libPath)

	screenAILib = syscall.NewLazyDLL(libPath)
	procSetFileContentFunctions = screenAILib.NewProc("SetFileContentFunctions")
	procInitOCRUsingCallback = screenAILib.NewProc("InitOCRUsingCallback")
	procSetOCRLightMode = screenAILib.NewProc("SetOCRLightMode")
	procPerformOCR = screenAILib.NewProc("PerformOCR")
	procFreeLibraryAllocatedCharArray = screenAILib.NewProc("FreeLibraryAllocatedCharArray")

	cbSize := syscall.NewCallback(func(pathPtr uintptr) uintptr {
		info, err := os.Stat(filepath.Join(gModelDir, cStr(pathPtr)))
		if err != nil {
			return 0
		}
		return uintptr(info.Size())
	})

	cbContent := syscall.NewCallback(func(pathPtr, size, buf uintptr) uintptr {
		data, err := os.ReadFile(filepath.Join(gModelDir, cStr(pathPtr)))
		if err != nil {
			return 0
		}
		n := int(size)
		if n > len(data) {
			n = len(data)
		}
		dst := (*[1 << 30]byte)(unsafe.Pointer(buf))[:n:n]
		copy(dst, data[:n])
		return 0
	})

	procSetFileContentFunctions.Call(cbSize, cbContent)

	if ret, _, _ := procInitOCRUsingCallback.Call(); ret == 0 {
		return fmt.Errorf("InitOCRUsingCallback() failed – verify that all model files are present in: %s", mDir)
	}

	procSetOCRLightMode.Call(0)
	time.Sleep(500 * time.Millisecond)

	log.Println("ScreenAI OCR ready")
	return nil
}

func runOCR(imgData []byte, width, height int) ([]byte, error) {
	ocrMu.Lock()
	defer ocrMu.Unlock()

	bmp := skBitmap{}
	bmp.fPixmap.fPixels = uintptr(unsafe.Pointer(&imgData[0]))
	bmp.fPixmap.fRowBytes = uintptr(width * 4)
	bmp.fPixmap.fInfo.fColorInfo.fColorType = 4
	bmp.fPixmap.fInfo.fColorInfo.fAlphaType = 1
	bmp.fPixmap.fInfo.fDimensions.fWidth = int32(width)
	bmp.fPixmap.fInfo.fDimensions.fHeight = int32(height)

	var outLen uint32
	ptr, _, _ := procPerformOCR.Call(
		uintptr(unsafe.Pointer(&bmp)),
		uintptr(unsafe.Pointer(&outLen)),
	)
	if ptr == 0 {
		return nil, fmt.Errorf("PerformOCR() returned NULL")
	}

	result := make([]byte, outLen)
	copy(result, (*[1 << 30]byte)(unsafe.Pointer(ptr))[:outLen:outLen])
	procFreeLibraryAllocatedCharArray.Call(ptr)
	return result, nil
}

func decodeToRGBA(buff []byte) ([]byte, int, int, error) {
	src, _, err := image.Decode(bytes.NewReader(buff))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("unable to decode image: %v", err)
	}

	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	const maxPx = 3_000_000
	if w*h > maxPx {
		aspect := float64(w) / float64(h)
		nw := int(math.Sqrt(float64(maxPx) * aspect))
		nh := int(float64(nw) / aspect)
		log.Printf("Resizing image: %dx%d → %dx%d", w, h, nw, nh)
		src = scaleImage(src, bounds, nw, nh)
		w, h = nw, nh
		bounds = src.Bounds()
	}

	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(rgba, rgba.Bounds(), src, bounds.Min, draw.Src)
	return rgba.Pix, w, h, nil
}

func scaleImage(src image.Image, bounds image.Rectangle, nw, nh int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	sw := bounds.Dx()
	sh := bounds.Dy()
	ox := bounds.Min.X
	oy := bounds.Min.Y
	switch s := src.(type) {
	case *image.RGBA:
		for y := 0; y < nh; y++ {
			srcY := y*sh/nh + oy
			dstRow := dst.Pix[y*dst.Stride:]
			for x := 0; x < nw; x++ {
				si := s.PixOffset(x*sw/nw+ox, srcY)
				di := x * 4
				dstRow[di] = s.Pix[si]
				dstRow[di+1] = s.Pix[si+1]
				dstRow[di+2] = s.Pix[si+2]
				dstRow[di+3] = s.Pix[si+3]
			}
		}
	case *image.NRGBA:
		for y := 0; y < nh; y++ {
			srcY := y*sh/nh + oy
			dstRow := dst.Pix[y*dst.Stride:]
			for x := 0; x < nw; x++ {
				si := s.PixOffset(x*sw/nw+ox, srcY)
				di := x * 4
				dstRow[di] = s.Pix[si]
				dstRow[di+1] = s.Pix[si+1]
				dstRow[di+2] = s.Pix[si+2]
				dstRow[di+3] = s.Pix[si+3]
			}
		}
	default:
		for y := 0; y < nh; y++ {
			for x := 0; x < nw; x++ {
				dst.Set(x, y, src.At(x*sw/nw+ox, y*sh/nh+oy))
			}
		}
	}
	return dst
}

type ocrWorkerMgr struct {
	mu      sync.Mutex
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  io.Reader
	libPath string
	mDir    string
}

func (w *ocrWorkerMgr) start() error {
	exe, _ := os.Executable()
	cmd := exec.Command(exe, "--ocr-worker", w.libPath, w.mDir)
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start worker: %w", err)
	}

	w.cmd = cmd
	w.stdin = stdin
	w.stdout = stdout

	ready := make(chan error, 1)
	go func() {
		b := make([]byte, 1)
		if _, err := io.ReadFull(w.stdout, b); err != nil {
			ready <- fmt.Errorf("worker ready signal: %w", err)
		} else if b[0] != 0x01 {
			ready <- fmt.Errorf("unexpected ready byte 0x%02x", b[0])
		} else {
			ready <- nil
		}
	}()
	select {
	case err := <-ready:
		return err
	case <-time.After(5 * time.Minute):
		_ = cmd.Process.Kill()
		return fmt.Errorf("worker startup timeout (5 min)")
	}
}

func (w *ocrWorkerMgr) stop() {
	if w.cmd != nil {
		_ = w.cmd.Process.Kill()
		_ = w.cmd.Wait()
		w.cmd = nil
		w.stdin = nil
		w.stdout = nil
	}
}

func (w *ocrWorkerMgr) send(imgData []byte, width, height int) ([]byte, error) {
	var hdr [8]byte
	binary.LittleEndian.PutUint32(hdr[0:], uint32(width))
	binary.LittleEndian.PutUint32(hdr[4:], uint32(height))
	bw := bufio.NewWriterSize(w.stdin, 64*1024)
	if _, err := bw.Write(hdr[:]); err != nil {
		return nil, fmt.Errorf("write header: %w", err)
	}
	if _, err := bw.Write(imgData); err != nil {
		return nil, fmt.Errorf("write pixels: %w", err)
	}
	if err := bw.Flush(); err != nil {
		return nil, fmt.Errorf("flush request: %w", err)
	}

	var respLen uint32
	if err := binary.Read(w.stdout, binary.LittleEndian, &respLen); err != nil {
		return nil, fmt.Errorf("read response length: %w", err)
	}
	if respLen == 0 {
		return nil, nil
	}
	result := make([]byte, respLen)
	if _, err := io.ReadFull(w.stdout, result); err != nil {
		return nil, fmt.Errorf("read response data: %w", err)
	}
	return result, nil
}

func (w *ocrWorkerMgr) OCR(imgData []byte, width, height int) ([]byte, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	for attempt := 1; attempt <= 3; attempt++ {
		if w.cmd == nil {
			log.Printf("Starting OCR worker (attempt %d/3)...", attempt)
			if err := w.start(); err != nil {
				log.Printf("Worker start failed: %v", err)
				time.Sleep(time.Second)
				continue
			}
			log.Println("OCR worker ready")
		}

		result, err := w.send(imgData, width, height)
		if err == nil {
			return result, nil
		}
		log.Printf("OCR worker crashed (attempt %d/3): %v – restarting...", attempt, err)
		w.stop()
		time.Sleep(500 * time.Millisecond)
	}
	return nil, fmt.Errorf("OCR failed after 3 restart attempts")
}

type ocrResponse struct {
	FullText string `json:"fullText"`
	Lines    []Line `json:"lines"`
}

func ocrHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Request: %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

	if r.Method != http.MethodPost {
		http.Error(w, "Only POST is allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 100<<20)

	var buff []byte
	if file, _, err := r.FormFile("image"); err == nil {
		defer file.Close()
		if buff, err = io.ReadAll(file); err != nil {
			http.Error(w, "Failed to read uploaded image", http.StatusBadRequest)
			return
		}
	} else {
		var payload struct {
			Type  string `json:"type"`
			Image string `json:"image"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, "Invalid JSON payload", http.StatusBadRequest)
			return
		}
		if payload.Type != "base64" || payload.Image == "" {
			http.Error(w, `JSON body must be: {"type":"base64","image":"<base64>"}`, http.StatusBadRequest)
			return
		}
		if buff, err = base64.StdEncoding.DecodeString(payload.Image); err != nil {
			http.Error(w, "Invalid base64 image data", http.StatusBadRequest)
			return
		}
	}

	imgData, wid, hgt, err := decodeToRGBA(buff)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ocrStart := time.Now()
	protoBytes, err := workerMgr.OCR(imgData, wid, hgt)
	if err != nil {
		http.Error(w, "OCR processing failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	ocrDur := time.Since(ocrStart)

	lines := parseVisualAnnotation(protoBytes)
	var textParts []string
	for _, l := range lines {
		if strings.TrimSpace(l.Text) != "" {
			textParts = append(textParts, l.Text)
		}
	}

	resp := ocrResponse{
		FullText: strings.Join(textParts, "\n"),
		Lines:    lines,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
	log.Printf("OCR done: %d lines detected (%.3fs)", len(lines), ocrDur.Seconds())
}

func runOCRWorker(libPath, mDir string) {
	if err := initScreenAI(libPath, mDir); err != nil {
		log.Fatalf("OCR worker: init failed: %v", err)
	}

	if _, err := os.Stdout.Write([]byte{0x01}); err != nil {
		log.Fatalf("OCR worker: failed to write ready signal: %v", err)
	}

	for {
		var hdr [8]byte
		if _, err := io.ReadFull(os.Stdin, hdr[:]); err != nil {
			return
		}
		width := binary.LittleEndian.Uint32(hdr[0:])
		height := binary.LittleEndian.Uint32(hdr[4:])

		const maxPixels = 50_000_000
		if width == 0 || height == 0 || uint64(width)*uint64(height) > maxPixels {
			log.Printf("OCR worker: invalid dimensions %dx%d, skipping", width, height)
			_ = binary.Write(os.Stdout, binary.LittleEndian, uint32(0))
			continue
		}
		pixels := make([]byte, uint64(width)*uint64(height)*4)
		if _, err := io.ReadFull(os.Stdin, pixels); err != nil {
			return
		}

		result, err := runOCR(pixels, int(width), int(height))
		if err != nil {
			log.Printf("OCR worker: runOCR error: %v", err)
			result = nil
		}

		var respLen uint32
		if result != nil {
			respLen = uint32(len(result))
		}
		if err := binary.Write(os.Stdout, binary.LittleEndian, respLen); err != nil {
			return
		}
		if respLen > 0 {
			if _, err := os.Stdout.Write(result); err != nil {
				return
			}
		}
	}
}

func main() {

	if len(os.Args) >= 4 && os.Args[1] == "--ocr-worker" {
		runOCRWorker(os.Args[2], os.Args[3])
		return
	}

	if len(os.Args) >= 2 {
		log.SetOutput(io.Discard)

		exePath, _ := os.Executable()
		exeDir := filepath.Dir(exePath)

		mDir, libPath := resolveScreenAI(exeDir)
		if mDir == "" {
			fmt.Fprintf(os.Stderr, "Error: screen_ai library not found in %s\n", exeDir)
			os.Exit(1)
		}
		if err := initScreenAI(libPath, mDir); err != nil {
			fmt.Fprintf(os.Stderr, "Error: failed to initialize Screen AI: %v\n", err)
			os.Exit(1)
		}

		imagePath := os.Args[1]
		if _, err := os.Stat(imagePath); os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "Error: image file not found: %s\n", imagePath)
			os.Exit(1)
		}
		buff, err := os.ReadFile(imagePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: cannot read image file: %v\n", err)
			os.Exit(1)
		}
		imgData, wid, hgt, err := decodeToRGBA(buff)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		protoBytes, err := runOCR(imgData, wid, hgt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: OCR failed: %v\n", err)
			os.Exit(1)
		}
		for _, l := range parseVisualAnnotation(protoBytes) {
			if strings.TrimSpace(l.Text) != "" {
				fmt.Println(l.Text)
			}
		}
		return
	}

	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)

	mDir, libPath := resolveScreenAI(exeDir)
	if mDir == "" {
		log.Fatalf(
			"Screen AI library not found.\n"+
				"Place the 'screen_ai' folder (containing chrome_screen_ai.dll and model files)\n"+
				"in the same directory as this executable: %s", exeDir,
		)
	}

	workerMgr = &ocrWorkerMgr{libPath: libPath, mDir: mDir}
	log.Println("Starting OCR worker subprocess...")
	if err := workerMgr.start(); err != nil {
		log.Fatalf("Failed to start OCR worker: %v", err)
	}
	log.Println("OCR worker ready")

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt)
	go func() {
		<-quit
		workerMgr.stop()
		os.Exit(0)
	}()

	const addr = ":62815"
	log.Printf("ScreenAI OCR %s - starting server on port %s", version, addr)
	log.Printf("(C) 2026 Fazx - GarudaMods")
	http.HandleFunc("/ocr", ocrHandler)
	log.Fatal(http.ListenAndServe(addr, nil))
}
