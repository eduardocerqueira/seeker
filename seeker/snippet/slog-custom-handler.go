//date: 2025-04-23T16:44:52Z
//url: https://api.github.com/gists/960a06334be81d05d101a23649566179
//owner: https://api.github.com/users/ilekehitys

Go's log/slog is great, but it's key=value format may be hard to read and it does not respect newlines. Below is a working
code that tries to find a solution for the formatting issue. If someone knows simpler way to achieve the same,
please drop a comment. 

You can easily add source file name and line number in main(). Also logging to a file instead of os.Stdout is easy.

```go
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
)

type ValueOnlyHandler struct {
	handler   slog.Handler
	addSource bool
	level     slog.Leveler
        file      *os.File
}

func NewValueOnlyHandler(file *os.File, options *slog.HandlerOptions) *ValueOnlyHandler {
	return &ValueOnlyHandler{
		handler:   slog.NewJSONHandler(nil, options),
		addSource: options.AddSource,
		level:     options.Level,
                file: file,
	}
}

func (h *ValueOnlyHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return level >= h.level.Level()
}

func (h *ValueOnlyHandler) Handle(ctx context.Context, r slog.Record) error {
	timeStr := r.Time.Format("[2006/01/02 15:04:05]")
	output := fmt.Sprintf("%s %v %s\n", timeStr, r.Level, r.Message)
	if h.addSource {
		frames := runtime.CallersFrames([]uintptr{r.PC})
		frame, _ := frames.Next()
		if frame.File != "" {
			output = fmt.Sprintf("%s %s %s:%d %s\n", timeStr, r.Level, filepath.Base(frame.File), frame.Line, r.Message)
		}
	}
	_, err := h.file.WriteString(output)
	return err
}

func (h *ValueOnlyHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &ValueOnlyHandler{handler: h.handler.WithAttrs(attrs)}
}

func (h *ValueOnlyHandler) WithGroup(name string) slog.Handler {
	return &ValueOnlyHandler{handler: h.handler.WithGroup(name)}
}

func main() {
        file, err := os.OpenFile("myapp.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		slog.Error("failed to open log file", "err", err)
		os.Exit(1) 
	}
	defer file.Close()
	level := new(slog.LevelVar)
	level.Set(slog.LevelDebug)
	logger := slog.New(NewValueOnlyHandler(os.Stdout, &slog.HandlerOptions{AddSource: true, Level: level}))

	logger.Info("This is an info message", "user", "Alice")
	logger.Warn("Something to be careful about", "file", "config.yaml")
	logger.Error("An error occurred", "code", 500)
}

```
output:
[2025/04/23 19:32:31] INFO log.go:65 This is an info message
[2025/04/23 19:32:31] WARN log.go:66 Something to be careful about
[2025/04/23 19:32:31] ERROR log.go:67 An error occurred
