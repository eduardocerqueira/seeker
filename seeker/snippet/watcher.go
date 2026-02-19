//date: 2026-02-19T17:34:26Z
//url: https://api.github.com/gists/560671c860a6b2134d3691a893772714
//owner: https://api.github.com/users/nfarina

package execute

import (
	"fmt"
	"path/filepath"
	"reflect"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/microsoft/typescript-go/internal/compiler"
	"github.com/microsoft/typescript-go/internal/core"
	"github.com/microsoft/typescript-go/internal/execute/incremental"
	"github.com/microsoft/typescript-go/internal/execute/tsc"
	"github.com/microsoft/typescript-go/internal/tsoptions"
)

type Watcher struct {
	sys                            tsc.System
	configFileName                 string
	config                         *tsoptions.ParsedCommandLine
	compilerOptionsFromCommandLine *core.CompilerOptions
	reportDiagnostic               tsc.DiagnosticReporter
	reportErrorSummary             tsc.DiagnosticsReporter
	testing                        tsc.CommandLineTesting

	host           compiler.CompilerHost
	program        *incremental.Program
	prevModified   map[string]time.Time
	configModified bool
}

var _ tsc.Watcher = (*Watcher)(nil)

func createWatcher(
	sys tsc.System,
	configParseResult *tsoptions.ParsedCommandLine,
	compilerOptionsFromCommandLine *core.CompilerOptions,
	reportDiagnostic tsc.DiagnosticReporter,
	reportErrorSummary tsc.DiagnosticsReporter,
	testing tsc.CommandLineTesting,
) *Watcher {
	w := &Watcher{
		sys:                            sys,
		config:                         configParseResult,
		compilerOptionsFromCommandLine: compilerOptionsFromCommandLine,
		reportDiagnostic:               reportDiagnostic,
		reportErrorSummary:             reportErrorSummary,
		testing:                        testing,
		// reportWatchStatus: createWatchStatusReporter(sys, configParseResult.CompilerOptions().Pretty),
	}
	if configParseResult.ConfigFile != nil {
		w.configFileName = configParseResult.ConfigFile.SourceFile.FileName()
	}
	return w
}

func (w *Watcher) start() {
	w.host = compiler.NewCompilerHost(w.sys.GetCurrentDirectory(), w.sys.FS(), w.sys.DefaultLibraryPath(), nil, getTraceFromSys(w.sys, w.config.Locale(), w.testing))
	w.program = incremental.ReadBuildInfoProgram(w.config, incremental.NewBuildInfoReader(w.host), w.host)

	if w.testing == nil {
		// Initial build
		w.DoCycle()

		// Set up filesystem watcher
		fsWatcher, err := fsnotify.NewWatcher()
		if err != nil {
			fmt.Fprintln(w.sys.Writer(), "filesystem watcher unavailable, falling back to polling:", err)
			w.pollLoop()
			return
		}
		defer fsWatcher.Close()

		w.updateWatchedDirs(fsWatcher)

		// Event loop with debouncing
		debounce := time.NewTimer(0)
		if !debounce.Stop() {
			<-debounce.C
		}

		for {
			select {
			case event, ok := <-fsWatcher.Events:
				if !ok {
					return
				}
				if event.Has(fsnotify.Write) || event.Has(fsnotify.Create) || event.Has(fsnotify.Remove) || event.Has(fsnotify.Rename) {
					if isRelevantFile(event.Name) {
						debounce.Reset(100 * time.Millisecond)
					}
				}
			case err, ok := <-fsWatcher.Errors:
				if !ok {
					return
				}
				fmt.Fprintln(w.sys.Writer(), "watch error:", err)
			case <-debounce.C:
				w.DoCycle()
				w.updateWatchedDirs(fsWatcher)
			}
		}
	} else {
		// Initial compilation in test mode
		w.DoCycle()
	}
}

// pollLoop is the fallback when filesystem watching is unavailable.
func (w *Watcher) pollLoop() {
	watchInterval := w.config.ParsedConfig.WatchOptions.WatchInterval()
	for {
		time.Sleep(watchInterval)
		w.DoCycle()
	}
}

// updateWatchedDirs synchronizes the fsnotify watch list with the current
// program's source file directories.
func (w *Watcher) updateWatchedDirs(fsWatcher *fsnotify.Watcher) {
	if w.program == nil {
		return
	}

	dirs := map[string]struct{}{}
	for _, sourceFile := range w.program.GetProgram().SourceFiles() {
		dirs[filepath.Dir(sourceFile.FileName())] = struct{}{}
	}
	// Watch the config file's directory too
	if w.configFileName != "" {
		dirs[filepath.Dir(w.configFileName)] = struct{}{}
	}

	// Remove stale watches
	existing := map[string]struct{}{}
	for _, p := range fsWatcher.WatchList() {
		existing[p] = struct{}{}
		if _, ok := dirs[p]; !ok {
			fsWatcher.Remove(p)
		}
	}
	// Add new watches
	for dir := range dirs {
		if _, ok := existing[dir]; !ok {
			fsWatcher.Add(dir)
		}
	}
}

// isRelevantFile returns true if the file has a TypeScript-relevant extension.
func isRelevantFile(name string) bool {
	switch filepath.Ext(name) {
	case ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs", ".json":
		return true
	}
	return false
}

func (w *Watcher) DoCycle() {
	// if this function is updated, make sure to update `RunWatchCycle` in export_test.go as needed

	if w.hasErrorsInTsConfig() {
		// these are unrecoverable errors--report them and do not build
		return
	}

	// Quick check: skip expensive program creation if no source files changed
	if !w.configModified && w.prevModified != nil && !w.hasAnyFileChanged() {
		return
	}

	// updateProgram()
	w.program = incremental.NewProgram(compiler.NewProgram(compiler.ProgramOptions{
		Config: w.config,
		Host:   w.host,
	}), w.program, nil, w.testing != nil)

	if w.hasBeenModified(w.program.GetProgram()) {
		fmt.Fprintln(w.sys.Writer(), "build starting at", w.sys.Now().Format("03:04:05 PM"))
		timeStart := w.sys.Now()
		w.compileAndEmit()
		fmt.Fprintf(w.sys.Writer(), "build finished in %.3fs\n", w.sys.Now().Sub(timeStart).Seconds())
	}
	if w.testing != nil {
		w.testing.OnProgram(w.program)
	}
}

// hasAnyFileChanged does a cheap check of previously-known source file mod
// times to avoid the expensive NewProgram call when nothing has changed.
func (w *Watcher) hasAnyFileChanged() bool {
	for fileName, prevTime := range w.prevModified {
		s := w.sys.FS().Stat(fileName)
		if s == nil {
			return true // file was deleted
		}
		if s.ModTime() != prevTime {
			return true
		}
	}
	return false
}

func (w *Watcher) compileAndEmit() {
	// !!! output/error reporting is currently the same as non-watch mode
	// diagnostics, emitResult, exitStatus :=
	tsc.EmitFilesAndReportErrors(tsc.EmitInput{
		Sys:                w.sys,
		ProgramLike:        w.program,
		Program:            w.program.GetProgram(),
		ReportDiagnostic:   w.reportDiagnostic,
		ReportErrorSummary: w.reportErrorSummary,
		Writer:             w.sys.Writer(),
		CompileTimes:       &tsc.CompileTimes{},
		Testing:            w.testing,
	})
}

func (w *Watcher) hasErrorsInTsConfig() bool {
	// only need to check and reparse tsconfig options/update host if we are watching a config file
	extendedConfigCache := &tsc.ExtendedConfigCache{}
	if w.configFileName != "" {
		// !!! need to check that this merges compileroptions correctly. This differs from non-watch, since we allow overriding of previous options
		configParseResult, errors := tsoptions.GetParsedCommandLineOfConfigFile(w.configFileName, w.compilerOptionsFromCommandLine, nil, w.sys, extendedConfigCache)
		if len(errors) > 0 {
			for _, e := range errors {
				w.reportDiagnostic(e)
			}
			return true
		}
		// CompilerOptions contain fields which should not be compared; clone to get a copy without those set.
		if !reflect.DeepEqual(w.config.CompilerOptions().Clone(), configParseResult.CompilerOptions().Clone()) {
			// fmt.Fprintln(w.sys.Writer(), "build triggered due to config change")
			w.configModified = true
		}
		w.config = configParseResult
	}
	w.host = compiler.NewCompilerHost(w.sys.GetCurrentDirectory(), w.sys.FS(), w.sys.DefaultLibraryPath(), extendedConfigCache, getTraceFromSys(w.sys, w.config.Locale(), w.testing))
	return false
}

func (w *Watcher) hasBeenModified(program *compiler.Program) bool {
	// checks watcher's snapshot against program file modified times
	currState := map[string]time.Time{}
	filesModified := w.configModified
	for _, sourceFile := range program.SourceFiles() {
		fileName := sourceFile.FileName()
		s := w.sys.FS().Stat(fileName)
		if s == nil {
			// do nothing; if file is in program.SourceFiles() but is not found when calling Stat, file has been very recently deleted.
			// deleted files are handled outside of this loop
			continue
		}
		currState[fileName] = s.ModTime()
		if !filesModified {
			if currState[fileName] != w.prevModified[fileName] {
				// fmt.Fprint(w.sys.Writer(), "build triggered from ", fileName, ": ", w.prevModified[fileName], " -> ", currState[fileName], "\n")
				filesModified = true
			}
			// catch cases where no files are modified, but some were deleted
			delete(w.prevModified, fileName)
		}
	}
	if !filesModified && len(w.prevModified) > 0 {
		// fmt.Fprintln(w.sys.Writer(), "build triggered due to deleted file")
		filesModified = true
	}
	w.prevModified = currState

	// reset state for next cycle
	w.configModified = false
	return filesModified
}
