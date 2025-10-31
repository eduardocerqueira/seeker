//date: 2025-10-31T17:06:17Z
//url: https://api.github.com/gists/dd817e49a6388c0a543e9f3b14b54cac
//owner: https://api.github.com/users/dcarley

//go:build linux

package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"time"

	"github.com/containerd/containerd/v2/pkg/sys/reaper"
	"golang.org/x/sys/unix"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "producer" {
		// Helper mode: produce some data
		for i := 0; i < 1000; i++ {
			fmt.Printf("line %d\n", i)
		}
		return
	}

	if len(os.Args) > 1 && os.Args[1] == "consumer" {
		// Helper mode: consume data and count lines
		count := 0
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			count++
		}
		if err := scanner.Err(); err != nil {
			fmt.Fprintf(os.Stderr, "DEBUG consumer: scan error after %d lines: %v\n", count, err)
		}
		if count != 1000 {
			fmt.Fprintf(os.Stderr, "ERROR: Expected 1000 lines, got %d\n", count)
			os.Exit(1)
		}
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Main test mode: progressively set up reaper components
	// IMPORTANT: We follow the same order as the real containerd shim:
	//   1. setupSignals() - includes SIGCHLD
	//   2. platformInit() - calls SetSubreaper()
	//   3. start reaper goroutine

	// Test 0: Baseline before any reaper setup
	fmt.Println("Test 0: Baseline (no reaper components)")
	fmt.Println("===============================================")
	fmt.Println("Establishes baseline: normal exec.Command behavior")
	fmt.Println()
	runTestSuite("WITHOUT Setpgid", 20, func() error { return runPipeline(false) })
	runTestSuite("WITH Setpgid", 20, func() error { return runPipeline(true) })
	fmt.Println()

	// Step 1: Set up SIGCHLD handler FIRST (like the real shim)
	// https://github.com/containerd/containerd/blob/v2.1.4/pkg/shim/shim.go#L101-L102
	// https://github.com/containerd/containerd/blob/v2.1.4/pkg/shim/shim_unix.go#L41-L43
	// https://github.com/containerd/containerd/blob/v2.1.4/pkg/shim/shim.go#L481
	// https://github.com/containerd/containerd/blob/v2.1.4/pkg/shim/shim_unix.go#L73-L92
	fmt.Println("Step 1: Setting up SIGCHLD handler...")
	signals := make(chan os.Signal, 32)
	signal.Notify(signals, unix.SIGCHLD)
	fmt.Println("✓ Registered SIGCHLD handler")
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case s := <-signals:
				if s == unix.SIGCHLD {
					if err := reaper.Reap(); err != nil {
						fmt.Fprintf(os.Stderr, "reaper.Reap() error: %v\n", err)
					}
				}
			}
		}
	}()
	fmt.Println("✓ Started reaper goroutine")
	fmt.Println()

	// Test 1: After SIGCHLD handler, before subreaper
	fmt.Println("Test 1: After SIGCHLD handler (no subreaper)")
	fmt.Println("===============================================")
	fmt.Println("Tests if SIGCHLD handler alone causes ECHILD races")
	fmt.Println("Handler only reaps direct children (not reparented descendants)")
	fmt.Println()
	runTestSuite("WITHOUT Setpgid", 20, func() error { return runPipeline(false) })
	runTestSuite("WITH Setpgid", 20, func() error { return runPipeline(true) })
	fmt.Println()

	// Step 2: Set subreaper SECOND (like the real shim)
	// https://github.com/containerd/containerd/blob/v2.1.4/pkg/shim/shim.go#L99-L100
	// https://github.com/containerd/containerd/blob/v2.1.4/pkg/shim/shim.go#L243-L247
	// https://github.com/containerd/containerd/blob/v2.1.4/pkg/shim/shim_linux.go#L29-L31
	fmt.Println("Step 2: Setting up subreaper...")
	if err := reaper.SetSubreaper(1); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to set subreaper: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("✓ Set PR_SET_CHILD_SUBREAPER")
	fmt.Println()

	// Test 2: Full reaper setup
	fmt.Println("Test 2: Full reaper (SIGCHLD + subreaper)")
	fmt.Println("===============================================")
	fmt.Println("This matches actual containerd shim configuration")
	fmt.Println("Handler reaps ALL children AND reparented descendants")
	fmt.Println()
	runTestSuite("WITHOUT Setpgid", 20, func() error { return runPipeline(false) })
	runTestSuite("WITH Setpgid", 20, func() error { return runPipeline(true) })
	runTestSuite("Using reaper.Default API", 20, runPipelineWithReaper)
}

// runTestSuite runs a test function multiple times and reports results
func runTestSuite(name string, iterations int, testFn func() error) {
	fmt.Println("---------------------------------------")
	fmt.Printf("%s\n", name)
	fmt.Println("---------------------------------------")
	successCount := 0
	failCount := 0

	for i := 0; i < iterations; i++ {
		if err := testFn(); err != nil {
			fmt.Printf("  Run %2d: ❌ FAILED - %v\n", i+1, err)
			failCount++
		} else {
			fmt.Printf("  Run %2d: ✓ succeeded\n", i+1)
			successCount++
		}
		time.Sleep(10 * time.Millisecond)
	}

	fmt.Printf("\n  RESULT: %d succeeded, %d failed", successCount, failCount)
	fmt.Println()

	if failCount > 0 {
		fmt.Printf("  ❌ Failures detected")
	} else {
		fmt.Printf("  ✓ All tests passed\n")
	}
	fmt.Println()
	fmt.Println()
}

func runPipeline(useSetpgid bool) error {
	// Producer: generate data
	producer := exec.Command(os.Args[0], "producer")
	if useSetpgid {
		producer.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	}

	// Consumer: read and count
	consumer := exec.Command(os.Args[0], "consumer")
	if useSetpgid {
		consumer.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	}

	// Connect pipeline
	pipe, err := producer.StdoutPipe()
	if err != nil {
		return fmt.Errorf("create pipe: %w", err)
	}
	consumer.Stdin = pipe
	// Capture stderr to see debug output
	consumerStderr, err := consumer.StderrPipe()
	if err != nil {
		return fmt.Errorf("consumer stderr pipe: %w", err)
	}

	// Start producer
	if err := producer.Start(); err != nil {
		return fmt.Errorf("start producer: %w", err)
	}

	// Start consumer
	if err := consumer.Start(); err != nil {
		producer.Wait() // Clean up
		return fmt.Errorf("start consumer: %w", err)
	}

	// Read stderr from consumer
	stderrOutput := make([]byte, 4096)
	n, _ := consumerStderr.Read(stderrOutput)

	// Wait for consumer
	consumerErr := consumer.Wait()

	// Wait for producer
	producerErr := producer.Wait()

	// Check errors
	if consumerErr != nil {
		if n > 0 {
			return fmt.Errorf("consumer failed: %w [stderr: %s]", consumerErr, string(stderrOutput[:n]))
		}
		return fmt.Errorf("consumer failed: %w", consumerErr)
	}
	if producerErr != nil {
		return fmt.Errorf("producer failed: %w", producerErr)
	}

	return nil
}

func runPipelineWithReaper() error {
	// Producer: generate data
	producer := exec.Command(os.Args[0], "producer")

	// Consumer: read and count
	consumer := exec.Command(os.Args[0], "consumer")

	// Connect pipeline
	pipe, err := producer.StdoutPipe()
	if err != nil {
		return fmt.Errorf("create pipe: %w", err)
	}
	consumer.Stdin = pipe
	// Capture stderr to see debug output
	consumerStderr, err := consumer.StderrPipe()
	if err != nil {
		return fmt.Errorf("consumer stderr pipe: %w", err)
	}

	// Start producer using reaper API
	producerEC, err := reaper.Default.Start(producer)
	if err != nil {
		return fmt.Errorf("start producer: %w", err)
	}

	// Start consumer using reaper API
	consumerEC, err := reaper.Default.Start(consumer)
	if err != nil {
		reaper.Default.Wait(producer, producerEC) // Clean up
		return fmt.Errorf("start consumer: %w", err)
	}

	// Read stderr from consumer
	stderrOutput := make([]byte, 4096)
	n, _ := consumerStderr.Read(stderrOutput)

	// Wait for consumer
	status, err := reaper.Default.Wait(consumer, consumerEC)
	if err != nil {
		reaper.Default.Wait(producer, producerEC) // Clean up
		return fmt.Errorf("wait consumer: %w", err)
	}
	if status != 0 {
		reaper.Default.Wait(producer, producerEC) // Clean up
		if n > 0 {
			return fmt.Errorf("consumer exit status: %d [stderr: %s]", status, string(stderrOutput[:n]))
		}
		return fmt.Errorf("consumer exit status: %d (didn't consume all lines)", status)
	}

	// Wait for producer
	status, err = reaper.Default.Wait(producer, producerEC)
	if err != nil {
		return fmt.Errorf("wait producer: %w", err)
	}
	if status != 0 {
		return fmt.Errorf("producer exit status: %d", status)
	}

	return nil
}