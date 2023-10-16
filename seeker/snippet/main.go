//date: 2023-10-16T17:06:37Z
//url: https://api.github.com/gists/b3ca951fb68184a5beb996a5ee45a469
//owner: https://api.github.com/users/godcodehunter

package main

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ЗАДАНИЕ:
// * сделать из плохого кода хороший;
// * важно сохранить логику появления ошибочных тасков;
// * сделать правильную мультипоточность обработки заданий.

// приложение эмулирует получение и обработку тасков, пытается и получать и обрабатывать в многопоточном режиме
// В конце должно выводить успешные таски и ошибки выполнены остальных тасков

const ProductionTime = time.Duration(500) * time.Millisecond

type Report struct {
	id        uuid.UUID
	creation  time.Time
	execution time.Duration
	err       error
	result    []byte
}

func main() {
	taskCreator := func(ctx context.Context, input chan Report) {
		for {
			creation := time.Now()
			var err error

			// Synthetic error condition
			if time.Now().Nanosecond()%2 > 0 {
				err = errors.New("Some error occurred")
			}

			id, err2 := uuid.NewUUID()
			if err2 != nil {
				err = err2
			}

			select {
			case <-ctx.Done():
				close(input)
				fmt.Printf("Task creation is done\n")
				return
			case input <- Report{creation: creation, id: id, err: err}:
				fmt.Printf("Report with id %s crated\n", id)
			}
		}
	}

	taskProcessor := func(input, processed chan Report, wg *sync.WaitGroup) {
		defer wg.Done()

		deadline := time.Now().Add(20 * time.Millisecond)
		for item := range input {
			// Synthetic error condition
			if item.creation.After(deadline) {
				item.result = []byte("Task has been successes")
			} else {
				item.err = errors.New("Something went wrong")
			}
			item.execution = time.Now().Sub(item.creation)

			time.Sleep(time.Millisecond * 150)

			fmt.Printf("Report with id %s processed\n", item.id)
			processed <- item
		}
	}

	taskSorter := func(processed, done, undone chan Report, wg *sync.WaitGroup) {
		defer wg.Done()

		for item := range processed {

			if item.err != nil {
				fmt.Errorf("Report with id %s is sorted as errored\n", item.id)
				undone <- item
			} else {
				fmt.Printf("Report with id %s is sorted as done\n", item.id)
				done <- item
			}
		}
	}

	input := make(chan Report, 10)
	processed := make(chan Report, 10)
	done := make(chan Report, 1000)
	undone := make(chan Report, 1000)

	ctx, cncl := context.WithTimeout(context.Background(), ProductionTime)
	defer cncl()

	go taskCreator(ctx, input)

	wgProc := new(sync.WaitGroup)
	wgSort := new(sync.WaitGroup)

	for i := 0; i < 10; i++ {
		wgProc.Add(1)
		go taskProcessor(input, processed, wgProc)
	}

	wgSort.Add(1)
	go taskSorter(processed, done, undone, wgSort)

	wgProc.Wait()
	close(processed)
	wgSort.Wait()
	close(done)
	close(undone)

	println("Errors:")
	for item := range undone {
		fmt.Printf("Task id %d duration %s, error %s\n", item.id.ID(), item.execution, item.err)
	}

	println("Done tasks:")
	for item := range done {
		fmt.Printf("Task id %d duration %s, result %s\n", item.id.ID(), item.execution, item.result)
	}
}
