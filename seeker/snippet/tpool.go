//date: 2026-02-03T17:37:43Z
//url: https://api.github.com/gists/b8961d48200d9d8c3e4be7aee45ae3e2
//owner: https://api.github.com/users/ashutoshdhande

package main

import (
	"fmt"
	"sync"
	"time"
)

func timeTakingTask() {
	time.Sleep(3 * time.Second)
}

type Task func()

type Pool struct {
	numOfThreads uint8
	wg           sync.WaitGroup
	taskCh       chan Task
	mtx          sync.Mutex
}

func (pool *Pool) createPool(n uint8) {
	pool.numOfThreads = n
	pool.taskCh = make(chan Task, 20)
	pool.wg.Add(int(pool.numOfThreads))

	for i := range n {
		go pool.workerRoutine(i)
	}

}

func (pool *Pool) workerRoutine(id uint8) {
	defer pool.wg.Done()

	for task := range pool.taskCh {
		task()
		pool.mtx.Lock()
		fmt.Printf("thread %v executed task\n", id)
		pool.mtx.Unlock()
	}

}

func (pool *Pool) enqueueTask(task Task) {
	pool.taskCh <- task
}

func (pool *Pool) shutdown() {
	close(pool.taskCh)
	pool.wg.Wait()
}

func main() {
	fmt.Println("Tpool program started")

	var tpool Pool

	tpool.createPool(4)

	for range 200 {
		tpool.enqueueTask(timeTakingTask)
	}

	tpool.shutdown()

}
