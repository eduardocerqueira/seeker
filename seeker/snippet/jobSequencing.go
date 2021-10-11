//date: 2021-10-11T17:01:45Z
//url: https://api.github.com/gists/93d98c28d9cd3314d05c4ec157906ca0
//owner: https://api.github.com/users/ghaini

package main

import "fmt"

type Job struct {
	Name     string
	Profits  int
	DeadLine int // hour
}

func main() {
	jobs := []Job{
		{
			Name:     "J1",
			Profits:  20,
			DeadLine: 2,
		},
		{
			Name:     "J2",
			Profits:  15,
			DeadLine: 2,
		},
		{
			Name:     "J3",
			Profits:  10,
			DeadLine: 1,
		},
		{
			Name:     "J4",
			Profits:  5,
			DeadLine: 3,
		},
		{
			Name:     "J5",
			Profits:  1,
			DeadLine: 3,
		},
	}
	sequentialJobs, profits := JobSequencing(jobs, 3)
	fmt.Println(profits)
	for _, sequentialJob := range sequentialJobs{
		fmt.Println(j.Name)
	}
}

func JobSequencing(jobs []Job, MaxDuration int) ([]Job,  int) {
	sequentialJob := make([]Job, MaxDuration)
	profits := 0
	for _, job := range jobs {
		for i := job.DeadLine - 1; i >= 0; i-- {
			if duration[i].Name == "" {
				profits += job.Profits
				sequentialJob[i] = job
				break
			}
		}
	}

	return sequentialJob, profits
}