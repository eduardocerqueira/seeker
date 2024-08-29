//date: 2024-08-29T16:54:35Z
//url: https://api.github.com/gists/c69cdc91a4dbf731bee5aba1696e00da
//owner: https://api.github.com/users/semenovdev

package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"log/slog"
	"time"

	"awesomeProject/runner"
)

type Service1 struct {
	io.Closer
}

func (s *Service1) Run(ctx context.Context) error {
	fmt.Println("start executing of Service1")
	time.Sleep(time.Minute * 10)
	fmt.Printf("Service1 is running (%s)\n", ctx.Value("Привет"))
	return nil
}

func (s *Service1) Close() error {
	fmt.Println("close service1")
	return nil
}

type Service2 struct {
	io.Closer

	Cancel func()
}

func (s *Service2) Run(ctx context.Context) error {
	time.Sleep(time.Second * 2)
	fmt.Printf("Service2 is running (%s)\n", ctx.Value("Привет"))
	time.Sleep(time.Minute)
	s.Cancel()
	return nil
}

func (s *Service2) Close() error {
	fmt.Println("close service2")
	return nil
}

type Service3 struct {
	io.Closer
}

func (s *Service3) Run(ctx context.Context) error {
	time.Sleep(time.Second * 3)
	fmt.Printf("Service3 is running (%s)\n", ctx.Value("Привет"))
	select {}
	return nil
}

func (s *Service3) Close() error {
	fmt.Println("close service3")
	return nil
}

type Service4 struct {
	io.Closer
}

func (s *Service4) Run(ctx context.Context) error {
	fmt.Printf("Service4 with error (%s)\n", ctx.Value("Привет"))
	return fmt.Errorf("error in service 4")
}

func (s *Service4) Close() error {
	fmt.Println("close service4")
	return nil
}

func main() {
	ctx := context.Background()
	ctx = context.WithValue(ctx, "Привет", "Мир")
	ctx, cancel := context.WithCancel(ctx)

	service1 := &Service1{} // завершится по таймауту и не затронет систему
	service2 := &Service2{  // упадёт через минуту после старта и утянет за собой всё
		Cancel: cancel,
	}
	service3 := &Service3{} // работал бы бесконечно, если бы не Service2
	service4 := &Service4{} // завершится с ошибкой в кроне и запишет её в лог

	app := runner.New(
		runner.WithContext(ctx),
		runner.WithCronJobTimeout(time.Second),
		runner.WithErrorLogger(slog.Default()),
	)
	err := app.AddCronJob("* * * * *", service1)
	if err != nil {
		log.Fatal(err)
	}
	err = app.AddCronJob("* * * * *", service4)
	if err != nil {
		log.Fatal(err)
	}

	app.RegisterService(service2)
	app.RegisterService(service3)
	app.Run()
}
