//date: 2023-04-14T17:02:32Z
//url: https://api.github.com/gists/0212b141ceba629cad9a19b7619de889
//owner: https://api.github.com/users/RaphGL

package main

import "fmt"

type EventHandler struct {
	// all events available on the system
	validEvents []string
	// a were an event string contains all of it's associated functions to be run
	subscribers map[string][]func(s string)
	// a queue of all raised events to be processed
	eventQueue [][2]string
}

func NewEventHandler() *EventHandler {
	return &EventHandler{
		subscribers: make(map[string][]func(s string)),
	}
}

func (ev *EventHandler) AddEvents(events ...string) {
	for _, event := range events {
		hasEvent := false
		for _, e := range ev.validEvents {
			if e == event {
				hasEvent = true
				break
			}
		}

		if !hasEvent {
			ev.validEvents = append(ev.validEvents, event)
		}
	}
}

func (ev *EventHandler) RaiseEvent(event string, s string) {
	ev.eventQueue = append(ev.eventQueue, [2]string{event, s})
}

func (ev *EventHandler) Subscribe(event string, handler func(s string)) {
	ev.subscribers[event] = append(ev.subscribers[event], handler)
}

func (ev *EventHandler) Unsubscribe(event string, handler func(s string)) {
}

func (ev *EventHandler) HandleEvents() {
	for _, e := range ev.eventQueue {
		for _, handler := range ev.subscribers[e[0]] {
			handler(e[1])
		}
	}

	ev.eventQueue = [][2]string{}
}

func main() {
	ev := NewEventHandler()
	ev.AddEvents("click", "typing", "sending", "inactive")

	ev.Subscribe("click", func(s string) {
		fmt.Println(s, "clicked somewhere")
	})

	ev.Subscribe("typing", func(s string) {
		fmt.Println(s, "the user is typing...")
	})

	ev.Subscribe("sending", func(s string) {
		fmt.Println(s, "the user is sending a message...")
	})

	ev.Subscribe("inactive", func(s string) {
		fmt.Println("the user is inactive!")
	})

	fmt.Println("the following events are available: click, typing, sending, inactive")
	for {
		var eventName string
		fmt.Scanln(&eventName)
		ev.RaiseEvent(eventName, "1")
		ev.RaiseEvent(eventName, "2")
		ev.RaiseEvent(eventName, "3")
		ev.HandleEvents()
	}
}