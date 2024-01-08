//date: 2024-01-08T17:07:39Z
//url: https://api.github.com/gists/625afb750add58df46a2fa336a7b2c77
//owner: https://api.github.com/users/weilsonwonder

// Tutorial: https://weilson.medium.com/go-build-host-telegram-bot-for-0-cost-telegram-google-compute-engine-7c7f155a8781

package main

import (
	"fmt"
	"log"

	"github.com/NicoNex/echotron/v3"
)

// define our bot token here
const botApiToken = "6311188803: "**********"

func main() {
	// setup our contact point
	api : "**********"

	// set a simple basic command
	_, err := api.SetMyCommands(nil, echotron.BotCommand{
		Command:     "/hello",
		Description: "replies 'hi <your_telegram_username>, your telegram id is <your_telegram_id>'",
	})
	if err != nil {
		panic(err)
	}

	// we simply poll for messages
	fmt.Println("bot started")
	for update : "**********"
		// upon receiving update, we check the text for our command
		if update.Message.Text == "/hello" {
			// we try to find out about the sender
			from := update.Message.From
			if from != nil {
				// retrieve the details
				id, username := from.ID, from.Username
				// construct message
				msg := fmt.Sprintf("hi %s, your telegram id is %d", username, id)
				// send a message
				_, err = api.SendMessage(msg, update.ChatID(), nil)
				if err != nil {
					log.Println("send message failed:", err)
				}
			}
		}
	}
}
 nil {
					log.Println("send message failed:", err)
				}
			}
		}
	}
}
