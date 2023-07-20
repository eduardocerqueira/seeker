//date: 2023-07-20T17:02:14Z
//url: https://api.github.com/gists/a920dcfea6cca057e517443c8a9423c0
//owner: https://api.github.com/users/delveper

package subs

import (
	"net/mail"
)

// Subscription represents aggregate subscription.
type Subscription struct {
	Subscriber Subscriber
	Topic      Topic
}

// Subscriber represents an entity that subscribes to emails.
type Subscriber struct {
	//TODO: extend entity.
	Address *mail.Address
}

// Topic represents a value object of a topic for subscription.
type Topic = CurrencyPair

// CurrencyPair represents a value object of a currency pair for subscription.
type CurrencyPair struct {
	Base  string
	Quote string
}

func NewSubscriber(address *mail.Address) Subscriber {
	return Subscriber{Address: address}
}

func NewTopic(base, quote string) Topic {
	return Topic{
		Base:  base,
		Quote: quote,
	}
}