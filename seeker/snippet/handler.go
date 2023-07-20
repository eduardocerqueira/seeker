//date: 2023-07-20T17:02:14Z
//url: https://api.github.com/gists/a920dcfea6cca057e517443c8a9423c0
//owner: https://api.github.com/users/delveper

package subs

import (
	"context"
	"errors"
	"net/http"
	"net/mail"

	"github.com/GenesisEducationKyiv/main-project-delveper/sys/web"
)

const StatusSubscribed = "subscribed"

//go:generate moq -out=../../test/mock/subscriber.go -pkg=mock . SubscriptionService

// SubscriptionService is an interface for subscription service.
type SubscriptionService interface {
	Subscribe(context.Context, Subscription) error
}

// Handler handles subscription.
type Handler struct {
	SubscriptionService
}

// Request is a request for subscription.
type Request struct {
	Email         string `json:"email"`
	BaseCurrency  string `json:"base_currency"`
	QuoteCurrency string `json:"quote_currency"`
}

// Response is a response for subscription.
type Response struct {
	Message string `json:"message"`
}

// NewHandler returns a new Handler instance.
func NewHandler(ss SubscriptionService) *Handler {
	return &Handler{SubscriptionService: ss}
}

func NewResponse(msg string) *Response {
	return &Response{Message: msg}
}

func toSubscription(subsReq *Request) (Subscription, error) {
	email, err := mail.ParseAddress(subsReq.Email)
	if err != nil {
		return Subscription{}, errors.Join(err, ErrMissingEmail)
	}

	subs := Subscription{
		Subscriber: NewSubscriber(email),
		Topic:      NewTopic(subsReq.BaseCurrency, subsReq.QuoteCurrency),
	}

	return subs, nil
}

// Subscribe subscribes to e-mails.
func (h *Handler) Subscribe(ctx context.Context, rw http.ResponseWriter, req *http.Request) error {
	ctx, cancel := context.WithTimeout(ctx, defaultTimeout)
	defer cancel()

	var newReq *Request
	if err := web.DecodeBody(req.Body, newReq); err != nil {
		return err
	}

	subs, err := toSubscription(newReq)
	if err != nil {
		return web.NewRequestError(err, http.StatusBadRequest)
	}

	if err := h.SubscriptionService.Subscribe(ctx, subs); err != nil {
		switch {
		case errors.Is(err, ErrSubscriptionExists):
			return web.NewRequestError(err, http.StatusConflict)
		case errors.Is(err, context.DeadlineExceeded):
			return web.NewRequestError(err, http.StatusRequestTimeout)
		}

		return err
	}

	return web.Respond(ctx, rw, NewResponse(StatusSubscribed), http.StatusCreated)
}