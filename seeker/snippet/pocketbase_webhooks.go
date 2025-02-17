//date: 2025-02-17T16:58:25Z
//url: https://api.github.com/gists/3aea59d11bb0c898d80fd9c028ed7d0c
//owner: https://api.github.com/users/mdarif551578

/*
The MIT License (MIT) Copyright (c) 2024 - present, Jonas Plum

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/labstack/echo/v5"
	"github.com/pocketbase/dbx"
	"github.com/pocketbase/pocketbase"
	"github.com/pocketbase/pocketbase/apis"
	"github.com/pocketbase/pocketbase/core"
	"github.com/pocketbase/pocketbase/daos"
	"github.com/pocketbase/pocketbase/migrations"
	"github.com/pocketbase/pocketbase/models"
	"github.com/pocketbase/pocketbase/models/schema"
)

const webhooksCollection = "webhooks"

type Webhook struct {
	ID          string `db:"id" json:"id"`
	Name        string `db:"name" json:"name"`
	Collection  string `db:"collection" json:"collection"`
	Destination string `db:"destination" json:"destination"`
}

func attachWebhooks(app *pocketbase.PocketBase) {
	migrations.Register(func(db dbx.Builder) error {
		return daos.New(db).SaveCollection(&models.Collection{
			Name:   webhooksCollection,
			Type:   models.CollectionTypeBase,
			System: true,
			Schema: schema.NewSchema(
				&schema.SchemaField{
					Name:     "name",
					Type:     schema.FieldTypeText,
					Required: true,
				},
				&schema.SchemaField{
					Name:     "collection",
					Type:     schema.FieldTypeText,
					Required: true,
				},
				&schema.SchemaField{
					Name:     "destination",
					Type:     schema.FieldTypeUrl,
					Required: true,
				},
			),
		})
	}, func(db dbx.Builder) error {
		dao := daos.New(db)

		id, err := dao.FindCollectionByNameOrId(webhooksCollection)
		if err != nil {
			return err
		}

		return dao.DeleteCollection(id)
	}, "1690000000_webhooks.go")

	app.OnRecordAfterCreateRequest().Add(func(e *core.RecordCreateEvent) error {
		return event(app, "create", e.Collection.Name, e.Record, e.HttpContext)
	})
	app.OnRecordAfterUpdateRequest().Add(func(e *core.RecordUpdateEvent) error {
		return event(app, "update", e.Collection.Name, e.Record, e.HttpContext)
	})
	app.OnRecordAfterDeleteRequest().Add(func(e *core.RecordDeleteEvent) error {
		return event(app, "delete", e.Collection.Name, e.Record, e.HttpContext)
	})
}

type Payload struct {
	Action     string         `json:"action"`
	Collection string         `json:"collection"`
	Record     *models.Record `json:"record"`
	Auth       *models.Record `json:"auth,omitempty"`
	Admin      *models.Admin  `json:"admin,omitempty"`
}

func event(app *pocketbase.PocketBase, action, collection string, record *models.Record, ctx echo.Context) error {
	auth, _ := ctx.Get(apis.ContextAuthRecordKey).(*models.Record)
	admin, _ := ctx.Get(apis.ContextAdminKey).(*models.Admin)

	var webhooks []Webhook
	if err := app.Dao().DB().
		Select().
		From(webhooksCollection).
		Where(dbx.HashExp{"collection": collection}).
		All(&webhooks); err != nil {
		return err
	}

	if len(webhooks) == 0 {
		return nil
	}

	payload, err := json.Marshal(&Payload{
		Action:     action,
		Collection: collection,
		Record:     record,
		Auth:       auth,
		Admin:      admin,
	})
	if err != nil {
		return err
	}

	for _, webhook := range webhooks {
		if err := sendWebhook(ctx.Request().Context(), webhook, payload); err != nil {
			app.Logger().Error("failed to send webhook", "action", action, "name", webhook.Name, "collection", webhook.Collection, "destination", webhook.Destination, "error", err.Error())
		} else {
			app.Logger().Info("webhook sent", "action", action, "name", webhook.Name, "collection", webhook.Collection, "destination", webhook.Destination)
		}
	}

	return nil
}

func sendWebhook(ctx context.Context, webhook Webhook, payload []byte) error {
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, webhook.Destination, bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)

		return fmt.Errorf("failed to send webhook: %s", string(b))
	}

	return nil
}
