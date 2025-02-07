//date: 2025-02-07T16:46:56Z
//url: https://api.github.com/gists/e8b1336b1dfc19fab154c99a817ba577
//owner: https://api.github.com/users/lucadboer

import (
    "encoding/json"
    "fmt"

    "github.com/go-resty/resty/v2"
)

var (
    PhoneNumberID       = "YOUR_PHONE_NUMBER_ID"
    FacebookAccessToken = "**********"
)

type WhatsAppResponse struct {
    Messages []struct {
        ID string `json:"id"`
    } `json:"messages"`
}

func sendWhatsAppMessage(message, recipientNumber string) (*WhatsAppResponse, error) {
    client := resty.New()

    payload := map[string]interface{}{
        "messaging_product": "whatsapp",
        "to":                recipientNumber,
        "type":              "text",
        "text": map[string]string{
            "body": message,
        },
    }

    resp, err := client.R().
        SetHeader("Authorization", "Bearer "+FacebookAccessToken).
        SetHeader("Content-Type", "application/json").
        SetBody(payload).
        Post(fmt.Sprintf("https://graph.facebook.com/v20.0/%s/messages", PhoneNumberID))

    if err != nil {
        return nil, fmt.Errorf("failed to send request: %v", err)
    }

    if resp.IsError() {
        return nil, fmt.Errorf("response error: %s", resp.String())
    }

    var waResp WhatsAppResponse
    if err := json.Unmarshal(resp.Body(), &waResp); err != nil {
        return nil, fmt.Errorf("failed to parse JSON: %v", err)
    }

    return &waResp, nil
}n &waResp, nil
}