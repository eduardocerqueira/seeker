//date: 2022-05-17T17:10:17Z
//url: https://api.github.com/gists/60b8672597b95294191fff45cbd0fb91
//owner: https://api.github.com/users/sparklive

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/chromedp/cdproto/cdp"
	"github.com/chromedp/cdproto/network"
	"github.com/chromedp/chromedp"
)

func main() {
	// create chrome instance
	ctx, cancel := chromedp.NewContext(
		context.Background(),
		chromedp.WithLogf(log.Printf),
	)
	defer cancel()

	// create a timeout
	ctx, cancel = context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	// navigate to a page, wait for an element, click
	var example string
	err := chromedp.Run(ctx,
		SetCookie("test", "vv", "golang.org", "/", false, false),
		chromedp.Navigate(`https://golang.org/pkg/time/`),
		chromedp.WaitVisible(`body > footer`),                            // wait for footer element is visible (ie, page is loaded)
		chromedp.Click(`#pkg-examples > div`, chromedp.NodeVisible),      // find and click "Expand All" link
		chromedp.Value(`#example_After .play .input textarea`, &example), // retrieve the value of the textarea
		ShowCookies(),
	)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Go's time.After example:\n%s", example)
}

func SetCookie(name, value, domain, path string, httpOnly, secure bool) chromedp.Action {
	return chromedp.ActionFunc(func(ctx context.Context) error {
		expr := cdp.TimeSinceEpoch(time.Now().Add(180 * 24 * time.Hour))
		success, err := network.SetCookie(name, value).
			WithExpires(&expr).
			WithDomain(domain).
			WithPath(path).
			WithHTTPOnly(httpOnly).
			WithSecure(secure).
			Do(ctx)
		if err != nil {
			return err
		}
		if !success {
			return fmt.Errorf("could not set cookie %s", name)
		}
		return nil
	})
}

func ShowCookies() chromedp.Action {
	return chromedp.ActionFunc(func(ctx context.Context) error {
		cookies, err := network.GetAllCookies().Do(ctx)
		if err != nil {
			return err
		}
		for i, cookie := range cookies {
			log.Printf("chrome cookie %d: %+v", i, cookie)
		}
		return nil
	})
}
