//date: 2022-05-31T16:54:22Z
//url: https://api.github.com/gists/abb17a4e9bf8fb1cb1c40ca726b5f4dd
//owner: https://api.github.com/users/carlosm27

case "POST":
            url := ""
            prompt := &survey.Input{
                Message: "Enter URL:",
            }
            survey.AskOne(prompt, &url)

            body := ""
            prompt = &survey.Input{
                Message: "Enter Body:",
            }
            survey.AskOne(prompt, &body)

            resp, err := client.R().SetBody(body).Post(url)

            if err != nil {
                log.Println(err)
            }

            fmt.Println(resp)
