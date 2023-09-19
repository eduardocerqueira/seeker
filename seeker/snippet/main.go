//date: 2023-09-19T17:08:25Z
//url: https://api.github.com/gists/e61b532b1163fc1c4ae1b2d1cea42645
//owner: https://api.github.com/users/Struki84

func Prompt(input string, options ...chains.ChainCallOption) {
	llm, err := openai.NewChat(openai.WithModel("gpt-4"))
	if err != nil {
		log.Fatal(err)
	}

	// dsn : "**********"
	// memory := memory.NewPostgreBuffer(dsn)
	// memory.SetSession("USID-001")

	ctx := context.Background()

	search, err := duckduckgo.New(5, "")
	if err != nil {
		log.Fatal(err)
	}

	agentTools := []tools.Tool{search}

	// executor, err := agents.Initialize(
	// 	llm,
	// 	agentTools,
	// 	agents.ConversationalReactDescription,
	// 	agents.WithMemory(memory),
	// 	agents.WithReturnIntermediateSteps(), // This throws an error
	// )

	executor, err := agents.Initialize(
		llm,
		agentTools,
		agents.ConversationalReactDescription,
	)

	if err != nil {
		log.Fatal(err)
	}

	response, err := chains.Run(
		ctx,
		executor,
		input,
		chains.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			fmt.Print(string(chunk))
			return nil
		}),
	)

	fmt.Println(response)

	if err != nil {
		log.Fatal(err)
	}
}
)

	fmt.Println(response)

	if err != nil {
		log.Fatal(err)
	}
}
