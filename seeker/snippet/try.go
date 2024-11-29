//date: 2024-11-29T17:11:20Z
//url: https://api.github.com/gists/32491c7543c0fefc8d2905bf1bd76c3a
//owner: https://api.github.com/users/khaledhikmat

package main

import (
	"context"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// TODO:
// Convert to use Gin framework
// Convert to use HTMX
// Upload several PDF books
// Provide a drop down for people to select a book
// Once they select a book, display the summary of the book
// Book summary can be cached
// Provide a text box for people to enter a prompt to ask a question
// Deploy to Google Cloud Run

func main5() {
	ctx := context.Background()
	// Access your API key as an environment variable
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		log.Fatal(err)
		return
	}
	defer client.Close()

	// Set up the model
	model := client.GenerativeModel("gemini-1.5-flash")

	// Setup the data source
	file, err := client.UploadFileFromPath(ctx, filepath.Join("", "islamic.pdf"), nil) // Replace with your test file
	if err != nil {
		log.Fatal(err)
		return
	}
	defer client.DeleteFile(ctx, file.Name)

	// Create an HTML template
	tmpl, err := template.New("response").Parse(`
<!DOCTYPE html>
<html>
<head>
	<title>Book Summary</title>
</head>
<body>
	<h1>Book Answer</h1>
	<p>{{.Summary}}</p>  <!-- Access the Summary field -->
    <hr>
    <form action="/prompt" method="GET">
        Prompt: <input type="text" name="q" value="{{.Prompt}}"> <br>
        <input type="submit" value="Submit">
    </form>
</body>
</html>
`)
	if err != nil {
		log.Fatal(err)
	}

	// Wrap with an API Endpoint
	http.HandleFunc("/prompt", func(w http.ResponseWriter, r *http.Request) {
		query := r.URL.Query()
		promptText := query.Get("q")

		if promptText == "" {
			// Render the form with an empty prompt if none is provided
			err := tmpl.Execute(w, struct{ Summary, Prompt string }{Summary: "", Prompt: ""})
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}

		resp, err := model.GenerateContent(ctx,
			genai.FileData{URI: file.URI},
			genai.Text(promptText))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		var summary string
		for _, cand := range resp.Candidates {
			if cand.Content != nil {
				for _, part := range cand.Content.Parts {
					summary += fmt.Sprintf("%s", part)
				}
			}
		}

		// Set the Content-Type header to text/html
		w.Header().Set("Content-Type", "text/html; charset=utf-8")

		// Execute the template, passing the summary
		err = tmpl.Execute(w, struct{ Summary, Prompt string }{Summary: summary, Prompt: promptText})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("Listening on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
