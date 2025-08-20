package openai

import (
	"context"
	"fmt"
	"os"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

type LabeledSample struct {
	Text  string `json:"text"`
	Label string `json:"label"`
}

func GenerateSamples(task string, count int) ([]LabeledSample, error) {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	prompt := fmt.Sprintf("Generate %d labeled text classification examples for the task: '%s'. Use format: <text> => <label>", count, task)
	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: openai.GPT4,
		Messages: []openai.ChatCompletionMessage{
			{Role: "system", Content: "You generate training data."},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		return nil, err
	}

	entries := strings.Split(resp.Choices[0].Message.Content, "\n")
	var samples []LabeledSample
	for _, line := range entries {
		parts := strings.Split(line, "=>")
		if len(parts) == 2 {
			samples = append(samples, LabeledSample{
				Text:  strings.TrimSpace(parts[0]),
				Label: strings.TrimSpace(parts[1]),
			})
		}
	}
	return samples, nil
}
