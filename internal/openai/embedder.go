package openai

import (
	"context"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

type EmbeddedSample struct {
	Label     string    `json:"label"`
	Embedding []float32 `json:"embedding"`
}

func EmbedSamples(samples []LabeledSample) ([]EmbeddedSample, error) {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	var embedded []EmbeddedSample

	for _, s := range samples {
		resp, err := client.CreateEmbeddings(context.Background(), openai.EmbeddingRequest{
			Input: []string{s.Text},
			Model: openai.AdaEmbeddingV2,
		})
		if err != nil {
			return nil, err
		}
		embedded = append(embedded, EmbeddedSample{
			Label:     s.Label,
			Embedding: resp.Data[0].Embedding,
		})
	}
	return embedded, nil
}
