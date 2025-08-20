package ml

import (
	"encoding/gob"
	"errors"
	"math"
	"os"
	"training-camp/internal/openai"
)

type Model struct {
	Features [][]float64
	Labels   []string
}

func euclidean(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func TrainClassifier(data []openai.EmbeddedSample) (*Model, error) {
	if len(data) == 0 {
		return nil, errors.New("no data to train on")
	}

	features := make([][]float64, 0, len(data))
	labels := make([]string, 0, len(data))
	for _, d := range data {
		vec := make([]float64, len(d.Embedding))
		for i, v := range d.Embedding {
			vec[i] = float64(v)
		}
		features = append(features, vec)
		labels = append(labels, d.Label)
	}
	return &Model{Features: features, Labels: labels}, nil
}

func SaveModel(path string, model *Model) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return gob.NewEncoder(file).Encode(model)
}

func LoadModel(path string) (*Model, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var model Model
	err = gob.NewDecoder(file).Decode(&model)
	return &model, err
}

func (m *Model) Predict(vec []float64) string {
	bestIdx := 0
	minDist := euclidean(m.Features[0], vec)
	for i := 1; i < len(m.Features); i++ {
		if dist := euclidean(m.Features[i], vec); dist < minDist {
			minDist = dist
			bestIdx = i
		}
	}
	return m.Labels[bestIdx]
}
