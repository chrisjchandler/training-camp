package ml

import (
	"encoding/gob"
	"errors"
	"math"
	"os"
	"sort"
	"training-camp/internal/openai"
)

type Model struct {
	Features [][]float64
	Labels   []string
}

// Euclidean distance between two vectors
func euclidean(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// TrainClassifier stores features and labels for a simple 1-NN classifier
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

// Predict returns the label of the single nearest neighbor
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

// Neighbor holds a neighbor result for Top-K queries
type Neighbor struct {
	Label    string  `json:"label"`
	Distance float64 `json:"distance"`
	Index    int     `json:"index"`
}

// TopK returns the K nearest labeled neighbors
func (m *Model) TopK(vec []float64, k int) []Neighbor {
	if k <= 0 {
		k = 1
	}
	if k > len(m.Features) {
		k = len(m.Features)
	}
	neigh := make([]Neighbor, 0, len(m.Features))
	for i := range m.Features {
		d := euclidean(m.Features[i], vec)
		neigh = append(neigh, Neighbor{Label: m.Labels[i], Distance: d, Index: i})
	}
	sort.Slice(neigh, func(i, j int) bool { return neigh[i].Distance < neigh[j].Distance })
	return neigh[:k]
}
