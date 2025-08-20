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
	f, err := os.Create(path)
	if err != nil { return err }
	defer f.Close()
	return gob.NewEncoder(f).Encode(model)
}

func LoadModel(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil { return nil, err }
	defer f.Close()
	var m Model
	return &m, gob.NewDecoder(f).Decode(&m)
}

// Online learning: append a new (vec,label)
func (m *Model) AddVector(vec []float64, label string) {
	m.Features = append(m.Features, vec)
	m.Labels = append(m.Labels, label)
}

// Convenience for EmbeddedSample
func (m *Model) AddEmbeddedSample(s openai.EmbeddedSample) {
	v := make([]float64, len(s.Embedding))
	for i, f := range s.Embedding { v[i] = float64(f) }
	m.AddVector(v, s.Label)
}

// Predict returns the label of the single nearest neighbor
func (m *Model) Predict(vec []float64) string {
	bestIdx := 0
	minDist := euclidean(m.Features[0], vec)
	for i := 1; i < len(m.Features); i++ {
		if d := euclidean(m.Features[i], vec); d < minDist {
			minDist = d
			bestIdx = i
		}
	}
	return m.Labels[bestIdx]
}

type Neighbor struct {
	Label    string  `json:"label"`
	Distance float64 `json:"distance"`
	Index    int     `json:"index"`
}

// TopK returns the K nearest labeled neighbors
func (m *Model) TopK(vec []float64, k int) []Neighbor {
	if k <= 0 { k = 1 }
	if k > len(m.Features) { k = len(m.Features) }
	neigh := make([]Neighbor, 0, len(m.Features))
	for i := range m.Features {
		d := euclidean(m.Features[i], vec)
		neigh = append(neigh, Neighbor{Label: m.Labels[i], Distance: d, Index: i})
	}
	sort.Slice(neigh, func(i, j int) bool { return neigh[i].Distance < neigh[j].Distance })
	return neigh[:k]
}

// Score returns predicted label + d1 + d2 + margin (d2 - d1). Larger margin = more confident.
func (m *Model) Score(vec []float64) (label string, d1, d2, margin float64) {
	neigh := m.TopK(vec, 2)
	label = neigh[0].Label
	d1 = neigh[0].Distance
	if len(neigh) > 1 { d2 = neigh[1].Distance } else { d2 = d1 }
	margin = d2 - d1
	return
}
