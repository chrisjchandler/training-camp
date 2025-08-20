package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"training-camp/internal/ml"
	"training-camp/internal/openai"

	"github.com/spf13/cobra"
)

var evalModel string
var evalData string

var evalCmd = &cobra.Command{
	Use:   "eval",
	Short: "Evaluate a trained classifier on embedded test data",
	Run: func(cmd *cobra.Command, args []string) {
		model, err := ml.LoadModel(evalModel)
		if err != nil {
			fmt.Println("Error loading model:", err)
			os.Exit(1)
		}

		file, err := os.Open(evalData)
		if err != nil {
			fmt.Println("Error opening eval data file:", err)
			os.Exit(1)
		}
		defer file.Close()

		var data []openai.EmbeddedSample
		if err := json.NewDecoder(file).Decode(&data); err != nil {
			fmt.Println("Error decoding eval data:", err)
			os.Exit(1)
		}

		total := len(data)
		correct := 0
		for _, sample := range data {
			pred := model.Predict(toFloat64(sample.Embedding))
			if pred == sample.Label {
				correct++
			}
		}

		fmt.Printf("Accuracy: %.2f%% (%d/%d correct)\n", float64(correct)*100/float64(total), correct, total)
	},
}

func toFloat64(f32 []float32) []float64 {
	f64 := make([]float64, len(f32))
	for i, v := range f32 {
		f64[i] = float64(v)
	}
	return f64
}

func init() {
	evalCmd.Flags().StringVarP(&evalModel, "model", "m", "model.gob", "Path to trained model")
	evalCmd.Flags().StringVarP(&evalData, "input", "i", "test_embeddings.json", "Evaluation dataset (embedded)")
}
