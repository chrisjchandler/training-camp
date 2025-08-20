package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"training-camp/internal/ml"
	"training-camp/internal/openai"

	"github.com/spf13/cobra"
)

var trainInput string
var modelOutput string

var trainCmd = &cobra.Command{
	Use:   "train",
	Short: "Train a classifier on embedded samples",
	Run: func(cmd *cobra.Command, args []string) {
		file, err := os.Open(trainInput)
		if err != nil {
			fmt.Println("Error opening input file:", err)
			os.Exit(1)
		}
		defer file.Close()

		var embedded []openai.EmbeddedSample
		if err := json.NewDecoder(file).Decode(&embedded); err != nil {
			fmt.Println("Error decoding input JSON:", err)
			os.Exit(1)
		}

		model, err := ml.TrainClassifier(embedded)
		if err != nil {
			fmt.Println("Error training classifier:", err)
			os.Exit(1)
		}

		if err := ml.SaveModel(modelOutput, model); err != nil {
			fmt.Println("Error saving model:", err)
			os.Exit(1)
		}

		fmt.Println("Model saved to", modelOutput)
	},
}

func init() {
	trainCmd.Flags().StringVarP(&trainInput, "input", "i", "embeddings.json", "Input JSON file with embedded samples")
	trainCmd.Flags().StringVarP(&modelOutput, "output", "o", "model.gob", "File path to save the trained model")
}
