package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"training-camp/internal/openai"

	"github.com/spf13/cobra"
)

var embedInput string
var embedOutput string

var embedCmd = &cobra.Command{
	Use:   "embed",
	Short: "Generate embeddings from labeled text using OpenAI",
	Run: func(cmd *cobra.Command, args []string) {
		file, err := os.Open(embedInput)
		if err != nil {
			fmt.Println("Error opening input file:", err)
			os.Exit(1)
		}
		defer file.Close()

		var samples []openai.LabeledSample
		if err := json.NewDecoder(file).Decode(&samples); err != nil {
			fmt.Println("Error decoding input JSON:", err)
			os.Exit(1)
		}

		embedded, err := openai.EmbedSamples(samples)
		if err != nil {
			fmt.Println("Error embedding samples:", err)
			os.Exit(1)
		}

		outFile, err := os.Create(embedOutput)
		if err != nil {
			fmt.Println("Error creating output file:", err)
			os.Exit(1)
		}
		defer outFile.Close()

		if err := json.NewEncoder(outFile).Encode(embedded); err != nil {
			fmt.Println("Error encoding output JSON:", err)
			os.Exit(1)
		}

		fmt.Println("Embeddings saved to", embedOutput)
	},
}

func init() {
	embedCmd.Flags().StringVarP(&embedInput, "input", "i", "", "Input JSON file of labeled samples")
	embedCmd.Flags().StringVarP(&embedOutput, "output", "o", "embeddings.json", "Output file to store embeddings")
	embedCmd.MarkFlagRequired("input")
}
