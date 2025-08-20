package cmd

import (
	"fmt"
	"os"
	"training-camp/internal/openai"

	"github.com/spf13/cobra"
)

var task string
var count int

var generateCmd = &cobra.Command{
	Use:   "generate",
	Short: "Generate labeled training data via OpenAI",
	Run: func(cmd *cobra.Command, args []string) {
		data, err := openai.GenerateSamples(task, count)
		if err != nil {
			fmt.Println("Error generating samples:", err)
			os.Exit(1)
		}
		for _, pair := range data {
			fmt.Printf("%s => %s\n", pair.Text, pair.Label)
		}
	},
}

func init() {
	generateCmd.Flags().StringVarP(&task, "task", "t", "", "Task description (e.g. 'watch-auth')")
	generateCmd.Flags().IntVarP(&count, "count", "c", 10, "Number of examples to generate")
	generateCmd.MarkFlagRequired("task")
}
