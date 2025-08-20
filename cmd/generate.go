// cmd/generate.go
package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"training-camp/internal/openai"

	"github.com/spf13/cobra"
)

var task string
var count int
var outPath string

var generateCmd = &cobra.Command{
	Use:   "generate",
	Short: "Generate labeled training data via OpenAI (outputs JSON)",
	Run: func(cmd *cobra.Command, args []string) {
		data, err := openai.GenerateSamples(task, count)
		if err != nil {
			fmt.Println("Error generating samples:", err)
			os.Exit(1)
		}

		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		var f *os.File

		if outPath != "" {
			var err error
			f, err = os.Create(outPath)
			if err != nil {
				fmt.Println("Error creating output file:", err)
				os.Exit(1)
			}
			defer f.Close()
			enc = json.NewEncoder(f)
			enc.SetIndent("", "  ")
		}

		if err := enc.Encode(data); err != nil {
			fmt.Println("Error writing JSON:", err)
			os.Exit(1)
		}

		if outPath != "" {
			fmt.Println("Wrote samples to", outPath)
		}
	},
}

func init() {
	generateCmd.Flags().StringVarP(&task, "task", "t", "", "Task description (e.g. 'watch-auth')")
	generateCmd.Flags().IntVarP(&count, "count", "c", 10, "Number of examples to generate")
	generateCmd.Flags().StringVarP(&outPath, "output", "o", "", "Optional file path to write JSON output")
	generateCmd.MarkFlagRequired("task")
}
