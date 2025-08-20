package cmd

import "github.com/spf13/cobra"

var rootCmd = &cobra.Command{
	Use:   "training-camp",
	Short: "Train narrow AI models using OpenAI embeddings and Go classifiers",
}

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.AddCommand(generateCmd)
	rootCmd.AddCommand(embedCmd)
	rootCmd.AddCommand(trainCmd)
	rootCmd.AddCommand(evalCmd)
	rootCmd.AddCommand(serveCmd)
}
