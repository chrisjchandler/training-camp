package cmd

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"training-camp/internal/ml"
	"github.com/spf13/cobra"
)

var serveModel string
var servePort int

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Serve the trained classifier as an HTTP API",
	Run: func(cmd *cobra.Command, args []string) {
		model, err := ml.LoadModel(serveModel)
		if err != nil {
			log.Fatalf("Failed to load model: %v", err)
		}

		http.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}

			var req struct {
				Embedding []float32 `json:"embedding"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "invalid request body", http.StatusBadRequest)
				return
			}

			label := model.Predict(toFloat64(req.Embedding))
			json.NewEncoder(w).Encode(map[string]string{"label": label})
		})

		addr := fmt.Sprintf(":%d", servePort)
		fmt.Printf("Model server running at http://localhost%s/predict\n", addr)
		log.Fatal(http.ListenAndServe(addr, nil))
	},
}

func init() {
	serveCmd.Flags().StringVarP(&serveModel, "model", "m", "model.gob", "Path to trained model file")
	serveCmd.Flags().IntVarP(&servePort, "port", "p", 8080, "Port to run the HTTP server on")
}
