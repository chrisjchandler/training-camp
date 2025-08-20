package cmd

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"

	"training-camp/internal/ml"
	"training-camp/internal/openai"

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
			log.Fatalf("failed to load model: %v", err)
		}

		// Minimal UI at "/"
		http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			fmt.Fprint(w, pageHTML)
		})

		// Health check
		http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		})

		// Single prediction: expects one embedding vector
		http.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}

			var req struct {
				Embedding []float32 `json:"embedding"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Embedding) == 0 {
				http.Error(w, "invalid request body", http.StatusBadRequest)
				log.Printf("invalid /predict body: %v", err)
				return
			}

			label := cleanLabel(model.Predict(f32ToF64(req.Embedding)))
			log.Printf("/predict -> %s", label)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"label": label})
		})

		// Batch prediction: expects multiple embedding vectors
		http.HandleFunc("/predict/batch", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}

			var req struct {
				Embeddings [][]float32 `json:"embeddings"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Embeddings) == 0 {
				http.Error(w, "invalid request body", http.StatusBadRequest)
				log.Printf("invalid /predict/batch body: %v", err)
				return
			}

			preds := make([]string, len(req.Embeddings))
			for i, emb := range req.Embeddings {
				preds[i] = cleanLabel(model.Predict(f32ToF64(emb)))
			}
			log.Printf("/predict/batch -> %d predictions", len(preds))
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string][]string{"labels": preds})
		})

		// Text classification (server-side embedding via OpenAI)
		http.HandleFunc("/classify_text", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}

			var req struct {
				Text string `json:"text"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Text == "" {
				http.Error(w, "invalid request body", http.StatusBadRequest)
				log.Printf("invalid /classify_text body: %v", err)
				return
			}

			// optional: extract source phrase from meta-questions like:
			// "how do i say X in latin" -> "X"
			q := strings.ToLower(strings.TrimSpace(req.Text))
			if strings.HasPrefix(q, "how do i say ") && strings.HasSuffix(q, " in latin") {
				inner := strings.TrimSuffix(strings.TrimPrefix(q, "how do i say "), " in latin")
				if inner != "" {
					req.Text = inner
				}
			}

			// Embed on server (keeps API key off the client)
			emb, err := openai.EmbedSamples([]openai.LabeledSample{{Text: req.Text, Label: ""}})
			if err != nil || len(emb) == 0 {
				http.Error(w, "embedding failed", http.StatusInternalServerError)
				log.Printf("embedding failed: %v", err)
				return
			}

			label := cleanLabel(model.Predict(f32ToF64(emb[0].Embedding)))
			log.Printf("/classify_text -> %s", label)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"label": label})
		})

		// Top-K neighbors for transparency/debugging
		// GET or POST /classify_text_topk?k=5  (POST body: {"text": "..."}; GET also supports ?text=...)
		http.HandleFunc("/classify_text_topk", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusOK)
				return
			}

			// read k from query
			k := 3
			if qk := r.URL.Query().Get("k"); qk != "" {
				if n, err := strconv.Atoi(qk); err == nil && n > 0 {
					k = n
				}
			}

			var text string
			if r.Method == http.MethodPost {
				var req struct{ Text string `json:"text"` }
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Text == "" {
					http.Error(w, "invalid request body", http.StatusBadRequest)
					log.Printf("invalid /classify_text_topk body: %v", err)
					return
				}
				text = req.Text
			} else { // allow GET ?text=...
				text = r.URL.Query().Get("text")
				if text == "" {
					http.Error(w, "missing text", http.StatusBadRequest)
					return
				}
			}

			// optional meta-question extraction (same as in /classify_text)
			q := strings.ToLower(strings.TrimSpace(text))
			if strings.HasPrefix(q, "how do i say ") && strings.HasSuffix(q, " in latin") {
				inner := strings.TrimSuffix(strings.TrimPrefix(q, "how do i say "), " in latin")
				if inner != "" {
					text = inner
				}
			}

			emb, err := openai.EmbedSamples([]openai.LabeledSample{{Text: text}})
			if err != nil || len(emb) == 0 {
				http.Error(w, "embedding failed", http.StatusInternalServerError)
				log.Printf("embedding failed: %v", err)
				return
			}

			neigh := model.TopK(f32ToF64(emb[0].Embedding), k)
			for i := range neigh {
				neigh[i].Label = cleanLabel(neigh[i].Label)
			}

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"text":      text,
				"neighbors": neigh,
				"k":         k,
				"note":      "Top-K nearest labeled examples. Increase k for more alternatives.",
			})
		})

		addr := fmt.Sprintf(":%d", servePort)
		log.Printf("model server running at http://localhost%s", addr)
		log.Printf("endpoints: / (UI), /healthz, /classify_text, /classify_text_topk, /predict, /predict/batch")
		log.Fatal(http.ListenAndServe(addr, nil))
	},
}

func applyCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")              // tighten in production
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS") // allowed methods
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")  // allowed headers
}

// renamed to avoid collision with cmd/eval.go
func f32ToF64(f32 []float32) []float64 {
	f64 := make([]float64, len(f32))
	for i, v := range f32 {
		f64[i] = float64(v)
	}
	return f64
}

// cleanLabel trims quotes, backticks, and surrounding whitespace
func cleanLabel(s string) string {
	// leading
	for len(s) > 0 {
		switch s[0] {
		case ' ', '\t', '\n', '\r', '"', '\'', '`':
			s = s[1:]
		default:
			goto tail
		}
	}
tail:
	// trailing
	for len(s) > 0 {
		switch s[len(s)-1] {
		case ' ', '\t', '\n', '\r', '"', '\'', '`':
			s = s[:len(s)-1]
		default:
			return s
		}
	}
	return s
}

func init() {
	serveCmd.Flags().StringVarP(&serveModel, "model", "m", "model.gob", "Path to trained model file")
	serveCmd.Flags().IntVarP(&servePort, "port", "p", 8080, "Port to run the HTTP server on")
}

const pageHTML = `<!doctype html>
<meta charset="utf-8">
<title>Training Camp — Minimal UI</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
  h1 { font-size: 1.4rem; margin-bottom: 1rem; }
  .row { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
  textarea, input[type=text] { width: 100%; max-width: 720px; padding: .6rem; font-size: 1rem; }
  button { padding: .6rem 1rem; border-radius: .5rem; border: 1px solid #888; background: #f2f2f2; cursor: pointer; }
  pre { background: #111; color: #eee; padding: 1rem; border-radius: .5rem; overflow: auto; max-width: 760px; }
  .hint { color: #666; font-size: .9rem; }
</style>

<h1>Training Camp — Text Classifier</h1>

<p class="hint">
  This UI sends your text to <code>/classify_text</code>. The server embeds it with OpenAI and classifies using your trained model.
</p>

<div class="row">
  <textarea id="t" rows="4" placeholder="Type text here..."></textarea>
</div>
<div class="row">
  <button id="go">Classify</button>
</div>
<pre id="out"></pre>

<script>
(async () => {
  const $ = sel => document.querySelector(sel);
  $('#go').onclick = async () => {
    const text = $('#t').value.trim();
    if (!text) { $('#out').textContent = 'Please enter some text.'; return; }
    try {
      const res = await fetch('/classify_text', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      $('#out').textContent = JSON.stringify(data, null, 2);
    } catch (e) {
      $('#out').textContent = 'Error: ' + e;
    }
  };
})();
</script>
`
