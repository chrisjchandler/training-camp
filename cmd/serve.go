package cmd

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"training-camp/internal/ml"
	"training-camp/internal/openai"

	"github.com/spf13/cobra"
)

var serveModel string
var servePort int

// Auto-bootstrap flags (optional at startup)
var autoTask string
var autoCount int
var autoEvery string // Go duration string, e.g. "10m", "1h"

type serverState struct {
	mu        sync.RWMutex
	model     *ml.Model
	modelPath string

	sched struct {
		mu       sync.Mutex
		running  bool
		task     string
		count    int
		interval time.Duration
		stopCh   chan struct{}
	}
}

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Serve the trained classifier as an HTTP API",
	Run: func(cmd *cobra.Command, args []string) {
		model, err := ml.LoadModel(serveModel)
		if err != nil {
			log.Fatalf("failed to load model: %v", err)
		}
		s := &serverState{model: model, modelPath: serveModel}

		// ---------- Minimal UI ----------
		http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			fmt.Fprint(w, pageHTML)
		})

		// ---------- Health ----------
		http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		})

		// ---------- Prediction endpoints ----------
		http.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions { w.WriteHeader(http.StatusOK); return }

			var req struct{ Embedding []float32 `json:"embedding"` }
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Embedding) == 0 {
				http.Error(w, "invalid request body", http.StatusBadRequest); log.Printf("invalid /predict body: %v", err); return
			}

			s.mu.RLock()
			label := cleanLabel(s.model.Predict(f32ToF64(req.Embedding)))
			s.mu.RUnlock()

			log.Printf("/predict -> %s", label)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"label": label})
		})

		http.HandleFunc("/predict/batch", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions { w.WriteHeader(http.StatusOK); return }

			var req struct{ Embeddings [][]float32 `json:"embeddings"` }
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Embeddings) == 0 {
				http.Error(w, "invalid request body", http.StatusBadRequest); log.Printf("invalid /predict/batch body: %v", err); return
			}

			out := make([]string, len(req.Embeddings))
			s.mu.RLock()
			for i, emb := range req.Embeddings {
				out[i] = cleanLabel(s.model.Predict(f32ToF64(emb)))
			}
			s.mu.RUnlock()

			log.Printf("/predict/batch -> %d predictions", len(out))
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string][]string{"labels": out})
		})

		// ---------- Server-side embedding + prediction ----------
		http.HandleFunc("/classify_text", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions { w.WriteHeader(http.StatusOK); return }

			var req struct{ Text string `json:"text"` }
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Text == "" {
				http.Error(w, "invalid request body", http.StatusBadRequest); log.Printf("invalid /classify_text body: %v", err); return
			}
			req.Text = normalizeQuestion(req.Text)

			emb, err := openai.EmbedSamples([]openai.LabeledSample{{Text: req.Text}})
			if err != nil || len(emb) == 0 {
				http.Error(w, "embedding failed", http.StatusInternalServerError); log.Printf("embedding failed: %v", err); return
			}

			s.mu.RLock()
			label := cleanLabel(s.model.Predict(f32ToF64(emb[0].Embedding)))
			s.mu.RUnlock()

			log.Printf("/classify_text -> %s", label)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"label": label})
		})

		http.HandleFunc("/classify_text_with_score", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions { w.WriteHeader(http.StatusOK); return }

			var req struct{ Text string `json:"text"` }
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Text == "" {
				http.Error(w, "invalid request body", http.StatusBadRequest); log.Printf("invalid /classify_text_with_score body: %v", err); return
			}
			req.Text = normalizeQuestion(req.Text)

			emb, err := openai.EmbedSamples([]openai.LabeledSample{{Text: req.Text}})
			if err != nil || len(emb) == 0 {
				http.Error(w, "embedding failed", http.StatusInternalServerError); log.Printf("embedding failed: %v", err); return
			}

			s.mu.RLock()
			pred, d1, d2, margin := s.model.Score(f32ToF64(emb[0].Embedding))
			s.mu.RUnlock()

			out := map[string]any{"label": cleanLabel(pred), "d1": d1, "d2": d2, "margin": margin, "note": "larger margin = more confident"}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(out)
		})

		http.HandleFunc("/classify_text_topk", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions { w.WriteHeader(http.StatusOK); return }

			k := 3
			if qk := r.URL.Query().Get("k"); qk != "" {
				if n, err := strconv.Atoi(qk); err == nil && n > 0 { k = n }
			}

			var text string
			if r.Method == http.MethodPost {
				var req struct{ Text string `json:"text"` }
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Text == "" {
					http.Error(w, "invalid request body", http.StatusBadRequest); log.Printf("invalid /classify_text_topk body: %v", err); return
				}
				text = req.Text
			} else {
				text = r.URL.Query().Get("text")
				if text == "" { http.Error(w, "missing text", http.StatusBadRequest); return }
			}
			text = normalizeQuestion(text)

			emb, err := openai.EmbedSamples([]openai.LabeledSample{{Text: text}})
			if err != nil || len(emb) == 0 {
				http.Error(w, "embedding failed", http.StatusInternalServerError); log.Printf("embedding failed: %v", err); return
			}

			s.mu.RLock()
			neigh := s.model.TopK(f32ToF64(emb[0].Embedding), k)
			s.mu.RUnlock()
			for i := range neigh { neigh[i].Label = cleanLabel(neigh[i].Label) }

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"text": text, "neighbors": neigh, "k": k})
		})

		// ---------- Online learning endpoints ----------
		http.HandleFunc("/learn_text", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions { w.WriteHeader(http.StatusOK); return }

			var req struct{ Text, Label string }
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Text == "" || req.Label == "" {
				http.Error(w, "invalid request body", http.StatusBadRequest); log.Printf("invalid /learn_text body: %v", err); return
			}

			emb, err := openai.EmbedSamples([]openai.LabeledSample{{Text: req.Text, Label: req.Label}})
			if err != nil || len(emb) == 0 {
				http.Error(w, "embedding failed", http.StatusInternalServerError); log.Printf("embedding failed: %v", err); return
			}

			s.mu.Lock()
			s.model.AddEmbeddedSample(emb[0])
			if err := ml.SaveModel(s.modelPath, s.model); err != nil {
				s.mu.Unlock()
				http.Error(w, "failed to save model", http.StatusInternalServerError); log.Printf("save failed: %v", err); return
			}
			s.mu.Unlock()

			log.Printf("/learn_text -> appended label %q", req.Label)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "added_label": cleanLabel(req.Label)})
		})

		http.HandleFunc("/bootstrap", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions { w.WriteHeader(http.StatusOK); return }

			var req struct{ Task string; Count int }
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Task == "" || req.Count <= 0 {
				http.Error(w, "invalid request body", http.StatusBadRequest); log.Printf("invalid /bootstrap body: %v", err); return
			}

			added, err := s.bootstrapOnce(req.Task, req.Count)
			if err != nil { http.Error(w, err.Error(), http.StatusInternalServerError); return }

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "added": added})
		})

		// ---------- Scheduler control endpoints ----------
		http.HandleFunc("/scheduler/start", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions { w.WriteHeader(http.StatusOK); return }

			var req struct {
				Task  string `json:"task"`
				Count int    `json:"count"`
				Every string `json:"every"` // e.g. "10m"
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Task == "" || req.Count <= 0 || req.Every == "" {
				http.Error(w, "invalid request body", http.StatusBadRequest); return
			}
			d, err := time.ParseDuration(req.Every)
			if err != nil || d <= 0 { http.Error(w, "invalid every duration", http.StatusBadRequest); return }

			if err := s.startScheduler(req.Task, req.Count, d); err != nil {
				http.Error(w, err.Error(), http.StatusConflict); return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "task": req.Task, "count": req.Count, "every": d.String()})
		})

		http.HandleFunc("/scheduler/stop", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			if r.Method == http.MethodOptions { w.WriteHeader(http.StatusOK); return }
			if err := s.stopScheduler(); err != nil {
				http.Error(w, err.Error(), http.StatusConflict); return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
		})

		http.HandleFunc("/scheduler/status", func(w http.ResponseWriter, r *http.Request) {
			applyCORS(w)
			s.sched.mu.Lock()
			st := map[string]any{
				"running": s.sched.running,
				"task":    s.sched.task,
				"count":   s.sched.count,
				"every":   s.sched.interval.String(),
			}
			s.sched.mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(st)
		})

		// Auto-start scheduler if flags provided
		if autoTask != "" && autoCount > 0 && autoEvery != "" {
			if dur, err := time.ParseDuration(autoEvery); err == nil && dur > 0 {
				if err := s.startScheduler(autoTask, autoCount, dur); err != nil {
					log.Printf("auto-scheduler error: %v", err)
				} else {
					log.Printf("auto-scheduler started: task=%q count=%d every=%s", autoTask, autoCount, dur)
				}
			} else {
				log.Printf("invalid --auto-every duration: %q", autoEvery)
			}
		}

		addr := fmt.Sprintf(":%d", servePort)
		log.Printf("model server running at http://localhost%s", addr)
		log.Printf("endpoints: / (UI), /healthz, /classify_text, /classify_text_with_score, /classify_text_topk, /predict, /predict/batch, /learn_text, /bootstrap, /scheduler/*")
		log.Fatal(http.ListenAndServe(addr, nil))
	},
}

// ---- scheduler helpers ----

func (s *serverState) startScheduler(task string, count int, every time.Duration) error {
	s.sched.mu.Lock()
	defer s.sched.mu.Unlock()
	if s.sched.running {
		return fmt.Errorf("scheduler already running")
	}
	s.sched.task = task
	s.sched.count = count
	s.sched.interval = every
	s.sched.stopCh = make(chan struct{})
	s.sched.running = true

	go func() {
		ticker := time.NewTicker(every)
		defer ticker.Stop()
		log.Printf("[scheduler] started: task=%q count=%d every=%s", task, count, every)

		// run immediately once
		if added, err := s.bootstrapOnce(task, count); err != nil {
			log.Printf("[scheduler] bootstrap error: %v", err)
		} else {
			log.Printf("[scheduler] bootstrap added=%d", added)
		}

		for {
			select {
			case <-ticker.C:
				if added, err := s.bootstrapOnce(task, count); err != nil {
					log.Printf("[scheduler] bootstrap error: %v", err)
				} else {
					log.Printf("[scheduler] bootstrap added=%d", added)
				}
			case <-s.sched.stopCh:
				log.Printf("[scheduler] stopped")
				return
			}
		}
	}()

	return nil
}

func (s *serverState) stopScheduler() error {
	s.sched.mu.Lock()
	defer s.sched.mu.Unlock()
	if !s.sched.running {
		return fmt.Errorf("scheduler not running")
	}
	close(s.sched.stopCh)
	s.sched.running = false
	return nil
}

func (s *serverState) bootstrapOnce(task string, count int) (int, error) {
	// 1) Generate labeled pairs via OpenAI
	pairs, err := openai.GenerateSamples(task, count)
	if err != nil { return 0, fmt.Errorf("generation failed: %w", err) }

	// 2) Embed them
	emb, err := openai.EmbedSamples(pairs)
	if err != nil { return 0, fmt.Errorf("embedding failed: %w", err) }

	// 3) Append & save
	s.mu.Lock()
	for _, e := range emb { s.model.AddEmbeddedSample(e) }
	if err := ml.SaveModel(s.modelPath, s.model); err != nil {
		s.mu.Unlock()
		return 0, fmt.Errorf("save failed: %w", err)
	}
	s.mu.Unlock()
	return len(emb), nil
}

// ---- misc helpers ----

func applyCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")              // tighten in production
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS") // allowed methods
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")  // allowed headers
}

// avoid collision with cmd/eval.go
func f32ToF64(f32 []float32) []float64 {
	f64 := make([]float64, len(f32))
	for i, v := range f32 { f64[i] = float64(v) }
	return f64
}

func cleanLabel(s string) string {
	for len(s) > 0 {
		switch s[0] { case ' ', '\t', '\n', '\r', '"', '\'', '`': s = s[1:]; default: goto tail }
	}
tail:
	for len(s) > 0 {
		switch s[len(s)-1] { case ' ', '\t', '\n', '\r', '"', '\'', '`': s = s[:len(s)-1]; default: return s }
	}
	return s
}

// Pull out "X" from "how do i say X in latin"
func normalizeQuestion(text string) string {
	q := strings.ToLower(strings.TrimSpace(text))
	if strings.HasPrefix(q, "how do i say ") && strings.HasSuffix(q, " in latin") {
		inner := strings.TrimSuffix(strings.TrimPrefix(q, "how do i say "), " in latin")
		if inner != "" { return inner }
	}
	return text
}

func init() {
	serveCmd.Flags().StringVarP(&serveModel, "model", "m", "model.gob", "Path to trained model file")
	serveCmd.Flags().IntVarP(&servePort, "port", "p", 8080, "Port to run the HTTP server on")

	serveCmd.Flags().StringVar(&autoTask, "auto-task", "", "Auto-bootstrap task description (optional)")
	serveCmd.Flags().IntVar(&autoCount, "auto-count", 0, "Auto-bootstrap sample count (optional)")
	serveCmd.Flags().StringVar(&autoEvery, "auto-every", "", "Auto-bootstrap interval (e.g., 10m, 1h) (optional)")
}

const pageHTML = `<!doctype html>
<meta charset="utf-8">
<title>Training Camp — Text Classifier</title>
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
<p class="hint">This UI calls <code>/classify_text</code>. You can also configure a background scheduler for continuous learning.</p>

<div class="row">
  <textarea id="t" rows="4" placeholder="Type text here..."></textarea>
</div>
<div class="row">
  <button id="go">Classify</button>
  <button id="score">Classify + Score</button>
</div>
<pre id="out"></pre>

<script>
const $ = s => document.querySelector(s);
async function post(url, body) {
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  return res.json();
}
$('#go').onclick = async () => {
  const text = $('#t').value.trim();
  if (!text) { $('#out').textContent='Enter text'; return; }
  $('#out').textContent = JSON.stringify(await post('/classify_text', {text}), null, 2);
};
$('#score').onclick = async () => {
  const text = $('#t').value.trim();
  if (!text) { $('#out').textContent='Enter text'; return; }
  $('#out').textContent = JSON.stringify(await post('/classify_text_with_score', {text}), null, 2);
};
</script>
`
