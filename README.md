# Training Camp

A Go-based CLI to generate, embed, train, serve, and evaluate narrow AI models using OpenAI embeddings and a custom KNN-style logic — no Python required.

---

## Features

-  Prompt-based training data generation via OpenAI
-  Embedding via OpenAI's `text-embedding-ada-002`
-  Custom KNN (1-NN) classifier in pure Go (serializable)
-  Accuracy evaluation tools
-  Offline prediction API with `training-camp serve`
-  Batch prediction endpoint with multi-label support
-  Detailed logging for prediction events
-  Task-based config YAMLs for reusable pipelines

---

## Install

```bash
git clone https://github.com/your-org/training-camp.git
cd training-camp
go build -o training-camp
```

---

## Quickstart 

```bash
# 1. Generate training examples
training-camp generate --task "Classify watch descriptions as real or fake" --count 100 > watch.json

# 2. Embed them via OpenAI
training-camp embed -i watch.json -o watch.embeddings.json

# 3. Train a classifier
training-camp train -i watch.embeddings.json -o watch.model.gob

# 4. Evaluate on test data
training-camp eval -m watch.model.gob -i test.embeddings.json

# 5. Serve predictions
training-camp serve -m watch.model.gob -p 8080
```

---

## Serving API

### Single Prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"embedding": [0.12, 0.47, ...]}'
```
Returns:
```json
{"label": "real"}
```

### Batch Prediction
```bash
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"embeddings": [[0.12, 0.47, ...], [0.21, 0.19, ...]]}'
```
Returns:
```json
{"labels": ["real", "fake"]}
```

### Logging
- Each request and prediction is logged to stdout.
- Batch requests log the number of predictions processed.
- Invalid requests are logged with error details.

>  CORS headers enabled for browser compatibility.
>  Logs provide traceability for debugging and auditing.

---

## Hugging Face Notes

You do NOT need Hugging Face unless you want to:
- Fine-tune a transformer model (Python + PyTorch)
- Use pre-trained HF models for embeddings
- Export or host on the Hugging Face Hub

If needed:
```bash
pip install sentence-transformers
python -m onnxruntime_tools.convert --model_path ./model --output_path ./model.onnx
```
Then use `onnx-go` for inference.

---

## Project Layout

```
training-camp/
├── main.go              # CLI entry
├── cmd/                 # Cobra commands
├── internal/            # OpenAI + ML logic
├── config/              # Task configs (YAML)
├── models/              # Trained models
├── templates/           # Prompt templates (optional)
├── output/              # Generated data & embeddings
└── README.md
```

---

##  Use Cases

- Generate lightweight classifiers from labeled text
- Deploy offline AI APIs for internal workflows

---

##  Roadmap
- [x] OpenAI generation + embedding
- [x] Offline training & eval
- [x] Serve predictions via HTTP
- [x] Batch prediction support
- [x] Logging functions
- [ ] Multi-label classification with top-N scoring
- [ ] Export predictions in batch to file (CSV/JSON)
- [ ] Config-driven fine-tuning presets
- [ ] Add `/healthz` and `/version` endpoints

---

Note: requires an OPENAI API key to perform training functions

##  License
MIT
