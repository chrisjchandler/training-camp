# Training Camp

A Go-based CLI to generate, embed, train, serve, and evaluate narrow AI models using OpenAI embeddings and KNN-style logic — no Python required.

---

## Features

- Prompt-based training data generation via OpenAI
- Embedding via OpenAI's `text-embedding-ada-002`
- Custom 1-NN classifier in pure Go (serializable)
- Accuracy evaluation tools
- Offline prediction API with `training-camp serve`
- Batch prediction endpoint
- Multi-label / top-K neighbor support
- Minimal browser-based UI served at `/`
- Task-based config YAMLs for reusable pipelines

---

## Install

```bash
git clone https://github.com/your-org/training-camp.git
cd training-camp
go build -o training-camp

Quickstart (Latin Translation Example)
# 1. Generate training examples
./training-camp generate --task "translate english text to latin" --count 20 --output latin.json

# 2. Embed them via OpenAI
./training-camp embed -i latin.json -o latin.embeddings.json

# 3. Train a classifier
./training-camp train -i latin.embeddings.json -o latin.model.gob

# 4. Evaluate on test data
./training-camp eval -m latin.model.gob -i test.embeddings.json

# 5. Serve predictions (HTTP API + UI)
./training-camp serve -m latin.model.gob -p 8080


Serving API

Once training-camp serve is running:

Health check
curl http://localhost:8080/healthz

Classify raw embedding
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"embedding": [0.12, 0.47, ...]}'


Response:
{"label": "salve mundi"}

Batch classification
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"embeddings": [[0.1,0.2,...], [0.3,0.4,...]]}'
{"label": "salve mundi"}


Response:
{"labels": ["salve mundi", "te amo"]}

Text classification (server embeds via OpenAI)
curl -X POST http://localhost:8080/classify_text \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I say hello in Latin?"}'

Response 
{"label": "salve"}

Top-K neighbors

curl -X POST "http://localhost:8080/classify_text_topk?k=3" \
  -H "Content-Type: application/json" \
  -d '{"text": "good night"}'


Response: 
{
  "text": "good night",
  "neighbors": [
    {"label": "nox bona", "distance": 0.11},
    {"label": "salve mundi", "distance": 0.37},
    {"label": "te amo", "distance": 0.44}
  ],
  "k": 3,
  "note": "Top-K nearest labeled examples. Increase k for more alternatives."
}


Minimal Frontend

When running serve, open http://localhost:8080
 in your browser.
A simple UI is included where you can type text, click Classify, and see JSON results.

Environment

Set your OpenAI API key before running:

export OPENAI_API_KEY=sk-xxxxx

Optionally create a .env file:

OPENAI_API_KEY=sk-xxxxx

Template Task Config (YAML)

task: "translate english text to latin"
count: 50
output: latin.json


Run With
./training-camp generate --config task.yaml

Project Layout

training-camp/
├── cmd/          # CLI commands (generate, embed, train, eval, serve)
├── internal/     # internal packages
│   ├── ml        # model + KNN logic
│   └── openai    # embedding + API helpers
├── go.mod
├── go.sum
└── README.md


Use Cases

Translate or classify text into narrow domains

Watch authenticity checks

Incident triage

Ticket classification

Any small supervised task with text

Roadmap

X OpenAI generation + embedding

X Offline training & eval

X Serve predictions via HTTP

X Batch prediction endpoint

X Top-K neighbor inspection

 Multi-label classification

 Export predictions in batch

 Config-based fine-tuning presets


License

MIT
